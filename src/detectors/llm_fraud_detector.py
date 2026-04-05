"""
Binary **fraud vs legit** classifier for FAST-payment-style sequences using Ollama.

Loads a coevolution JSON dict (id → ``label``, ``sequence``), renders each sequence as text,
and runs an ensemble of LLM calls. Writes error and summary CSVs.

Also writes **``pattern_mining_json``** (default under ``output/evaluation/``): ground-truth
**fraud** sequences that the model classified as **fraud** (true positives), plus **all**
ground-truth **legit** sequences from the same dataset. That file matches the shape expected by
``src.analysis.extract_fraud_behavior_patterns`` (contrast fraud vs legit for pattern mining).

Run from repo root::

    python -m src.detectors.llm_fraud_detector --dataset output/datasets/coev_dataset1.json

Requires: ``requests``, ``pandas``, Ollama.
"""

from __future__ import annotations

import argparse
import json
import os
import textwrap
from collections import Counter
from pathlib import Path
from typing import List, Optional, Tuple

import pandas as pd
import requests

OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434/api/generate")


class LLMDetector:
    """
    Ollama-based fraud vs legit classifier for rendered sequence strings.
    """

    def __init__(self, coev_file_path: str, model: str):
        self.coev_file_path = coev_file_path
        self.model = model

    def classify_sequence(self, seq: str, max_attempts: int = 5, timeout_s: int = 60) -> Optional[str]:
        prompt = (
            textwrap.dedent(
                f"""\
        You are a strict binary classifier for FAST-payment sequences.

        Your job: label the *behavior* described in the sequence.
        - "fraud" = unauthorized / deceptive activity leading to an improper transfer (phishing, impersonation, credential theft, account takeover, SIM swap, coercion, etc.)
        - "legit" = ordinary, authorized payments for goods/services, bills, tuition, rent, reimbursements, etc.

        IMPORTANT:
        - Do NOT treat this as a "system log" task.
        - The presence of the words "phishing", "fraudulent", "malware", etc. indicates FRAUD behavior (not "legit input").

        Output rules (MUST follow):
        - Output exactly ONE word: fraud OR legit
        - No punctuation, no quotes, no extra words, no explanations.

        Examples:
        Sequence:
        action(alice, consultation, bob, email, agreed on terms)
        action(bob, payment, alice, app, paid for consultation)
        transaction(acc_bob, fast payment, acc_alice, 200.00)
        Output: legit

        Sequence:
        action(scammr, phishing, bob, email, sent fake login link)
        action(bob, entered_credentials, scammr, website, shared password)
        action(scammr, account_takeover, acc_bob, online, gained access)
        transaction(acc_bob, fast payment, acc_scammr, 2000.00)
        Output: fraud

        Input sequence:
        {seq}

        Output:
        """
            ).strip()
            + "\n"
        )

        attempts = 0
        while attempts < max_attempts:
            attempts += 1
            try:
                response = requests.post(
                    OLLAMA_URL,
                    json={
                        "model": self.model,
                        "prompt": prompt,
                        "stream": False,
                        "options": {
                            "temperature": 0,
                            "num_predict": 4,
                            "stop": ["\n"],
                        },
                    },
                    timeout=timeout_s,
                )
                response.raise_for_status()
                payload = response.json()
            except Exception:
                continue

            raw = (payload.get("response") or "").strip()
            raw = raw.splitlines()[0].strip() if raw else ""
            token = raw.split()[0] if raw.split() else ""
            token = token.strip().strip("\"'").strip(".,:;!?()[]{}")
            token = token.lower()

            if token in {"fraud", "fraudulent"} or token.startswith("fraud"):
                return "fraud"
            if token in {"legit", "legitimate"} or token.startswith("legit"):
                return "legit"

            prompt += (
                "\nYour previous output was invalid: "
                + raw
                + "\nRemember: output exactly one word: fraud OR legit.\nOutput:\n"
            )

        return None

    def ensemble_classify_sequence(self, seq: str, num_calls: int = 5) -> Tuple[Optional[str], List[Optional[str]], float, float]:
        labels: List[Optional[str]] = [self.classify_sequence(seq) for _ in range(num_calls)]
        valid = [x for x in labels if x in ("fraud", "legit")]
        valid_rate = len(valid) / num_calls if num_calls else 0.0
        if not valid:
            return None, labels, 0.0, valid_rate
        winner, count = Counter(valid).most_common(1)[0]
        stability = count / len(valid)
        return winner, labels, stability, valid_rate

    def explain_classification(self, seq: str, result: str) -> str:
        prompt = (
            f"Explain why you classified this sequence as {result}.\n\n"
            f"Input:\n{seq}\n\nOutput:"
        )
        response = requests.post(
            OLLAMA_URL,
            json={
                "model": self.model,
                "prompt": prompt,
                "stream": False,
                "options": {"temperature": 0},
            },
            timeout=60,
        )
        response.raise_for_status()
        return (response.json().get("response") or "").strip().strip("'")

    def run_detector(
        self,
        *,
        errors_csv: Optional[Path] = None,
        results_csv: Optional[Path] = None,
        pattern_mining_json: Optional[Path] = None,
    ) -> Tuple[float, float, float, float]:
        """
        Evaluate all sequences in ``self.coev_file_path``; write CSV summaries.

        If ``pattern_mining_json`` is set, write a coevolution-shaped JSON dict: every
        ground-truth legit row, plus only fraud rows where the ensemble predicted fraud.
        """
        errors_csv = errors_csv or Path("output/evaluation/llm_detector_errors.csv")
        results_csv = results_csv or Path("output/evaluation/llm_detector_results.csv")
        errors_csv.parent.mkdir(parents=True, exist_ok=True)
        results_csv.parent.mkdir(parents=True, exist_ok=True)

        error_seq: list = []
        res: list = []

        num_correct = 0
        num_legit_correct = 0
        num_fraud_correct = 0
        false_pos = 0
        false_neg = 0
        total_seq = 0
        unclassifiable = 0
        n_fraud_labels = 0
        n_legit_labels = 0
        predictions: dict = {}

        with open(self.coev_file_path, encoding="utf-8") as f:
            data = json.load(f)

        for sid in data:
            total_seq += 1
            print(f"Classifying sequence {sid}.")
            label = data[sid]["label"]
            sequence = data[sid]["sequence"]
            if label == "fraud":
                n_fraud_labels += 1
            elif label == "legit":
                n_legit_labels += 1

            sequence_text = ""
            for step, action in enumerate(sequence):
                sequence_text += (
                    f"Step {step + 1}: {action['entity1']} performs {action['action']} "
                    f"to {action['entity2']} via {action['channel']}\n"
                )

            classification, labels, stability, valid_rate = self.ensemble_classify_sequence(
                sequence_text
            )
            predictions[sid] = classification

            if classification == label:
                num_correct += 1
                if classification == "fraud":
                    num_fraud_correct += 1
                if classification == "legit":
                    num_legit_correct += 1
            else:
                if classification is None:
                    unclassifiable += 1
                elif classification == "fraud" and label == "legit":
                    false_pos += 1
                elif classification == "legit" and label == "fraud":
                    false_neg += 1
                else:
                    unclassifiable += 1
                error_seq.append(
                    {
                        "Sequence id": sid,
                        "Sequence": sequence,
                        "Label": label,
                        "LLM Generated Label": classification,
                        "Stability": stability,
                        "Valid rate": valid_rate,
                        "Votes": labels,
                    }
                )

        df_error = pd.DataFrame(error_seq)
        print(df_error)
        df_error.to_csv(errors_csv, index=False)

        sens = num_fraud_correct / n_fraud_labels if n_fraud_labels else 0.0
        spec = num_legit_correct / n_legit_labels if n_legit_labels else 0.0
        prec_denom = num_fraud_correct + false_pos
        prec = num_fraud_correct / prec_denom if prec_denom else 0.0

        res.append(
            {
                "Accuracy": num_correct / total_seq if total_seq else 0.0,
                "Sensitivity": sens,
                "Specificity": spec,
                "Precision": prec,
                "False positive": false_pos / total_seq if total_seq else 0.0,
                "False negative": false_neg / total_seq if total_seq else 0.0,
                "Unclassifiable": unclassifiable / total_seq if total_seq else 0.0,
            }
        )
        df_res = pd.DataFrame(res)
        print(df_res)
        df_res.to_csv(results_csv, index=False)

        if pattern_mining_json is not None:
            pattern_mining_json = Path(pattern_mining_json)
            pattern_mining_json.parent.mkdir(parents=True, exist_ok=True)
            subset: dict = {}
            for sid, row in data.items():
                if row.get("label") == "legit":
                    subset[sid] = {
                        "label": "legit",
                        "sequence": row.get("sequence", []),
                    }
            for sid, row in data.items():
                if row.get("label") == "fraud" and predictions.get(sid) == "fraud":
                    subset[sid] = {
                        "label": "fraud",
                        "sequence": row.get("sequence", []),
                    }
            with open(pattern_mining_json, "w", encoding="utf-8") as out:
                json.dump(subset, out, indent=2)
            n_tp = sum(
                1
                for sid, row in data.items()
                if row.get("label") == "fraud" and predictions.get(sid) == "fraud"
            )
            n_legit_out = sum(1 for sid, row in data.items() if row.get("label") == "legit")
            print(
                f"Wrote pattern-mining subset ({n_tp} TP fraud, {n_legit_out} GT legit) to "
                f"{pattern_mining_json}"
            )

        return (
            num_correct / total_seq if total_seq else 0.0,
            false_pos / total_seq if total_seq else 0.0,
            false_neg / total_seq if total_seq else 0.0,
            unclassifiable / total_seq if total_seq else 0.0,
        )


def main() -> None:
    repo = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description="Ensemble LLM fraud vs legit evaluation")
    parser.add_argument(
        "--dataset",
        type=Path,
        default=repo / "output" / "datasets" / "coev_dataset1.json",
        help="Coevolution JSON (dict: id -> label, sequence)",
    )
    parser.add_argument("--model", default=os.getenv("OLLAMA_MODEL", "llama3.2"))
    parser.add_argument(
        "--errors-csv",
        type=Path,
        default=repo / "output" / "evaluation" / "llm_detector_errors.csv",
    )
    parser.add_argument(
        "--results-csv",
        type=Path,
        default=repo / "output" / "evaluation" / "llm_detector_results.csv",
    )
    parser.add_argument(
        "--pattern-mining-json",
        type=Path,
        default=repo / "output" / "evaluation" / "tp_fraud_for_pattern_mining.json",
        help="Coevolution-shaped JSON: GT legit + fraud only where model said fraud (for extract_fraud_behavior_patterns).",
    )
    parser.add_argument(
        "--no-pattern-mining",
        action="store_true",
        help="Skip writing the pattern-mining JSON file.",
    )
    args = parser.parse_args()

    pm_path: Optional[Path] = None if args.no_pattern_mining else args.pattern_mining_json

    detector = LLMDetector(str(args.dataset), args.model)
    detector.run_detector(
        errors_csv=args.errors_csv,
        results_csv=args.results_csv,
        pattern_mining_json=pm_path,
    )


if __name__ == "__main__":
    main()
