"""
LLM-based validator for legitimate / authorized FAST-payment-style sequences.

Mirrors fraud_validator.py: binary valid/invalid verdict, optional explanation
for failures, CLI over JSON with a top-level \"sequences\" list.
"""

import argparse
import json
import os
from pathlib import Path
from typing import Optional, Tuple

import requests
from dotenv import load_dotenv

load_dotenv()

_ROOT = Path(__file__).resolve().parents[2]
_DEFAULT_INPUT = _ROOT / "output" / "legit_sequences_100.json"
_DEFAULT_INVALID_BANK = _ROOT / "output" / "invalid_legit_sequences_bank.json"

OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434/api/generate")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.2")

PROMPT_TEMPLATE = """
You are a retail banking analyst validating whether a sequence of actions represents
realistic AUTHORIZED, benign customer behavior (not fraud).

Rules for a valid legitimate sequence:
1. Actions should look like ordinary banking: balance checks, bill pay, P2P to known payees,
   rent, small utilities, recurring transfers, verified new payees, etc.
2. Money movement should be plausible in context (e.g. not a sudden \"safe account\" or
   coerced transfer; no phishing / impersonation / credential theft narrative).
3. Steps should connect in a believable order for a normal session.
4. If the sequence reads like a scam, takeover, or impossible jumps, mark invalid.

Sequence to evaluate:
{input_sequence}

Your task:
Return exactly one word and nothing else.
Output must be either:
valid
or
invalid

Do not explain your answer.
Do not add punctuation.
Do not add any other words.
"""

EXPLAIN_INVALID_PROMPT = """
You are a retail banking analyst. This sequence was judged INVALID as a legitimate
(normal, authorized) banking session.

Sequence:
{input_sequence}

Rules the validator used (for reference):
1. Ordinary authorized behavior (bills, rent, friends, balance checks, recurring transfers).
2. No fraud-style coercion, phishing, or \"move money to a safe account\" narratives.
3. Coherent, plausible order for a benign session.
4. Contradictory or impossible transitions → invalid.

Task: In 2–5 short sentences, explain specifically why this sequence does NOT read as
realistic legitimate banking. Reference concrete steps (entity names, action names) and
which expectations above are violated.
Do not repeat the word \"invalid\" as a verdict. Do not suggest fixes. Plain text only, no JSON.
"""


def _format_actions(actions: list) -> str:
    lines = []
    for index, action in enumerate(actions):
        lines.append(
            f"{index + 1}. {action['entity1']} - {action['action']} -> "
            f"{action['entity2']} via {action['channel']}"
        )
    return "\n".join(lines)


def _normalize_validator_token(raw: str) -> str:
    if not raw:
        return ""
    line = raw.splitlines()[0].strip()
    token = line.split()[0] if line.split() else ""
    token = token.strip().strip("\"'").strip(".,:;!?()[]{}")
    return token.lower()


def _parse_valid_invalid(raw: str) -> Optional[str]:
    t = _normalize_validator_token(raw)
    if t == "valid" or t.startswith("valid"):
        return "valid"
    if t == "invalid" or t.startswith("invalid"):
        return "invalid"
    return None


def validate_one_legit_sequence(
    actions: list,
    *,
    max_attempts: int = 5,
    timeout_s: int = 120,
) -> Tuple[Optional[str], str]:
    """
    Returns (label, raw_last_response) where label is 'valid', 'invalid', or None.
    """
    input_sequence = ""
    for index, action in enumerate(actions):
        input_sequence += (
            f"{index + 1}. {action['entity1']} - {action['action']} -> "
            f"{action['entity2']} via {action['channel']}\n"
        )

    prompt = PROMPT_TEMPLATE.format(input_sequence=input_sequence)
    last_raw = ""

    for _ in range(max_attempts):
        try:
            response = requests.post(
                OLLAMA_URL,
                json={
                    "model": OLLAMA_MODEL,
                    "prompt": prompt,
                    "stream": False,
                    "options": {
                        "temperature": 0,
                        "num_predict": 8,
                        "stop": ["\n"],
                    },
                },
                timeout=timeout_s,
            )
            response.raise_for_status()
            last_raw = (response.json().get("response") or "").strip()
            label = _parse_valid_invalid(last_raw)
            if label is not None:
                return label, last_raw
            prompt += (
                "\nYour previous output was not accepted: "
                + repr(last_raw)
                + "\nReply with exactly one word: valid or invalid.\n"
            )
        except Exception as e:
            last_raw = f"<error: {e}>"
            continue

    return None, last_raw


def explain_invalid_legit_sequence(
    actions: list,
    *,
    max_attempts: int = 3,
    timeout_s: int = 120,
) -> Tuple[str, str]:
    """Second-pass explanation after an 'invalid' verdict for legit labeling."""
    input_sequence = _format_actions(actions)
    prompt = EXPLAIN_INVALID_PROMPT.format(input_sequence=input_sequence)
    last_raw = ""

    for _ in range(max_attempts):
        try:
            response = requests.post(
                OLLAMA_URL,
                json={
                    "model": OLLAMA_MODEL,
                    "prompt": prompt,
                    "stream": False,
                    "options": {
                        "temperature": 0,
                        "num_predict": 400,
                    },
                },
                timeout=timeout_s,
            )
            response.raise_for_status()
            last_raw = (response.json().get("response") or "").strip()
            if last_raw:
                text = last_raw.split("\n\n")[0].strip()
                return text, last_raw
        except Exception as e:
            last_raw = f"<error: {e}>"

    return "", last_raw


def main():
    parser = argparse.ArgumentParser(
        description="LLM-validate legitimate banking sequences JSON"
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=_DEFAULT_INPUT,
        help="JSON file with top-level 'sequences' list (e.g. label legit)",
    )
    parser.add_argument(
        "--output-invalid-bank",
        type=Path,
        default=_DEFAULT_INVALID_BANK,
        help="Where to write invalid / unclassified sequences",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print each verdict and explanation",
    )
    parser.add_argument(
        "--no-explain",
        action="store_true",
        help="Skip second-pass explanation for invalid sequences (faster)",
    )
    args = parser.parse_args()
    do_explain = not args.no_explain

    with open(args.input, encoding="utf-8") as f:
        sequences = json.load(f)["sequences"]

    valid_sequences = 0
    invalid_sequences = 0
    unclassified = 0
    invalid_sequence_bank = []

    for sequence in sequences:
        sid = sequence.get("sequence_id", "?")
        print(f"Validating sequence {sid}")
        actions = sequence.get("actions") or []
        label, raw = validate_one_legit_sequence(actions)

        if args.verbose:
            print("---------------------------------")
            print(f"sequence_id={sid} -> {label!r} raw={raw!r}")

        if label == "valid":
            valid_sequences += 1
        elif label == "invalid":
            invalid_sequences += 1
            entry = {
                "sequence_id": sid,
                "label": "invalid_legit",
                "validator_output": raw,
                "actions": actions,
            }
            if do_explain:
                reason, reason_raw = explain_invalid_legit_sequence(actions)
                entry["invalid_reason"] = reason
                entry["invalid_reason_raw"] = reason_raw
                if args.verbose:
                    print(f"  explain: {reason!r}")
            invalid_sequence_bank.append(entry)
        else:
            unclassified += 1
            entry = {
                "sequence_id": sid,
                "label": "unclassified",
                "validator_output": raw,
                "actions": actions,
            }
            if do_explain:
                entry["invalid_reason"] = (
                    "Verdict could not be parsed as valid/invalid after retries; "
                    "see validator_output."
                )
            invalid_sequence_bank.append(entry)

    args.output_invalid_bank.parent.mkdir(parents=True, exist_ok=True)
    out_payload = {
        "metadata": {
            "validator": "legit",
            "total_input": len(sequences),
            "valid": valid_sequences,
            "invalid": invalid_sequences,
            "unclassified": unclassified,
        },
        "sequences": invalid_sequence_bank,
    }
    with open(args.output_invalid_bank, "w", encoding="utf-8") as f:
        json.dump(out_payload, f, indent=2)

    print(f"Valid sequences: {valid_sequences}")
    print(f"Invalid sequences: {invalid_sequences}")
    if unclassified:
        print(f"Unclassified (after retries): {unclassified}")
    print(f"Saved invalid / unclassified bank to: {args.output_invalid_bank}")


if __name__ == "__main__":
    main()
