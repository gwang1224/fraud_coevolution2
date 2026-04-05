"""
Use an LLM (Ollama) to extract **high-level fraud behavioral patterns** from a labeled dataset.

Input JSON must be a dict keyed by sequence id, each value containing ``label`` (``fraud`` or
``legit``) and ``sequence`` (list of step dicts). The prompt contrasts fraud vs legit examples
and asks for a semicolon-separated table of abstract patterns (not copied literals).

Typical input: ``output/evaluation/tp_fraud_for_pattern_mining.json``, produced by
``src.detectors.llm_fraud_detector`` (true-positive fraud + all ground-truth legit from the
same run). You can also pass the full coevolution dataset.

CLI: ``python -m src.analysis.extract_fraud_behavior_patterns --input ... --max-patterns 5``

Requires: ``requests``, running Ollama.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any, Dict

import requests

OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434/api/generate")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.2")

PROMPT_TEMPLATE = """
YOUR TASK

Identify {max_patterns} distinct, high-level behavioral patterns that occur in fraudulent sequences and do NOT occur in legitimate sequences.

These patterns should describe general strategies, tactics, or behavioral structures—not specific entities or specific wording from examples.

Fraudulent sequences:
{fraud_seqs}

Legitimate sequences:
{legit_seqs}

You must:
• Extract generalizable fraud patterns (e.g., "phishing to obtain credentials," "account takeover followed by unauthorized transfer," "social engineering leading to information disclosure").
• Ensure none of the patterns are present in legitimate sequences.
• Avoid sequence-specific details.
• Ensure each pattern describes behavior, not a literal line from the dataset.

Format:
Return the identified patterns in a semicolon-separated file (similar to a CSV) as shown in the example,
with pattern_number and pattern_name columns:
semicolon-separated file
pattern_number;pattern_name
1;pattern 1
2;pattern 2
3;pattern 3
Only return the semicolon-separated list, no comments or explanation needed.
"""


def extract_fraud_behavior_patterns(
    dataset_path: Path,
    max_patterns: int,
    *,
    ollama_url: str = OLLAMA_URL,
    model: str = OLLAMA_MODEL,
    timeout_s: int = 120,
) -> str:
    """
    Load a coevolution-style JSON dict and return the model's pattern list (raw text).
    """
    with open(dataset_path, encoding="utf-8") as f:
        data: Dict[str, Any] = json.load(f)

    formatted_fraud = {k: v for k, v in data.items() if v.get("label") == "fraud"}
    formatted_legit = {k: v for k, v in data.items() if v.get("label") == "legit"}

    prompt = PROMPT_TEMPLATE.format(
        fraud_seqs=formatted_fraud,
        legit_seqs=formatted_legit,
        max_patterns=max_patterns,
    )

    response = requests.post(
        ollama_url,
        json={"model": model, "prompt": prompt, "stream": False},
        timeout=timeout_s,
    )
    response.raise_for_status()
    return (response.json().get("response") or "").strip()


def main() -> None:
    repo = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(
        description="Extract fraud-only behavioral patterns from a labeled sequence dataset via LLM."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=repo / "output" / "evaluation" / "tp_fraud_for_pattern_mining.json",
        help="Coevolution JSON (dict: id -> label, sequence). Default: detector TP-fraud + GT-legit subset.",
    )
    parser.add_argument("--max-patterns", type=int, default=5)
    parser.add_argument("--model", default=OLLAMA_MODEL)
    args = parser.parse_args()

    text = extract_fraud_behavior_patterns(
        args.input,
        args.max_patterns,
        model=args.model,
    )
    print(text)


if __name__ == "__main__":
    main()
