"""
LLM **fraud-narrative** validator for saved sequence JSON.

Reads a file with top-level ``sequences`` (each with ``actions``: ``entity1``, ``action``,
``entity2``, ``channel``). Asks Ollama for a single token ``valid`` or ``invalid`` against
rules for realistic scam progression. Optionally runs a second pass to explain invalid rows.

Writes:
- ``output/valid_sequences_bank.json`` (default) — accepted sequences plus metadata.
- ``output/invalid_sequences_bank.json`` (default) — rejected / unclassified rows.

Defaults resolve relative to the **repository root** (two levels above this file).

Requires: ``requests``, ``python-dotenv``, Ollama.
"""

import argparse
import json
import os
from pathlib import Path
from typing import Optional, Tuple

import requests
from dotenv import load_dotenv

load_dotenv()

# Project root (…/fraud_coevolution2), not src/validators — so default paths match output/
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_DEFAULT_INPUT = _PROJECT_ROOT / "output" / "fraud_sequences_100.json"
_DEFAULT_INVALID_BANK = _PROJECT_ROOT / "output" / "invalid_sequences_bank.json"
_DEFAULT_VALID_BANK = _PROJECT_ROOT / "output" / "valid_sequences_bank.json"

OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434/api/generate")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.2")

PROMPT_TEMPLATE = """
You are a fraud analyst validating whether a sequence of actions represents a realistic FAST-payment fraud scenario.

Rules for a valid fraud sequence:
1. The sequence should follow a logical progression (recon / contact / manipulation toward theft, etc.).
2. A money-moving or terminal transfer step should not appear out of nowhere: there should be some prior step that plausibly explains why the victim would pay or move funds (e.g. phishing, impersonation, social engineering, credential or OTP disclosure, malware, or similar — not necessarily a literal action named "compromise").
3. Actions should mostly connect in a believable order for a scam narrative.
4. If transitions are impossible or contradictory, mark invalid.

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
You are a fraud analyst. This FAST-payment sequence was judged INVALID (not a realistic fraud narrative).

Sequence:
{input_sequence}

Rules the validator used (for reference):
1. Logical progression toward theft.
2. Transfers should not appear without prior plausible manipulation (phishing, impersonation, credential/OTP disclosure, etc.).
3. Actions should connect believably.
4. Contradictory or impossible transitions → invalid.

Task: In 2–5 short sentences, explain specifically why this sequence fails as a realistic fraud scenario. Reference concrete steps (entity names, action names) and which expectations above are violated.
Do not repeat the word "invalid" as a verdict. Do not suggest fixes. Plain text only, no JSON.
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
    """First line, first token, lowercased, punctuation stripped."""
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


def validate_one_sequence(
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

    for attempt in range(max_attempts):
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


def explain_invalid_sequence(
    actions: list,
    *,
    max_attempts: int = 3,
    timeout_s: int = 120,
) -> Tuple[str, str]:
    """
    Second-pass explanation after an 'invalid' verdict.
    Returns (explanation_text, raw_model_response).
    """
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
                # Single paragraph preferred; trim excess
                text = last_raw.split("\n\n")[0].strip()
                return text, last_raw
        except Exception as e:
            last_raw = f"<error: {e}>"

    return "", last_raw


def main():
    parser = argparse.ArgumentParser(description="LLM-validate fraud sequences JSON")
    parser.add_argument(
        "--input",
        type=Path,
        default=_DEFAULT_INPUT,
        help="JSON file with top-level 'sequences' list",
    )
    parser.add_argument(
        "--output-invalid-bank",
        type=Path,
        default=_DEFAULT_INVALID_BANK,
        help="Where to write invalid / unclassified sequences",
    )
    parser.add_argument(
        "--output-valid-bank",
        type=Path,
        default=_DEFAULT_VALID_BANK,
        help="Where to write sequences judged valid",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print each prompt and response",
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
    valid_sequence_bank = []

    for sequence in sequences:
        sid = sequence.get("sequence_id", "?")
        print(f"Validating sequence {sid}")
        actions = sequence.get("actions") or []
        label, raw = validate_one_sequence(actions)

        if args.verbose:
            print("---------------------------------")
            print(f"sequence_id={sid} -> {label!r} raw={raw!r}")

        if label == "valid":
            valid_sequences += 1
            valid_entry = {**sequence, "label": "valid", "validator_output": raw}
            valid_sequence_bank.append(valid_entry)
        elif label == "invalid":
            invalid_sequences += 1
            entry = {
                "sequence_id": sid,
                "label": "invalid",
                "validator_output": raw,
                "actions": actions,
            }
            if do_explain:
                reason, reason_raw = explain_invalid_sequence(actions)
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
    args.output_valid_bank.parent.mkdir(parents=True, exist_ok=True)

    with open(args.output_invalid_bank, "w", encoding="utf-8") as f:
        json.dump(
            {
                "metadata": {
                    "validator": "fraud",
                    "total_input": len(sequences),
                    "invalid": invalid_sequences,
                    "unclassified": unclassified,
                },
                "sequences": invalid_sequence_bank,
            },
            f,
            indent=2,
        )
    with open(args.output_valid_bank, "w", encoding="utf-8") as f:
        json.dump(
            {
                "metadata": {
                    "validator": "fraud",
                    "total_input": len(sequences),
                    "valid": valid_sequences,
                },
                "sequences": valid_sequence_bank,
            },
            f,
            indent=2,
        )

    print(f"Valid sequences: {valid_sequences}")
    print(f"Invalid sequences: {invalid_sequences}")
    if unclassified:
        print(f"Unclassified (after retries): {unclassified}")
    print(f"Saved valid bank to: {args.output_valid_bank}")
    print(f"Saved invalid / unclassified bank to: {args.output_invalid_bank}")


if __name__ == "__main__":
    main()
