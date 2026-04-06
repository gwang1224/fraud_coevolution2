"""
Synthetic **legitimate** banking sequences for classification datasets.

Picks a customer from Neo4j (``victims`` / accounts), loads the action catalog from
``data/graph/victim_actions.json`` under ``legit_actions``, samples scenario categories, and uses
an LLM to choose steps. Reuses ``Neo4jSequenceBuilder`` from ``build_fraud_sequence`` only for
connectivity, ``pick_victim``, and ``NEXT``-edge weighting—not for fraud actions.

Emits the same per-step shape as fraud sequences (``entity1``, ``action``, ``entity2``,
``channel``) with ``label: "legit"``. Optional balance adjustments for payment-like actions.

Requires: Neo4j, Ollama, ``python-dotenv``, ``requests``.
"""

from __future__ import annotations

import json
import os
import random
import requests
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from dotenv import load_dotenv

from src.generators.build_fraud_sequence import Action, Neo4jSequenceBuilder

load_dotenv()

OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434/api/generate")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.2")

_DEFAULT_VICTIM_ACTIONS_PATH = (
    Path(__file__).resolve().parents[2] / "data" / "graph" / "victim_actions.json"
)

# ---------------------------------------------------------------------------
# LLM-driven evaluation helpers (mirror of build_fraud_sequence approach)
# ---------------------------------------------------------------------------

_LEGIT_FLAGS_PROMPT = """\
You are a banking behavior analyst. Given this FAST-payment sequence of \
legitimate customer activity, list the distinct activity categories / tags \
present. Use short snake_case labels \
(e.g. bill_payment, balance_check, peer_transfer, recurring_payment, \
rent_payment, savings, direct_debit, payroll, subscription, new_payee).

Sequence:
{sequence_text}

Return ONLY a JSON array of strings, e.g. ["bill_payment","balance_check"].
No explanation, no markdown fences, no extra text.
"""

_LEGIT_DETECTION_PROMPT = """\
You are a bank's real-time fraud detection system analyzing a FAST-payment sequence.

Rate how suspicious this sequence looks on a scale from 0.0 (clearly legitimate, \
no red flags) to 1.0 (highly suspicious, looks fraudulent). Consider:
- Whether all actions are ordinary authorized banking tasks.
- Whether amounts, recipients, and channels look normal.
- Whether the ordering of actions is natural for a real customer.

Sequence:
{sequence_text}

Return ONLY a single decimal number between 0.0 and 1.0. \
No explanation, no extra text.
"""


def _llm_extract_legit_flags(sequence_text: str, max_attempts: int = 3) -> List[str]:
    """Ask the LLM to identify activity categories in a legit sequence."""
    prompt = _LEGIT_FLAGS_PROMPT.format(sequence_text=sequence_text)
    for _ in range(max_attempts):
        try:
            resp = requests.post(
                OLLAMA_URL,
                json={
                    "model": OLLAMA_MODEL,
                    "prompt": prompt,
                    "stream": False,
                    "options": {"temperature": 0, "num_predict": 120, "stop": ["\n\n"]},
                },
                timeout=60,
            )
            resp.raise_for_status()
            raw = (resp.json().get("response") or "").strip()
            start = raw.find("[")
            end = raw.rfind("]")
            if start != -1 and end != -1:
                flags = json.loads(raw[start : end + 1])
                return sorted(set(str(f).strip().lower().replace(" ", "_") for f in flags if f))
        except Exception:
            continue
    return []


def _llm_legit_detection_score(sequence_text: str, num_calls: int = 3) -> float:
    """Run multiple LLM calls asking for a 0-1 suspicion score, return the mean."""
    scores: List[float] = []
    prompt = _LEGIT_DETECTION_PROMPT.format(sequence_text=sequence_text)
    for _ in range(num_calls):
        try:
            resp = requests.post(
                OLLAMA_URL,
                json={
                    "model": OLLAMA_MODEL,
                    "prompt": prompt,
                    "stream": False,
                    "options": {"temperature": 0.3, "num_predict": 12, "stop": ["\n"]},
                },
                timeout=60,
            )
            resp.raise_for_status()
            raw = (resp.json().get("response") or "").strip()
            for token in raw.replace(",", " ").split():
                try:
                    val = float(token)
                    if 0.0 <= val <= 1.0:
                        scores.append(val)
                        break
                except ValueError:
                    continue
        except Exception:
            continue
    return round(sum(scores) / len(scores), 2) if scores else 0.1


def load_legit_actions_from_victim_actions_json(
    path: Optional[Path] = None,
) -> Tuple[List[Dict], Tuple[str, ...]]:
    """
    Load legit catalog from victim_actions.json under key \"legit_actions\".
    Normalizes to dicts with: action, channels, description, category, is_payment, target.
    \"target\" becomes entity2 in emitted Action rows (from JSON only).
    Returns (actions, sorted unique category strings for scenario sampling).
    """
    p = Path(path or os.getenv("VICTIM_ACTIONS_JSON", _DEFAULT_VICTIM_ACTIONS_PATH))
    with open(p, encoding="utf-8") as f:
        data = json.load(f)
    raw = data.get("legit_actions")
    if not raw:
        raise ValueError(
            f"{p} must contain a non-empty \"legit_actions\" array for legit generation."
        )
    rows: List[Dict] = []
    for item in raw:
        name = item.get("name")
        if not name:
            continue
        ch = item.get("common_channels") or []
        cat = item.get("category") or ""
        tgt = (item.get("target") or "").strip()
        if not tgt:
            raise ValueError(
                f'{p}: legit_actions entry "{name}" must include non-empty "target" '
                "(used as entity2 in sequences)."
            )
        terminal = item.get("is_terminal") or ""
        rows.append(
            {
                "action": name,
                "channels": list(ch),
                "description": item.get("description", ""),
                "category": cat,
                "is_payment": bool(item.get("is_payment", False)),
                "target": tgt,
                "is_terminal": terminal
            }
        )
    if not rows:
        raise ValueError(f"{p}: legit_actions parsed to zero usable entries.")
    categories = tuple(sorted({r["category"] for r in rows if r["category"]}))
    if not categories:
        raise ValueError(f"{p}: legit_actions entries need a \"category\" field.")
    return rows, categories


@dataclass
class LegitEnvState:
    user_name: str
    user_account: str
    balance: float
    history: List[Action] = field(default_factory=list)
    terminal: bool = False
    scenario_tags: Tuple[str, ...] = field(default_factory=tuple)


class Neo4jLegitSequenceBuilder:
    """
    Builds legitimate behavior sequences using the same Action schema and
    Neo4j edge-weight diversity as Neo4jSequenceBuilder (shared fraud builder).
    """

    def __init__(
        self,
        uri=None,
        user=None,
        password=None,
        *,
        victim_actions_json: Optional[Path] = None,
    ):
        self._graph = Neo4jSequenceBuilder(uri, user, password)
        self._legit_actions, self._scenario_categories = (
            load_legit_actions_from_victim_actions_json(victim_actions_json)
        )

    def close(self) -> None:
        self._graph.close()

    def pick_user(self) -> Optional[Dict]:
        return self._graph.pick_victim()

    def _random_scenario_tags(self) -> Tuple[str, ...]:
        tags = self._scenario_categories
        if len(tags) <= 1:
            return tags
        k = random.randint(2, min(4, len(tags)))
        return tuple(sorted(set(random.sample(list(tags), k))))

    def _allowed_actions_for_tags(
        self, scenario_tags: Tuple[str, ...]
    ) -> List[Dict]:
        """Prefer actions matching chosen scenario tags; always allow balance checks."""
        tagged = [a for a in self._legit_actions if a["category"] in scenario_tags]
        balance_fallback = [
            a for a in self._legit_actions if a["category"] == "check_balance"
        ]
        merged = tagged + [a for a in balance_fallback if a not in tagged]
        random.shuffle(merged)
        return merged

    def filter_repeated_actions(
        self, actions: List[Dict], history: List[Action]
    ) -> List[Dict]:
        if not history:
            return actions
        seen = {h.action for h in history}
        filtered = [a for a in actions if a["action"] not in seen]
        return filtered if filtered else actions

    def choose_action_with_llm(
        self, state: LegitEnvState, possible_actions: List[Dict]
    ) -> Optional[Dict]:
        current_action = state.history[-1].action if state.history else None
        weights = self._graph.get_weights_for_actions(current_action, possible_actions)

        if state.history:
            history_lines = [
                f"  {i}. {a.entity1} -> {a.action} -> {a.entity2} (via {a.channel})"
                for i, a in enumerate(state.history, 1)
            ]
            history_text = "\n".join(history_lines)
        else:
            history_text = "  (No actions yet)"

        scenario_text = ", ".join(state.scenario_tags)
        actions_list = []
        for i, action in enumerate(possible_actions, 1):
            name = action.get("action", "unknown")
            desc = action.get("description", "")
            ch = ", ".join(action.get("channels", []))
            w = weights.get(name, 0)
            if w == 0:
                note = " [NEW - never used before]"
            elif w <= 2:
                note = f" [LOW frequency: {w}x]"
            elif w <= 5:
                note = f" [MEDIUM frequency: {w}x]"
            else:
                note = f" [HIGH frequency: {w}x - consider alternatives]"
            actions_list.append(f"{i}. {name} (channels: {ch}) - {desc}{note}")
        actions_text = "\n".join(actions_list)

        div = (
            f"DIVERSITY: Prefer LOWER-frequency actions. Previous step: {current_action}."
            if current_action
            else "DIVERSITY: First step — pick something natural that fits the scenario mix."
        )

        prompt = f"""
You are simulating ORDINARY, AUTHORIZED retail banking behavior for research on payment fraud detection.
The customer is doing legitimate tasks only (bills, rent, friends, recurring savings, balance checks).

Customer: {state.user_name}
Account: {state.user_account}
Balance: ${state.balance:.2f}
Intended scenario mix (cover these naturally over the session): {scenario_text}

History:
{history_text}

Available actions:
{actions_text}

{div}

Choose exactly ONE action name from the list above (exact spelling).
Return ONLY the action name, nothing else.
"""

        try:
            response = requests.post(
                OLLAMA_URL,
                json={"model": OLLAMA_MODEL, "prompt": prompt, "stream": False},
                timeout=60,
            )
            response.raise_for_status()
            result = response.json()
            raw = (result.get("response") or "").strip()
            chosen = raw.splitlines()[0].strip() if raw else ""

            for a in possible_actions:
                if a.get("action", "").lower() == chosen.lower():
                    return a
            for a in possible_actions:
                an = a.get("action", "").lower()
                if an in chosen.lower() or chosen.lower() in an:
                    return a
            print(f"Warning: could not parse legit action '{chosen}', using first action")
            return possible_actions[0] if possible_actions else None
        except Exception as e:
            print(f"Error with LLM (legit): {e}")
            return possible_actions[0] if possible_actions else None

    def apply_action_effects(self, state: LegitEnvState, action: Dict) -> None:
        if not action.get("is_payment"):
            return
        # Small, plausible outflows; keep balance positive when possible.
        cap = max(20.0, min(800.0, state.balance * 0.15))
        amt = random.uniform(15.0, cap)
        state.balance = max(0.0, state.balance - amt)

    def _make_action_object(
        self, user: str, chosen: Dict
    ) -> Action:
        name = chosen.get("action", "")
        channels = chosen.get("channels") or ["app"]
        channel = random.choice(channels)
        entity2 = chosen.get("target") or ""
        return Action(
            entity1=user,
            action=name,
            entity2=entity2,
            channel=channel,
        )

    def generate_sequence(self, max_steps: int = 10) -> Tuple[List[Action], "LegitEnvState", float]:
        """
        Generate a complete legit sequence.

        Returns:
            (history, state, initial_balance)
        """
        user_info = self.pick_user()
        if not user_info:
            print("No users found in graph (victims dataset).")
            empty_state = LegitEnvState(user_name="", user_account="", balance=0.0)
            return [], empty_state, 0.0

        user_name = user_info["victim_name"]
        account = user_info["account_name"]
        balance = float(user_info["balance"] or 0.0)
        initial_balance = balance
        scenario_tags = self._random_scenario_tags()

        state = LegitEnvState(
            user_name=user_name,
            user_account=account,
            balance=balance,
            scenario_tags=scenario_tags,
        )

        hi = min(max_steps, 8)
        lo = min(3, hi) if hi >= 1 else 1
        target_len = random.randint(lo, hi) if hi >= lo else hi

        step = 0
        while step < max_steps and not state.terminal:
            step += 1
            pool = self._allowed_actions_for_tags(state.scenario_tags)
            pool = self.filter_repeated_actions(pool, state.history)

            if not pool:
                break

            chosen = self.choose_action_with_llm(state, pool)
            if not chosen:
                break

            prev_name = state.history[-1].action if state.history else None
            next_name = chosen.get("action", "")
            self._graph.create_edge(prev_name, next_name)

            action_obj = self._make_action_object(user_name, chosen)
            state.history.append(action_obj)
            self.apply_action_effects(state, chosen)

            if len(state.history) >= target_len:
                state.terminal = True

            if chosen.get("is_terminal"):
                state.terminal = True

        return state.history, state, initial_balance
    
    def evaluate_sequence(
        self,
        history: List[Action],
        state: "LegitEnvState",
        initial_balance: float,
    ) -> Dict:
        """
        Build the ``evaluation`` block for a completed legit sequence.

        ``detection_score`` and ``flags_triggered`` are both LLM-driven so they
        adapt to novel action vocabularies without hardcoded keyword lists.
        """
        total_spent = max(0.0, initial_balance - state.balance)

        if state.terminal:
            terminal_reason = "session_complete"
        else:
            terminal_reason = "max_steps_reached"

        sequence_text = "\n".join(
            f"{i}. {a.entity1} -> {a.action} -> {a.entity2} (via {a.channel})"
            for i, a in enumerate(history, 1)
        )

        return {
            "valid": True,
            "fraud_success": False,
            "payout": 0.0,
            "num_steps": len(history),
            "terminal_reason": terminal_reason,
        }
    
    def audit_sequence(self, history: List[Action]) -> Dict:
        """
        Audit a sequence with LLM
        """
        sequence_text = "\n".join(
            f"{i}. {a.entity1} -> {a.action} -> {a.entity2} (via {a.channel})"
            for i, a in enumerate(history, 1)
        )
        return {
            "flags_triggered": _llm_extract_legit_flags(sequence_text),
            "detection_score": _llm_legit_detection_score(sequence_text)
        }

    @staticmethod
    def _format_step(step_num: int, action: Action) -> Dict:
        """Format one Action into the rich step dict."""
        return {
            "step": step_num,
            "actor_type": "customer",
            "actor_id": action.entity1,
            "action": action.action,
            "target_type": "payee",
            "target_id": action.entity2,
            "channel": action.channel,
        }

    def generate_one_sequence(self, max_steps: int = 10) -> Dict:
        """
        Generate and evaluate a single legit sequence.

        Returns a dict matching the coevolution schema::

            {
              "label": "legit",
              "sequence": [ {step…}, … ],
              "evaluation": { … }
            }
        """
        while True:
            history, state, initial_balance = self.generate_sequence(max_steps=max_steps)
            if not history:
                print("Empty legit sequence, retrying…")
                continue

            evaluation = self.evaluate_sequence(history, state, initial_balance)
            audit = self.audit_sequence(history)

            steps: List[Dict] = []
            for i, action in enumerate(history, 1):
                steps.append(self._format_step(i, action))

            return {
                "label": "legit",
                "sequence": steps,
                "evaluation": evaluation,
                "audit": audit,
            }

    def generate_multiple_sequences(
        self, count: int, max_steps: int = 10
    ) -> List[Dict]:
        """Generate *count* legit sequences with evaluation metadata."""
        sequences: List[Dict] = []
        for i in range(count):
            print(f"\n{'=' * 60}")
            print(f"Generating legit sequence {i + 1}/{count}")
            print(f"{'=' * 60}")
            try:
                entry = self.generate_one_sequence(max_steps=max_steps)
                entry["sequence_id"] = i + 1
                entry["timestamp"] = datetime.now().isoformat()
                sequences.append(entry)
                print(f"✓ Legit sequence {i + 1} complete ({entry['evaluation']['num_steps']} steps)")
            except Exception as e:
                print(f"✗ Error generating legit sequence {i + 1}: {e}")
        return sequences

    def save_sequences_to_file(
        self, sequences: List[Dict], filename: str = "legit_sequences.json"
    ) -> None:
        """Save sequences to a JSON file with metadata."""
        total_steps = sum(
            s.get("evaluation", {}).get("num_steps", len(s.get("sequence", [])))
            for s in sequences
        )
        avg_steps = total_steps / len(sequences) if sequences else 0

        output = {
            "metadata": {
                "dataset": "legit",
                "total_sequences": len(sequences),
                "generated_at": datetime.now().isoformat(),
                "total_steps": total_steps,
                "average_steps_per_sequence": round(avg_steps, 2),
            },
            "sequences": sequences,
        }
        with open(filename, "w") as f:
            json.dump(output, f, indent=2)
        print(f"\n✓ Saved {len(sequences)} legit sequences to {filename}")
        print(f"  - Total steps: {total_steps}")
        print(f"  - Average steps per sequence: {avg_steps:.2f}")


def main():
    builder = Neo4jLegitSequenceBuilder()
    try:
        print("Generating legit banking sequences...")
        result = builder.generate_one_sequence()
        print(json.dumps({"0": result}, indent=2))

        out = os.getenv("LEGIT_SEQUENCES_OUT", "output/legit_sequences.json")
        os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
        builder.save_sequences_to_file([result], filename=out)
    finally:
        builder.close()


if __name__ == "__main__":
    main()
