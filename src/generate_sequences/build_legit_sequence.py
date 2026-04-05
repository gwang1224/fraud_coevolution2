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

from src.generate_sequences.build_fraud_sequence import Action, Neo4jSequenceBuilder

load_dotenv()

OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434/api/generate")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.2")

_DEFAULT_VICTIM_ACTIONS_PATH = (
    Path(__file__).resolve().parents[2] / "data" / "graph" / "victim_actions.json"
)


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

    def generate_sequence(self, max_steps: int = 10) -> Tuple[List[Action], Tuple[str, ...]]:
        user_info = self.pick_user()
        if not user_info:
            print("No users found in graph (victims dataset).")
            return [], ()

        user_name = user_info["victim_name"]
        account = user_info["account_name"]
        balance = float(user_info["balance"] or 0.0)
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

        return state.history, scenario_tags
    
    def generate_one_sequence(
            self, max_steps: int=10
    ):
        actions, scenario_tags = self.generate_sequence(max_steps=max_steps)
        return [
                    {
                        "entity1": a.entity1,
                        "action": a.action,
                        "entity2": a.entity2,
                        "channel": a.channel,
                    }
                    for a in actions
                ]

    def generate_multiple_sequences(
        self, count: int, max_steps: int = 10
    ) -> List[Dict]:
        sequences: List[Dict] = []
        for i in range(count):
            print(f"\n{'=' * 60}")
            print(f"Generating legit sequence {i + 1}/{count}")
            print(f"{'=' * 60}")
            try:
                actions, scenario_tags = self.generate_sequence(max_steps=max_steps)
                seq_dict = [
                    {
                        "entity1": a.entity1,
                        "action": a.action,
                        "entity2": a.entity2,
                        "channel": a.channel,
                    }
                    for a in actions
                ]
                sequences.append(
                    {
                        "sequence_id": i + 1,
                        "label": "legit",
                        "timestamp": datetime.now().isoformat(),
                        "scenario_tags": list(scenario_tags),
                        "actions": seq_dict,
                        "action_count": len(actions),
                        "successful": len(actions) > 0,
                    }
                )
                print(f"✓ Legit sequence {i + 1} completed: {len(actions)} actions")
            except Exception as e:
                print(f"✗ Error generating legit sequence {i + 1}: {e}")
                sequences.append(
                    {
                        "sequence_id": i + 1,
                        "label": "legit",
                        "timestamp": datetime.now().isoformat(),
                        "scenario_tags": [],
                        "actions": [],
                        "action_count": 0,
                        "successful": False,
                        "error": str(e),
                    }
                )
        return sequences

    def save_sequences_to_file(
        self, sequences: List[Dict], filename: str = "legit_sequences.json"
    ) -> None:
        output = {
            "metadata": {
                "dataset": "legit",
                "total_sequences": len(sequences),
                "generated_at": datetime.now().isoformat(),
                "successful_sequences": sum(1 for s in sequences if s.get("successful")),
                "total_actions": sum(s.get("action_count", 0) for s in sequences),
                "average_actions_per_sequence": (
                    sum(s.get("action_count", 0) for s in sequences) / len(sequences)
                    if sequences
                    else 0
                ),
            },
            "sequences": sequences,
        }
        with open(filename, "w") as f:
            json.dump(output, f, indent=2)
        print(f"\n✓ Saved {len(sequences)} legit sequences to {filename}")


def main():
    builder = Neo4jLegitSequenceBuilder()
    try:
        print("Generating legit banking sequences...")
        sequences = builder.generate_one_sequence()
        print(sequences)
        # out = os.getenv("LEGIT_SEQUENCES_OUT", "output/legit_sequences_100.json")
        # os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
        # builder.save_sequences_to_file(sequences, filename=out)
        
    finally:
        builder.close()


if __name__ == "__main__":
    main()
