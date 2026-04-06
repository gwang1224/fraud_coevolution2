"""
Synthetic **fraud** FAST-payment sequences backed by Neo4j.

Loads victims, fraudsters, and action definitions from the graph; alternates fraudster and
victim steps; uses an LLM (Ollama) to pick among legal next actions and Neo4j ``NEXT`` edge
weights for diversity. Applies compromise/terminal transfer rules in simulation.

Finished sequences can be checked with ``validators.fraud_validator.validate_one_sequence``
before acceptance (see ``generate_multiple_sequences``). CLI supports ``--output`` and
``FRAUD_SEQUENCES_OUT`` for JSON export paths.

Requires: Neo4j, ``python-dotenv``, ``requests``, and Ollama for LLM calls.
"""

from neo4j import GraphDatabase
import argparse
import json
import os
import random
import sys
import requests
from datetime import datetime
from dotenv import load_dotenv
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Optional
from pathlib import Path

# Running as `python .../build_fraud_sequence.py` does not put `src` on sys.path
_src = Path(__file__).resolve().parents[1]
if str(_src) not in sys.path:
    sys.path.insert(0, str(_src))

from validators.fraud_validator import validate_one_sequence


load_dotenv()

NEO4J_URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USERNAME = os.getenv("NEO4J_USERNAME", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "password")


@dataclass
class Action:
    """Represents an action in the sequence"""
    entity1: str
    action: str 
    entity2: str
    channel: str


@dataclass
class EnvState:
    """Represents the current environment state"""
    victim: str
    fraudster: str
    victim_account: str 
    current_actor_type: str
    current_actor_id: str
    accounts: Dict[str, Dict]
    compromised: bool 
    history: List[Action] = field(default_factory=list)
    terminal: bool = False

OLLAMA_EVAL_URL = os.getenv("OLLAMA_URL", "http://localhost:11434/api/generate")
OLLAMA_EVAL_MODEL = os.getenv("OLLAMA_MODEL", "llama3.2")

_FLAGS_PROMPT = """\
You are a fraud analyst. Given this FAST-payment sequence, list the distinct \
fraud tactics / red-flag categories present. Use short snake_case labels \
(e.g. phishing, credential_compromise, social_engineering, fast_transfer, \
malware, sim_swap, account_takeover, payment_redirect, impersonation).

Sequence:
{sequence_text}

Return ONLY a JSON array of strings, e.g. ["phishing","credential_compromise"].
No explanation, no markdown fences, no extra text.
"""

_DETECTION_PROMPT = """\
You are a bank's real-time fraud detection system analyzing a FAST-payment sequence.

Rate how likely this sequence is fraudulent on a scale from 0.0 (certainly legitimate) \
to 1.0 (certainly fraudulent). Consider:
- Presence of social engineering, phishing, credential theft, etc.
- Whether the money transfer is authorized or coerced/stolen.
- How sophisticated the attack is (harder to detect = lower score).

Sequence:
{sequence_text}

Return ONLY a single decimal number between 0.0 and 1.0. \
No explanation, no extra text.
"""


def _llm_extract_flags(sequence_text: str, max_attempts: int = 3) -> List[str]:
    """Ask the LLM to identify fraud tactic flags in a sequence."""
    prompt = _FLAGS_PROMPT.format(sequence_text=sequence_text)
    for _ in range(max_attempts):
        try:
            resp = requests.post(
                OLLAMA_EVAL_URL,
                json={
                    "model": OLLAMA_EVAL_MODEL,
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


def _llm_detection_score(sequence_text: str, num_calls: int = 3) -> float:
    """Run multiple LLM calls asking for a 0-1 fraud probability, return the mean."""
    scores: List[float] = []
    prompt = _DETECTION_PROMPT.format(sequence_text=sequence_text)
    for _ in range(num_calls):
        try:
            resp = requests.post(
                OLLAMA_EVAL_URL,
                json={
                    "model": OLLAMA_EVAL_MODEL,
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
    return round(sum(scores) / len(scores), 2) if scores else 0.5


class Neo4jSequenceBuilder:
    """Class for building sequences from Neo4j graph"""
    
    def __init__(self, uri=None, user=None, password=None):
        """
        Initialize Neo4j connection
        
        Args:
            uri: Neo4j URI (defaults to env var or localhost)
            user: Neo4j username (defaults to env var or 'neo4j')
            password: Neo4j password (defaults to env var or 'password')
        """
        self.uri = uri or NEO4J_URI
        self.user = user or NEO4J_USERNAME
        self.password = password or NEO4J_PASSWORD
        
        self.driver = GraphDatabase.driver(self.uri, auth=(self.user, self.password))
        self.driver.verify_connectivity()
        print(f"Connected to Neo4j at {self.uri}")
    
    def close(self):
        """Close the driver connection when done."""
        self.driver.close()
    
    def get_possible_actions(self, current_actor_type, current_actor_id):
        """
        Deterministic function for retrieving possible next actions
        Given (current_actor_type, current_actor_id, current_state) → return list of possible actions.
        
        Args:
            current_actor_type: Type of actor ('victim' or 'fraudster')
            current_actor_id: ID/name of the current actor
            current_state: Current state information (dict)
        
        Returns:
            List of possible actions
        """
        if current_actor_type == "victim":
            # Match victim performing action - don't need to match target fraudster (causes duplicates)
            query = """
                MATCH (v:victims {name: $victim_name})-[:PERFORMS]->(a:action)
                RETURN DISTINCT a.action AS action,
                       a.channels AS channels,
                       a.is_terminal AS is_terminal,
                       a.target AS target,
                       a.description AS description,
                       a.compromises_account AS compromises_account
            """
        else:
            # Match fraudster performing action - don't need to match target victim (causes duplicates)
            query = """
                MATCH (f:fraudsters {name: $fraudster_name})-[:PERFORMS]->(a:action)
                RETURN DISTINCT a.action AS action,
                       a.channels AS channels,
                       a.is_terminal AS is_terminal,
                       a.target AS target,
                       a.description AS description,
                       a.compromises_account AS compromises_account
            """
        result, summary, keys = self.driver.execute_query(
            query,
            victim_name=current_actor_id,
            fraudster_name=current_actor_id,
            database_="neo4j",
        )
        return result

    def pick_victim(self) -> Optional[Dict]:
        """
        STEP 1: Pick a victim from Neo4j graph
        
        Returns:
            Dict with victim name and account info, or None if no victims found
        """
        query = """
            MATCH (v:victims)-[:OWNS]->(a:account)
            RETURN v.name AS victim_name, 
                   a.acc_name AS account_name,
                   a.bank AS bank,
                   a.balance AS balance
            LIMIT 100
        """
        result, summary, keys = self.driver.execute_query(
            query,
            database_="neo4j",
        )
        
        if result:
            victim = random.choice(result)
            bal = victim.get("balance")
            try:
                bal_f = float(bal) if bal is not None else 0.0
            except (TypeError, ValueError):
                bal_f = 0.0
            acc = victim.get("account_name") or ""
            if not acc:
                name = victim.get("victim_name") or "unknown"
                acc = f"acc_{name}"
            return {
                "victim_name": victim["victim_name"],
                "account_name": acc,
                "bank": victim.get("bank") or "",
                "balance": bal_f,
            }
        return None

    def pick_fraudster(self) -> Optional[str]:
        """
        Pick a random fraudster from Neo4j graph
        
        Returns:
            Fraudster name or None if no fraudsters found
        """
        query = """
            MATCH (f:fraudsters)-[:OWNS]->(a:account)
            RETURN f.name AS fraudster_name,
                   a.acc_name AS account_name,
                   a.bank AS bank,
                   a.balance AS balance
            LIMIT 100
        """
        result, summary, keys = self.driver.execute_query(
            query,
            database_="neo4j",
        )
        
        if result:
            fraudster = random.choice(result)
            return {
                "fraudster_name": fraudster["fraudster_name"],
                "account_name": fraudster["account_name"],
                "bank": fraudster["bank"],
                "balance": fraudster["balance"]
            }
        return None

    def create_edge(self, current_action: str, next_action: str):
        """
        Create or update an edge between current_action and next_action.
        If edge exists, increment weight by 1. Otherwise, create with weight = 1.
        If action nodes don't exist, they will be created.
        """
        if not current_action or not next_action:
            print(f"Warning: Cannot create edge - current_action={current_action}, next_action={next_action}")
            return
        
        query = """
            MERGE (a:action {action: $current_action})
            MERGE (b:action {action: $next_action})
            MERGE (a)-[r:NEXT]->(b)
            ON CREATE SET r.weight = 1
            ON MATCH SET r.weight = r.weight + 1
            RETURN r.weight AS weight, a.action AS current_action, b.action AS next_action
        """
        try:
            result, summary, keys = self.driver.execute_query(
                query,
                current_action=current_action,
                next_action=next_action,
                database_="neo4j",
            )
            if result:
                print(f"Edge: {result[0]['current_action']} -> {result[0]['next_action']} (weight: {result[0]['weight']})")
            else:
                print(f"Warning: No result returned for edge between {current_action} and {next_action}")
        except Exception as e:
            print(f"Error creating edge: {e}")

    def get_weight(self, current_action: str, next_action: str) -> int:
        """
        Get the weight of the edge between current_action and next_action.
        Returns 0 if edge doesn't exist.
        """
        if not current_action or not next_action:
            return 0
        
        query = """
            MATCH (a:action {action: $current_action})-[r:NEXT]->(b:action {action: $next_action})
            RETURN r.weight AS weight
        """
        result, summary, keys = self.driver.execute_query(
            query,
            current_action=current_action,
            next_action=next_action,
            database_="neo4j",
        )
        return result[0]["weight"] if result else 0
    
    def get_weights_for_actions(self, current_action:str, possible_actions) -> Dict[str, int]:
        """
        Get weights for all possible next actions
        """
        weights = {}

        for action in possible_actions:
            weight = self.get_weight(current_action, action.get("action", ""))
            weights[action.get("action", "")] = weight
        
        return weights
            

    def initialize_state(self, fraudster_name: str, victim_name: str, victim_account: str, account_balance: float) -> EnvState:
        """
        STEP 2: Initialize an environment state
        
        Args:
            fraudster_name: Name of the fraudster
            victim_name: Name of the victim
            victim_account: Name of the victim's account
            account_balance: Initial account balance
        
        Returns:
            Initialized EnvState
        """
        accounts = {
            victim_account: {
                "owner": victim_name,
                "balance": account_balance,
                "compromised": False
            }
        }
        
        state = EnvState(
            victim=victim_name,
            fraudster=fraudster_name,
            victim_account=victim_account,
            current_actor_type="fraudster",
            current_actor_id=fraudster_name,
            accounts=accounts,
            compromised=False,
            history=[],
            terminal=False
        )
        
        return state

    def filter_repeated_actions(self, actions: List[Dict], history: List[Action]) -> List[Dict]:
        """
        Filter out actions that have been repeated in history
        
        Args:
            actions: List of possible actions
            history: History of previous actions
        
        Returns:
            Filtered list of actions
        """
        if not history:
            return actions
        
        # Get action names from history
        history_action_names = {action.action for action in history}
        
        # Filter out actions that match history
        filtered = []
        for action in actions:
            action_name = action.get("action", "")
            if action_name not in history_action_names:
                filtered.append(action)
        
        return filtered if filtered else actions

    def is_account_compromised(self, state: EnvState) -> bool:
        """
        Check if the victim's account is compromised
        
        Args:
            state: Current environment state
            victim_name: Name of the victim
        
        Returns:
            True if account is compromised, False otherwise
        """
        return state.compromised

    def filter_money_transfer_actions(self, actions: List[Dict], state: EnvState) -> List[Dict]:
        """
        Filter actions based on account compromise status:
        - If account is compromised: only return actions where is_terminal = true
        - If account is NOT compromised: allow all actions EXCEPT transfer_money
          (to allow actions that can compromise the account)
        
        Args:
            actions: List of possible actions
            state: Current environment state
        
        Returns:
            Filtered list of actions
        """
        if state.compromised:
            return [item for item in actions if item.get("is_terminal") is True]
        else:
            return [item for item in actions if item.get("is_terminal") is False]

    def choose_action(self, state: EnvState, possible_actions: List[Dict]) -> Optional[Dict]:
        """
        Use Ollama LLM to choose the best action from possible actions, considering edge weights for diversity
        
        Args:
            state: Current environment state
            possible_actions: List of possible actions to choose from
        
        Returns:
            Selected action dict or None if LLM fails
        """
        # Get current action from history (last action)
        current_action = None
        if state.history:
            current_action = state.history[-1].action
        
        # Get weights for all possible next actions
        action_weights = self.get_weights_for_actions(current_action, possible_actions)
        
        # Format state summary
        acc_info = state.accounts.get(state.victim_account, {})
        try:
            bal_f = float(acc_info.get("balance") or 0)
        except (TypeError, ValueError):
            bal_f = 0.0

        # Format history as a readable list
        if state.history:
            history_lines = []
            for i, action in enumerate(state.history, 1):
                history_lines.append(f"  {i}. {action.entity1} -> {action.action} -> {action.entity2} (via {action.channel})")
            history_text = "\n".join(history_lines)
        else:
            history_text = "  (No actions taken yet)"
        
        state_summary = f"""
            Current State:
            - Victim: {state.victim}
            - Fraudster: {state.fraudster}
            - Victim Account: {state.victim_account}
            - Account Balance: ${bal_f:.2f}
            - Account Compromised: {state.compromised}
            - Current Actor: {state.current_actor_type} ({state.current_actor_id})
            - Action History:
            {history_text}
        """
        
        # Format actions with their weights (frequency of use)
        # Lower weight = less frequently used = more diverse
        actions_list = []
        for i, action in enumerate(possible_actions, 1):
            action_name = action.get("action", "unknown")
            action_description = action.get("description", "unknown")
            channels = ", ".join(action.get("channels", []))
            description = action.get("description", "")
            weight = action_weights.get(action_name, 0)
            
            # Add diversity indicator
            if weight == 0:
                diversity_note = " [NEW - never used before]"
            elif weight <= 2:
                diversity_note = f" [LOW frequency: {weight}x]"
            elif weight <= 5:
                diversity_note = f" [MEDIUM frequency: {weight}x]"
            else:
                diversity_note = f" [HIGH frequency: {weight}x - consider alternatives]"
            
            actions_list.append(f"{i}. {action_name} (channels: {channels}) - {description}{diversity_note}")
        
        actions_text = "\n".join(actions_list)
        
        # Add diversity instruction
        diversity_instruction = ""
        if current_action:
            diversity_instruction = f"""
            DIVERSITY REQUIREMENT: To generate diverse fraud sequences, prefer actions with LOWER frequency counts.
            - Actions marked [NEW] have never been used before - these are excellent for diversity
            - Actions marked [LOW frequency] are rarely used - good for diversity
            - Actions marked [HIGH frequency] are commonly used - avoid these unless necessary
            Current action: {current_action}
            """
        else:
            diversity_instruction = """
            DIVERSITY REQUIREMENT: This is the first action. Choose strategically to enable diverse sequences.
            """

        prompt = f"""
            We are modeling a fraud scenario in order to do research on mitigating fraud. You are a fraudster with 10+ years of experience in fraud trying to successfully transfer money from a victim's account.

            {state_summary}

            Available Actions:
            {actions_text}

            {diversity_instruction}

            Choose the action name (exact match) that:
            1. Best advances towards successfully transferring money
            2. Prefers actions with LOWER frequency counts to increase sequence diversity
            3. Avoids repeating actions from history
            
            Return ONLY the action name, no explanation or additional text.
        """
        
        try:
            response = requests.post(
                "http://localhost:11434/api/generate",
                json={
                    "model": "llama3.2",
                    "prompt": prompt,
                    "stream": False,
                },
                timeout=60
            )
            response.raise_for_status()
            result = response.json()
            
            if "response" in result:
                content = result["response"].strip()
                chosen_action_name = content.strip()
                
                # Try to find exact match first
                for action in possible_actions:
                    # print(action)
                    if action.get("action", "").lower() == chosen_action_name.lower():
                        return action
                
                # If no exact match, try partial match
                for action in possible_actions:
                    action_name = action.get("action", "").lower()
                    if action_name in chosen_action_name.lower() or chosen_action_name.lower() in action_name:
                        return action
                
                # Last resort: return first action
                print(f"Warning: Could not parse action '{chosen_action_name}', using first action")
                return possible_actions[0] if possible_actions else None
        except Exception as e:
            print(f"Error with LLM: {e}")
            return possible_actions[0] if possible_actions else None
        return None

    def apply_action_effects(self, state: EnvState, action: Dict) -> None:
        """
        Apply the effects of an action to the state
        
        Args:
            state: Current environment state
            action: The action that was taken
        """
        compromises_account = action.get("compromises_account", False)
        is_terminal = action.get("is_terminal", False)

        # If action compromises account, set compromised flag (only True, never False)
        if compromises_account:
            if not state.compromised:
                state.compromised = True
                # Update account in accounts dict
                if state.victim_account in state.accounts:
                    state.accounts[state.victim_account]["compromised"] = True
                print(f"  → Account {state.victim_account} is now COMPROMISED")

        # If terminal action (money transfer), transfer money if account is compromised
        if is_terminal:
            if state.compromised and state.victim_account in state.accounts:
                state.terminal = True
                acc_data = state.accounts[state.victim_account]
                transfer_amount = random.uniform(100, min(1000, acc_data["balance"] * 0.1))
                acc_data["balance"] = max(0, acc_data["balance"] - transfer_amount)
                print(f"  → Money transferred: ${transfer_amount:.2f} from {state.victim_account}")
                print(f"  → Remaining balance: ${acc_data['balance']:.2f}")
            else:
                print(f"  → Cannot transfer money: Account {state.victim_account} is not compromised")

    def generate_sequence(self, max_steps: int = 10) -> tuple:
        """
        Generate a complete fraud sequence.
        
        Args:
            max_steps: Maximum number of steps before stopping
        
        Returns:
            (history: List[Action], state: EnvState, initial_balance: float)
        """
        victim_info = self.pick_victim()
        if not victim_info:
            print("No victims found in graph")
            return [], EnvState("", "", "", "", "", {}, False), 0.0
        
        print(f"Selected victim: {victim_info['victim_name']}")
        
        fraudster_info = self.pick_fraudster()
        if not fraudster_info:
            print("No fraudsters found in graph")
            return [], EnvState("", "", "", "", "", {}, False), 0.0
        
        print(f"Selected fraudster: {fraudster_info['fraudster_name']}")
        
        try:
            bal_init = float(victim_info["balance"])
        except (TypeError, ValueError, KeyError):
            bal_init = 0.0
        state = self.initialize_state(
            fraudster_info["fraudster_name"],
            victim_info["victim_name"],
            victim_info["account_name"],
            bal_init,
        )

        print(
            f"Initialized state with account: {victim_info['account_name']} (balance: ${bal_init})"
        )
        
        # STEP 3: Loop until terminal or max steps
        step = 0

        while not state.terminal and step < max_steps:
            step += 1
            print(f"\n--- Step {step} ---")
            print(f"Current actor: {state.current_actor_type} ({state.current_actor_id})")
            
            # 1. Query Neo4j for all Action nodes reachable from current_actor_id
            possible_actions_raw = self.get_possible_actions(
                state.current_actor_type,
                state.current_actor_id
            )

            try:
                current_action = state.history[-1].action
            except IndexError:
                current_action = None
            print("Current action: ", current_action)

            if not possible_actions_raw:
                print("No more actions available")
                break
            
            # Convert to list of dicts
            possible_actions = []
            for record in possible_actions_raw:
                action_dict = {
                    "action": record.get("action", ""),
                    "channels": record.get("channels", []),
                    "is_terminal": record.get("is_terminal", False),
                    "target": record.get("target", ""),
                    "description": record.get("description", ""),
                    "compromises_account": record.get("compromises_account", False)
                }  
                possible_actions.append(action_dict)
            
            # 2. Filter out repeated actions
            filtered_actions = self.filter_repeated_actions(possible_actions, state.history)
            
            # 3. Filter out money transfer actions if account is not compromised
            filtered_actions = self.filter_money_transfer_actions(filtered_actions, state)
            if not filtered_actions:
                print("No valid actions available")
                break
            
            print(f"Available actions: {len(filtered_actions)}")
            
            # 3. Use LLM to choose action
            chosen_action = self.choose_action(state, filtered_actions)
            next_action = chosen_action.get("action", "")
            print("Next action: ", next_action)

            if not chosen_action:
                print("LLM failed to choose action")
                break

            # Create or update an edge between current_action and next_action
            self.create_edge(current_action, next_action)
            current_action = next_action
            
            action_name = chosen_action.get("action", "")
            channels = chosen_action.get("channels", [])
            channel = random.choice(channels) if channels else "unknown"
            
            print(f"LLM chose: {action_name} via {channel}")
            
            # 4. Create Action object and add to history
            if state.current_actor_type == "fraudster":
                entity1 = state.current_actor_id
                entity2 = state.victim
            else:
                entity1 = state.current_actor_id
                entity2 = state.fraudster
            
            action_obj = Action(
                entity1=entity1,
                action=action_name,
                entity2=entity2,
                channel=channel
            )
            
            state.history.append(action_obj)
            
            # 5. Apply action effects
            self.apply_action_effects(state, chosen_action)
            
            # 6. Switch actor if needed (alternate between fraudster and victim)
            if state.current_actor_type == "fraudster":
                state.current_actor_type = "victim"
                state.current_actor_id = state.victim
            else:
                state.current_actor_type = "fraudster"
                state.current_actor_id = state.fraudster
            
            # 7. Check terminal condition
            if state.terminal:
                print(f"\n✓ Terminal condition reached: Money successfully transferred!")
                break
        
        if step >= max_steps:
            print(f"\nReached maximum steps ({max_steps})")
        
        return state.history, state, bal_init

    def evaluate_sequence(
        self,
        history: List[Action],
        state: EnvState,
        initial_balance: float,
    ) -> Dict:
        """
        Build the ``evaluation`` block for a completed fraud sequence.

        Derives every field from the simulation state.  ``detection_score`` and
        ``flags_triggered`` are both produced by LLM calls so they adapt to
        novel action vocabularies without hardcoded keyword lists.
        """
        fraud_success = state.terminal and state.compromised
        current_balance = 0.0
        if state.victim_account in state.accounts:
            current_balance = state.accounts[state.victim_account].get("balance", 0.0)
        payout = max(0.0, initial_balance - current_balance) if fraud_success else 0.0

        if state.terminal:
            terminal_reason = "fraud_transfer_completed"
        elif not state.compromised:
            terminal_reason = "account_not_compromised"
        else:
            terminal_reason = "max_steps_reached"

        flat = [
            {"entity1": a.entity1, "action": a.action, "entity2": a.entity2, "channel": a.channel}
            for a in history
        ]
        label, _ = validate_one_sequence(flat)

        sequence_text = "\n".join(
            f"{i}. {a.entity1} -> {a.action} -> {a.entity2} (via {a.channel})"
            for i, a in enumerate(history, 1)
        )

        return {
            "valid": label == "valid",
            "fraud_success": fraud_success,
            "payout": round(payout, 2),
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
            "flags_triggered": _llm_extract_flags(sequence_text),
            "detection_score": _llm_detection_score(sequence_text)
        }


    @staticmethod
    def _format_step(step_num: int, action: Action, actor_type: str) -> Dict:
        """Format one Action into the rich step dict."""
        target_type = "victim" if actor_type == "fraudster" else "fraudster"
        return {
            "step": step_num,
            "actor_type": actor_type,
            "actor_id": action.entity1,
            "action": action.action,
            "target_type": target_type,
            "target_id": action.entity2,
            "channel": action.channel,
        }

    def generate_one_sequence(self, max_steps: int = 10) -> Dict:
        """
        Generate, validate, and evaluate a single fraud sequence.

        Returns a dict matching the coevolution schema::

            {
              "label": "fraud",
              "sequence": [ {step…}, … ],
              "evaluation": { … }
            }
        """
        while True:
            history, state, initial_balance = self.generate_sequence(max_steps=max_steps)
            if not history:
                print("Empty sequence, retrying…")
                continue

            evaluation = self.evaluate_sequence(history, state, initial_balance)
            audit = self.audit_sequence(history)
            if not evaluation["valid"]:
                print("\nInvalid sequence, retrying…")
                continue

            steps: List[Dict] = []
            actor_type = "fraudster"
            for i, action in enumerate(history, 1):
                steps.append(self._format_step(i, action, actor_type))
                actor_type = "victim" if actor_type == "fraudster" else "fraudster"

            return {
                "label": "fraud",
                "sequence": steps,
                "evaluation": evaluation,
                "audit": audit,
            }

    def generate_multiple_sequences(self, count: int, max_steps: int = 10) -> List[Dict]:
        """
        Generate *count* validated fraud sequences with evaluation metadata.

        Each entry uses the rich step format from ``generate_one_sequence``.
        """
        sequences: List[Dict] = []

        for i in range(count):
            print(f"\n{'=' * 60}")
            print(f"Generating sequence {i + 1}/{count}")
            print(f"{'=' * 60}")
            try:
                entry = self.generate_one_sequence(max_steps=max_steps)
                entry["sequence_id"] = i + 1
                entry["timestamp"] = datetime.now().isoformat()
                sequences.append(entry)
                print(f"✓ Sequence {i + 1} complete ({entry['evaluation']['num_steps']} steps)")
            except Exception as e:
                print(f"✗ Error generating sequence {i + 1}: {e}")

        return sequences

    def save_sequences_to_file(self, sequences: List[Dict], filename: str = "sequences.json") -> None:
        """Save sequences to a JSON file with metadata."""
        total_steps = sum(
            s.get("evaluation", {}).get("num_steps", len(s.get("sequence", [])))
            for s in sequences
        )
        avg_steps = total_steps / len(sequences) if sequences else 0

        output = {
            "metadata": {
                "total_sequences": len(sequences),
                "generated_at": datetime.now().isoformat(),
                "total_steps": total_steps,
                "average_steps_per_sequence": round(avg_steps, 2),
            },
            "sequences": sequences,
        }
        
        with open(filename, "w") as f:
            json.dump(output, f, indent=2)
        
        print(f"\n✓ Saved {len(sequences)} sequences to {filename}")
        print(f"  - Total steps: {total_steps}")
        print(f"  - Average steps per sequence: {avg_steps:.2f}")

    

def main():
    parser = argparse.ArgumentParser(description="Generate fraud sequences")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(os.getenv("FRAUD_SEQUENCES_OUT", "fraud_sequences.json")),
        help="Output JSON path (default: FRAUD_SEQUENCES_OUT or fraud_sequences.json)",
    )
    args = parser.parse_args()

    builder = Neo4jSequenceBuilder()
    
    try:
        result = builder.generate_one_sequence(max_steps=10)
        print(json.dumps({"0": result}, indent=2))

        args.output.parent.mkdir(parents=True, exist_ok=True)
        builder.save_sequences_to_file([result], filename=str(args.output))
    finally:
        builder.close()

if __name__ == "__main__":
    main()