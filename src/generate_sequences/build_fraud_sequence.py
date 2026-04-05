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

    def choose_action_with_llm(self, state: EnvState, possible_actions: List[Dict]) -> Optional[Dict]:
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

    def generate_sequence(self, max_steps: int = 10) -> List[Action]:
        """
        STEP 3: Generate a complete fraud sequence
        
        Args:
            max_steps: Maximum number of steps before stopping
        
        Returns:
            List of Action objects representing the sequence
        """
        # STEP 1: Pick a victim
        victim_info = self.pick_victim()
        if not victim_info:
            print("No victims found in graph")
            return []
        
        print(f"Selected victim: {victim_info['victim_name']}")
        
        # Pick a fraudster
        fraudster_info = self.pick_fraudster()
        if not fraudster_info:
            print("No fraudsters found in graph")
            return []
        
        print(f"Selected fraudster: {fraudster_info['fraudster_name']}")
        
        # STEP 2: Initialize state
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
            chosen_action = self.choose_action_with_llm(state, filtered_actions)
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
        
        return state.history
    
    def generate_one_sequence(self, max_steps: int = 10) -> List[Action]:
        """
        Generate and validate a single sequence

        Args:
            max_steps: Maximum steps per sequence
        Returns:
            List of Action objects representing the sequence
        """
        valid_sequence = False
        
        while not valid_sequence:
            sequence = self.generate_sequence(max_steps=max_steps)

            sequence_dict = [
                    {
                        "entity1": action.entity1,
                        "action": action.action,
                        "entity2": action.entity2,
                        "channel": action.channel
                    }
                    for action in sequence
                ]

            label, reason = validate_one_sequence(sequence_dict)
            
            if label == "valid":
                valid_sequence = True
                sequence_dict = [
                    {
                        "entity1": action.entity1,
                        "action": action.action,
                        "entity2": action.entity2,
                        "channel": action.channel
                    }
                    for action in sequence
                ]
            else:
                print(f"\nInvalid sequence")
        return sequence_dict

    def generate_multiple_sequences(self, count: int, max_steps: int = 10) -> List[Dict]:
        """
        Generate multiple sequences and return them as a list of dictionaries
        
        Args:
            count: Number of sequences to generate
            max_steps: Maximum steps per sequence
        
        Returns:
            List of sequence dictionaries with metadata
        """
        sequences = []

        successful_count = 0

        while successful_count < count:
            print(f"\n{'='*60}")
            print(f"Generating sequence {successful_count+1}/{count}")
            print(f"{'='*60}")
            
            try:
                sequence = self.generate_sequence(max_steps=max_steps)

                sequence_dict = [
                    {
                        "entity1": action.entity1,
                        "action": action.action,
                        "entity2": action.entity2,
                        "channel": action.channel
                    }
                    for action in sequence
                ]
                
                # Validate sequence
                label, reason = validate_one_sequence(sequence_dict)
                if label == "valid":
                    print("\nValid sequence: True")
                    successful_count += 1
                    sequences.append(
                        {
                            "sequence_id": successful_count,
                            "label": "fraud",
                            "timestamp": datetime.now().isoformat(),
                            "actions": sequence_dict,
                            "action_count": len(sequence),
                        }
                    )
                    print(f"✓ Sequence {successful_count} is valid")
                else:
                    print("\nValid sequence: False")
                
            except Exception as e:
                print(f"✗ Error generating sequence {successful_count+1}: {e}")
                continue
        
        return sequences

    def save_sequences_to_file(self, sequences: List[Dict], filename: str = "sequences.json") -> None:
        """
        Save sequences to a JSON file with metadata
        
        Args:
            sequences: List of sequence dictionaries
            filename: Output filename
        """
        output = {
            "metadata": {
                "total_sequences": len(sequences),
                "generated_at": datetime.now().isoformat(),
                "successful_sequences": sum(1 for s in sequences if s.get("successful", False)),
                "total_actions": sum(s.get("action_count", 0) for s in sequences),
                "average_actions_per_sequence": sum(s.get("action_count", 0) for s in sequences) / len(sequences) if sequences else 0
            },
            "sequences": sequences
        }
        
        with open(filename, "w") as f:
            json.dump(output, f, indent=2)
        
        print(f"\n✓ Saved {len(sequences)} sequences to {filename}")
        print(f"  - Successful: {output['metadata']['successful_sequences']}")
        print(f"  - Total actions: {output['metadata']['total_actions']}")
        print(f"  - Average actions per sequence: {output['metadata']['average_actions_per_sequence']:.2f}")

    

def main():
    parser = argparse.ArgumentParser(description="Generate fraud sequences")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(os.getenv("FRAUD_SEQUENCES_OUT", "fraud_sequences.json")),
        help="Output JSON path (default: FRAUD_SEQUENCES_OUT or fraud_sequences.json)",
    )
    args = parser.parse_args()

    SequenceBuilder = Neo4jSequenceBuilder()
    
    try:
        # sequences = SequenceBuilder.generate_multiple_sequences(count=1, max_steps=10)
        # args.output.parent.mkdir(parents=True, exist_ok=True)
        # SequenceBuilder.save_sequences_to_file(sequences, filename=str(args.output))

        sequence = SequenceBuilder.generate_one_sequence(max_steps=10)
        print(sequence)
        
    finally:
        SequenceBuilder.close()

if __name__ == "__main__":
    main()