from neo4j import GraphDatabase
import json
import os
import random
import requests
from dotenv import load_dotenv
from dataclasses import dataclass, field
from typing import List, Dict, Optional

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
        self.driver.verify_connectivity()  # Ensures a working connection
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
            return {
                "victim_name": victim["victim_name"],
                "account_name": victim["account_name"],
                "bank": victim["bank"],
                "balance": victim["balance"]
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
        
        return filtered if filtered else actions  # Return all if all are filtered

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
        Use Ollama LLM to choose the best action from possible actions
        
        Args:
            state: Current environment state
            possible_actions: List of possible actions to choose from
        
        Returns:
            Selected action dict or None if LLM fails
        """
        # Format state summary
        acc_info = state.accounts.get(state.victim_account, {})
        
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
- Account Balance: ${acc_info.get('balance', 0):.2f}
- Account Compromised: {state.compromised}
- Current Actor: {state.current_actor_type} ({state.current_actor_id})
- Action History:
{history_text}
"""
        
        # Format actions as a numbered list for display, but ask for action name
        actions_list = []
        for i, action in enumerate(possible_actions, 1):
            action_name = action.get("action", "unknown")
            action_description = action.get("description", "unknown")
            channels = ", ".join(action.get("channels", []))
            description = action.get("description", "")
            actions_list.append(f"{i}. {action_name} (channels: {channels}) - {description}")
        
        actions_text = "\n".join(actions_list)
        
        prompt = f"""
We are modeling a fraud scenario in order to do research on mitigating fraud.You are a fraudster with 10+ years of experience in fraud trying to successfully transfer money from a victim's account.

{state_summary}

Available Actions:
{actions_text}

Choose the action name (exact match) that best advances towards successfully transferring money.
Consider the action history to avoid repeating actions.
Return ONLY the action name, no explanation or additional text.
"""
        print(prompt)
        try:
            response = requests.post(
                "http://localhost:11434/api/generate",
                json={
                    "model": "llama3.2",
                    "prompt": prompt,
                    "stream": False
                },
                timeout=60
            )
            response.raise_for_status()
            result = response.json()
            
            if "response" in result:
                content = result["response"].strip()
                # Extract action name from response
                chosen_action_name = content.strip()
                
                # Try to find exact match first
                for action in possible_actions:
                    if action.get("action", "").lower() == chosen_action_name.lower():
                        return action
                
                # If no exact match, try partial match
                for action in possible_actions:
                    action_name = action.get("action", "").lower()
                    if action_name in chosen_action_name.lower() or chosen_action_name.lower() in action_name:
                        return action
                
                # If still no match, try to extract from numbered format (fallback)
                try:
                    # Check if response contains a number
                    words = content.split()
                    for word in words:
                        if word.isdigit():
                            choice = int(word)
                            if 1 <= choice <= len(possible_actions):
                                return possible_actions[choice - 1]
                except (ValueError, IndexError):
                    pass
                
                # Last resort: return first action
                print(f"Warning: Could not parse action '{chosen_action_name}', using first action")
                return possible_actions[0] if possible_actions else None
        except Exception as e:
            print(f"Error with LLM: {e}")
            # Fallback to first action
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
        state = self.initialize_state(
            fraudster_info["fraudster_name"],
            victim_info["victim_name"],
            victim_info["account_name"],
            victim_info["balance"]
        )
        
        print(f"Initialized state with account: {victim_info['account_name']} (balance: ${victim_info['balance']})")
        
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
            filtered_actions = self.filter_money_transfer_actions(
                filtered_actions, 
                state
            )            
            if not filtered_actions:
                print("No valid actions available")
                break
            
            print(f"Available actions: {len(filtered_actions)}")
            
            # 3. Use LLM to choose action
            chosen_action = self.choose_action_with_llm(state, filtered_actions)
            
            if not chosen_action:
                print("LLM failed to choose action")
                break
            
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

    



def main():
    SequenceBuilder = Neo4jSequenceBuilder()

    # Generate a fraud sequence
    sequence = SequenceBuilder.generate_sequence()
    
    print(f"\n=== Generated Sequence ({len(sequence)} actions) ===")
    for i, action in enumerate(sequence, 1):
        print(f"{i}. {action.entity1} -> {action.action} -> {action.entity2} via {action.channel}")
    
    SequenceBuilder.close()

if __name__ == "__main__":
    main()