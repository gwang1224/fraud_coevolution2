"""
LLM-driven **data generation** for fraud-simulation JSON assets (Ollama).

``OllamaClient`` uses prompts under the repository ``llm_config/`` directory and optional
``data/llm/guidelines.json`` to
author new victims, fraudsters, fraudster actions, victim (fraud) actions, and **legit**
actions. Outputs are written under ``data/graph/`` (e.g. ``victims.json``, ``fraudsters.json``,
``fraudster_actions.json``, ``victim_actions.json``). Legit rows are merged into
``victim_actions.json`` under the ``legit_actions`` key for ``build_legit_sequence``.

**Loading into Neo4j** is a separate step: use ``src/build_graph.py`` after these files exist.

Requires: ``requests``, ``python-dotenv``, Ollama.
"""

import json
import random
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import requests
from pathlib import Path


# Categories for LLM-generated legit actions (must match build_normal_sequence / victim_actions.json).
LEGIT_ACTION_CATEGORIES: Tuple[str, ...] = (
    "check_balance",
    "pay_friend",
    "pay_rent",
    "small_bill_payment",
    "recurring_transfer",
    "new_benign_recipient",
)


@dataclass
class Victim:
    """Represents a victim entity"""
    name: str
    risk_profile: str
    avg_balance: float


@dataclass
class Account:
    """Represents an account entity"""
    name: str
    owner: str
    bank: str
    balance: float

class OllamaClient:
    """Client for interacting with Ollama LLM"""
    
    def __init__(self, base_url: str, model: str, prompt_dir: str, guidelines_path: str = "data/llm/guidelines.json"):
        self.base_url = base_url
        self.model = model
        self.prompt_dir = Path(prompt_dir)
        self.guidelines_path = guidelines_path
        self.guidelines = self._load_guidelines()
        self.prompts = {
            "victim": self._load("victim.txt"),
            "fraudster": self._load("fraudster.txt"),
            "fraudster_actions": self._load("fraudster_actions.txt"),
            "victim_actions": self._load("victim_actions.txt"),
            "legit_actions": self._load("legit_actions.txt"),
        }
    
    def _load(self, filename):
        with open(self.prompt_dir / filename, "r") as f:
            return f.read()
    
    def _load_guidelines(self) -> Dict:
        """Load guidelines from JSON file"""
        try:
            with open(self.guidelines_path, "r") as f:
                return json.load(f)
        except FileNotFoundError:
            print(f"Warning: Guidelines file not found at {self.guidelines_path}")
            return {}
        except json.JSONDecodeError:
            print(f"Warning: Could not parse guidelines file {self.guidelines_path}")
            return {}
        except Exception as e:
            print(f"Warning: Error loading guidelines: {e}")
            return {}
    
    def _format_guidelines_for_fraudster_actions(self) -> str:
        """Format guidelines relevant to fraudster actions"""
        if not self.guidelines:
            return ""
        
        sections = []
        
        if "fraud_lifecycle" in self.guidelines:
            sections.append("Fraud Lifecycle:\n" + "\n".join(f"  - {item}" for item in self.guidelines["fraud_lifecycle"]))
        
        if "fraud_techniques" in self.guidelines:
            sections.append("\nFraud Techniques:\n" + "\n".join(f"  - {item}" for item in self.guidelines["fraud_techniques"]))
        
        if "payment_manipulation_patterns" in self.guidelines:
            sections.append("\nPayment Manipulation Patterns:\n" + "\n".join(f"  - {item}" for item in self.guidelines["payment_manipulation_patterns"]))
        
        if "money_movement_patterns" in self.guidelines:
            sections.append("\nMoney Movement Patterns:\n" + "\n".join(f"  - {item}" for item in self.guidelines["money_movement_patterns"]))
        
        if "design_rules_for_fraud_simulation" in self.guidelines:
            sections.append("\nDesign Rules for Fraud Simulation:\n" + "\n".join(f"  - {item}" for item in self.guidelines["design_rules_for_fraud_simulation"]))
        
        if sections:
            return "\n\nBackground Information (Use these guidelines to inform your action generation):\n" + "\n".join(sections) + "\n"
        return ""
    
    def _format_guidelines_for_victim_actions(self) -> str:
        """Format guidelines relevant to victim actions"""
        if not self.guidelines:
            return ""
        
        sections = []
        
        if "fraud_lifecycle" in self.guidelines:
            sections.append("Fraud Lifecycle:\n" + "\n".join(f"  - {item}" for item in self.guidelines["fraud_lifecycle"]))
        
        if "victim_behaviors" in self.guidelines:
            sections.append("\nVictim Behaviors:\n" + "\n".join(f"  - {item}" for item in self.guidelines["victim_behaviors"]))
        
        if "payment_manipulation_patterns" in self.guidelines:
            sections.append("\nPayment Manipulation Patterns:\n" + "\n".join(f"  - {item}" for item in self.guidelines["payment_manipulation_patterns"]))
        
        if "case_study_insights" in self.guidelines:
            sections.append("\nCase Study Insights:\n" + "\n".join(f"  - {item}" for item in self.guidelines["case_study_insights"]))
        
        if "design_rules_for_fraud_simulation" in self.guidelines:
            sections.append("\nDesign Rules for Fraud Simulation:\n" + "\n".join(f"  - {item}" for item in self.guidelines["design_rules_for_fraud_simulation"]))
        
        if sections:
            return "\n\nBackground Information (Use these guidelines to inform your action generation):\n" + "\n".join(sections) + "\n"
        return ""
    
    def load_existing_victims(self, filepath: str = "data/graph/victims.json") -> List[str]:
        """
        Load existing victim names from victims.json to check for uniqueness
        
        Args:
            filepath: Path to victims.json file
        
        Returns:
            List of existing victim names
        """
        existing_names = []
        try:
            with open(filepath, "r") as f:
                data = json.load(f)
                if "victims" in data:
                    existing_names = [victim.get("name", "") for victim in data["victims"] if victim.get("name")]
        except FileNotFoundError:
            # File doesn't exist yet, no existing victims
            pass
        except json.JSONDecodeError:
            print(f"Warning: Could not parse {filepath}, assuming no existing victims")
        except Exception as e:
            print(f"Warning: Error loading existing victims: {e}")
        
        return existing_names

    def load_existing_fraudsters(self, filepath: str = "data/graph/fraudsters.json") -> List[str]:
        """
        Load existing fraudster names from fraudsters.json to check for uniqueness
        
        Args:
            filepath: Path to fraudsters.json file
        
        Returns:
            List of existing fraudster names
        """
        existing_names = []
        try:
            with open(filepath, "r") as f:
                data = json.load(f)
                if "fraudsters" in data:
                    existing_names = [fraudster.get("name", "") for fraudster in data["fraudsters"] if fraudster.get("name")]
        except FileNotFoundError:
            # File doesn't exist yet, no existing fraudsters
            pass
        except json.JSONDecodeError:
            print(f"Warning: Could not parse {filepath}, assuming no existing fraudsters")
        except Exception as e:
            print(f"Warning: Error loading existing fraudsters: {e}")
        
        return existing_names

    def load_existing_fraudster_actions(self, filepath: str = "data/graph/fraudster_actions.json") -> List[str]:
        """
        Load existing fraudster action names from fraudster_actions.json to check for uniqueness
        
        Args:
            filepath: Path to fraudster_actions.json file
        
        Returns:
            List of existing fraudster action names
        """
        existing_names = []
        try:
            with open(filepath, "r") as f:
                data = json.load(f)
                if "fraudster_actions" in data:
                    existing_names = [action.get("name", "") for action in data["fraudster_actions"] if action.get("name")]
        except FileNotFoundError:
            # File doesn't exist yet, no existing actions
            pass
        except json.JSONDecodeError:
            print(f"Warning: Could not parse {filepath}, assuming no existing fraudster actions")
        except Exception as e:
            print(f"Warning: Error loading existing fraudster actions: {e}")
        
        return existing_names

    def load_existing_victim_actions(self, filepath: str = "data/graph/victim_actions.json") -> List[str]:
        """
        Load existing victim action names from victim_actions.json to check for uniqueness
        
        Args:
            filepath: Path to victim_actions.json file
        
        Returns:
            List of existing victim action names
        """
        existing_names = []
        try:
            with open(filepath, "r") as f:
                data = json.load(f)
                if "victim_actions" in data:
                    existing_names = [action.get("name", "") for action in data["victim_actions"] if action.get("name")]
        except FileNotFoundError:
            # File doesn't exist yet, no existing actions
            pass
        except json.JSONDecodeError:
            print(f"Warning: Could not parse {filepath}, assuming no existing victim actions")
        except Exception as e:
            print(f"Warning: Error loading existing victim actions: {e}")
        
        return existing_names

    def load_existing_legit_actions(self, filepath: str = "data/graph/victim_actions.json") -> List[str]:
        """Existing legit action names (legit_actions key) for uniqueness."""
        existing_names: List[str] = []
        try:
            with open(filepath, "r") as f:
                data = json.load(f)
                if "legit_actions" in data:
                    existing_names = [
                        a.get("name", "")
                        for a in data["legit_actions"]
                        if a.get("name")
                    ]
        except FileNotFoundError:
            pass
        except json.JSONDecodeError:
            print(f"Warning: Could not parse {filepath}, assuming no existing legit actions")
        except Exception as e:
            print(f"Warning: Error loading existing legit actions: {e}")
        return existing_names

    def _load_victim_actions_file(self, filepath: str) -> Dict:
        """Full victim_actions.json payload (fraud victim_actions + legit_actions)."""
        try:
            with open(filepath, "r") as f:
                return json.load(f)
        except FileNotFoundError:
            return {}
        except json.JSONDecodeError as e:
            print(f"Warning: Could not parse {filepath}: {e}")
            return {}

    def _validate_legit_action(self, action: Dict) -> bool:
        if not action.get("name") or not action.get("common_channels"):
            return False
        cat = action.get("category", "")
        if cat not in LEGIT_ACTION_CATEGORIES:
            return False
        if "is_payment" not in action:
            return False
        if not str(action.get("target", "")).strip():
            return False
        return True

    def generate_single_victim(self, existing_names: List[str] = None) -> Optional[Dict]:
        """
        Use LLM to generate a single victim and account
        Checks against existing names to ensure uniqueness
        
        Args:
            existing_names: List of existing victim names to avoid duplicates
        
        Returns:
            Dictionary with 'name' and 'account' keys, or None if generation fails
        """
        if existing_names is None:
            existing_names = []
        
        # Build prompt with existing names to avoid duplicates
        base_prompt = self.prompts.get("victim")
        
        if existing_names:
            names_list = ", ".join(existing_names[:50])  # Limit to first 50 to avoid prompt being too long
            uniqueness_constraint = f"\n\nIMPORTANT: Do NOT use any of these existing names: {names_list}\nGenerate a completely unique name that is not in this list."
            prompt = base_prompt + uniqueness_constraint
        else:
            prompt = base_prompt
        
        try:
            response = requests.post(
                f"{self.base_url}/api/generate",
                json={
                    "model": self.model,
                    "prompt": prompt,
                    "stream": False,
                    "format": "json"
                },
                timeout=60
            )
            response.raise_for_status()
            result = response.json()
            
            # Extract JSON from response
            if "response" in result:
                content = result["response"].strip()
                if content.startswith("```"):
                    parts = content.split("```")
                    for part in parts:
                        part = part.strip()
                        if part.startswith("json"):
                            part = part[4:].strip()
                        if part.startswith("{") and part.endswith("}"):
                            content = part
                            break
                
                parsed = json.loads(content)
                if "name" in parsed and "account" in parsed:
                    # Double-check uniqueness
                    victim_name = parsed.get("name", "").lower().strip()
                    if victim_name in [name.lower().strip() for name in existing_names]:
                        print(f"Warning: Generated duplicate name '{victim_name}', will regenerate")
                        return None
                    return parsed
                else:
                    raise ValueError("Response missing 'name' or 'account' keys")
            else:
                raise ValueError("Unexpected response format from Ollama")
                
        except Exception as e:
            print(f"Error generating victim with Ollama: {e}")
            return None

    def generate_victim_database(self, num_victims=25, filepath: str = "data/graph/victims.json"):
        """
        Generate multiple unique victims and save to file
        
        Args:
            num_victims: Number of victims to generate
            filepath: Path to save victims.json file
        """
        # Load existing victims to check for uniqueness
        existing_names = self.load_existing_victims(filepath)
        print(f"Found {len(existing_names)} existing victims")
        
        # Load existing victims data if file exists
        existing_victims = []
        try:
            with open(filepath, "r") as f:
                data = json.load(f)
                if "victims" in data:
                    existing_victims = data["victims"]
        except FileNotFoundError:
            pass
        
        victims = existing_victims.copy()
        max_attempts = 10 
        
        for i in range(num_victims):
            print(f"Generating victim {i+1}/{num_victims}...")
            attempts = 0
            new_victim = None
            
            while attempts < max_attempts:
                new_victim = self.generate_single_victim(existing_names)
                if new_victim:
                    victim_name = new_victim.get("name", "").lower().strip()
                    new_victim["id"] = i
                    if victim_name not in [v.get("name", "").lower().strip() for v in victims]:
                        victims.append(new_victim)
                        existing_names.append(victim_name)
                        print(f"  ✓ Generated unique victim: {victim_name}")
                        break
                    else:
                        print(f"  ✗ Duplicate detected: {victim_name}, retrying...")
                        attempts += 1
                else:
                    attempts += 1
            
            if not new_victim or attempts >= max_attempts:
                print(f"  ✗ Failed to generate unique victim after {max_attempts} attempts")
        
        # Save all victims to file
        output = {"victims": victims}
        with open(filepath, "w") as f:
            json.dump(output, f, indent=4)
        
        print(f"\n✓ Saved {len(victims)} total victims to {filepath}")
        print(f"  - New victims generated: {num_victims}")
        print(f"  - Total unique victims: {len(victims)}")
            
    def generate_single_fraudster(self, existing_names: List[str] = None) -> Optional[Dict]:
        """
        Use LLM to generate a single fraudster and account
        Checks against existing names to ensure uniqueness
        
        Args:
            existing_names: List of existing fraudster names to avoid duplicates
        
        Returns:
            Dictionary with 'name' and 'account' keys, or None if generation fails
        """
        if existing_names is None:
            existing_names = []
        
        # Build prompt with existing names to avoid duplicates
        base_prompt = self.prompts.get("fraudster")
        
        # Add instruction to vary entity types and generate appropriate names
        diversity_instruction = "\n\nCRITICAL INSTRUCTION: You MUST generate diverse entity types. Randomly choose from: individual, business, merchant, organization, criminal_network, shell_company, fraud_ring, money_laundering_operation. Do NOT always generate individuals. Generate business names (like 'global_trading_llc', 'premium_services_inc') for businesses, organization names for organizations, person names for individuals, etc. The name format MUST match the entity_type."
        
        if existing_names:
            names_list = ", ".join(existing_names[:50])  # Limit to first 50 to avoid prompt being too long
            uniqueness_constraint = f"\n\nIMPORTANT: Do NOT use any of these existing names: {names_list}\nGenerate a completely unique name that is not in this list."
            prompt = base_prompt + diversity_instruction + uniqueness_constraint
        else:
            prompt = base_prompt + diversity_instruction
        
        try:
            response = requests.post(
                f"{self.base_url}/api/generate",
                json={
                    "model": self.model,
                    "prompt": prompt,
                    "stream": False,
                    "format": "json"
                },
                timeout=60
            )
            response.raise_for_status()
            result = response.json()
            
            # Extract JSON from response
            if "response" in result:
                content = result["response"].strip()
                if content.startswith("```"):
                    parts = content.split("```")
                    for part in parts:
                        part = part.strip()
                        if part.startswith("json"):
                            part = part[4:].strip()
                        if part.startswith("{") and part.endswith("}"):
                            content = part
                            break
                
                parsed = json.loads(content)
                if "name" in parsed and "account" in parsed:
                    # Double-check uniqueness
                    fraudster_name = parsed.get("name", "").lower().strip()
                    if fraudster_name in [name.lower().strip() for name in existing_names]:
                        print(f"Warning: Generated duplicate name '{fraudster_name}', will regenerate")
                        return None
                    # Ensure entity_type is present, default to "individual" if missing
                    if "entity_type" not in parsed:
                        parsed["entity_type"] = "individual"
                    return parsed
                else:
                    raise ValueError("Response missing 'name' or 'account' keys")
            else:
                raise ValueError("Unexpected response format from Ollama")
                
        except Exception as e:
            print(f"Error generating fraudster with Ollama: {e}")
            return None

    def generate_fraudster_database(self, num_fraudsters=25, filepath: str = "data/graph/fraudsters.json"):
        """
        Generate multiple unique fraudsters and save to file
        
        Args:
            num_fraudsters: Number of fraudsters to generate
            filepath: Path to save fraudsters.json file
        """
        # Load existing fraudsters to check for uniqueness
        existing_names = self.load_existing_fraudsters(filepath)
        print(f"Found {len(existing_names)} existing fraudsters")
        
        # Load existing fraudsters data if file exists
        existing_fraudsters = []
        try:
            with open(filepath, "r") as f:
                data = json.load(f)
                if "fraudsters" in data:
                    existing_fraudsters = data["fraudsters"]
        except FileNotFoundError:
            pass
        
        fraudsters = existing_fraudsters.copy()  # Start with existing fraudsters
        max_attempts = 10  # Maximum attempts to generate a unique fraudster
        
        for i in range(num_fraudsters):
            print(f"Generating fraudster {i+1}/{num_fraudsters}...")
            attempts = 0
            new_fraudster = None
            
            while attempts < max_attempts:
                new_fraudster = self.generate_single_fraudster(existing_names)
                if new_fraudster:
                    fraudster_name = new_fraudster.get("name", "").lower().strip()
                    entity_type = new_fraudster.get("entity_type", "individual")
                    new_fraudster["id"] = i
                    if fraudster_name not in [f.get("name", "").lower().strip() for f in fraudsters]:
                        fraudsters.append(new_fraudster)
                        existing_names.append(fraudster_name)
                        print(f"  ✓ Generated unique fraudster: {fraudster_name} (type: {entity_type})")
                        break
                    else:
                        print(f"  ✗ Duplicate detected: {fraudster_name}, retrying...")
                        attempts += 1
                else:
                    attempts += 1
            
            if not new_fraudster or attempts >= max_attempts:
                print(f"  ✗ Failed to generate unique fraudster after {max_attempts} attempts")
        
        # Save all fraudsters to file
        output = {"fraudsters": fraudsters}
        with open(filepath, "w") as f:
            json.dump(output, f, indent=4)
        
        # Count entity types for summary
        entity_type_counts = {}
        for f in fraudsters:
            entity_type = f.get("entity_type", "individual")
            entity_type_counts[entity_type] = entity_type_counts.get(entity_type, 0) + 1
        
        print(f"\n✓ Saved {len(fraudsters)} total fraudsters to {filepath}")
        print(f"  - New fraudsters generated: {num_fraudsters}")
        print(f"  - Total unique fraudsters: {len(fraudsters)}")
        print(f"  - Entity type distribution: {entity_type_counts}")

    def generate_fraudster_actions_database(self, num_actions=25, filepath: str = "data/graph/fraudster_actions.json") -> List[Dict]:
        """
        Use LLM to generate fraudster actions with uniqueness checking
        Returns a list of fraudster actions
        
        Args:
            num_actions: Number of fraudster actions to generate
            filepath: Path to save fraudster_actions.json file
        
        Returns:
            List of fraudster action dictionaries
        """
        # Load existing actions to check for uniqueness
        existing_names = self.load_existing_fraudster_actions(filepath)
        print(f"Found {len(existing_names)} existing fraudster actions")
        
        # Load existing actions data if file exists
        existing_actions = []
        try:
            with open(filepath, "r") as f:
                data = json.load(f)
                if "fraudster_actions" in data:
                    existing_actions = data["fraudster_actions"]
        except FileNotFoundError:
            pass
        
        actions = existing_actions.copy()
        base_prompt = self.prompts.get("fraudster_actions")
        guidelines_context = self._format_guidelines_for_fraudster_actions()
        max_attempts = 10
        
        print(f"\nGenerating {num_actions} unique fraudster actions...")
        
        for i in range(num_actions):
            print(f"  Generating fraudster action {i+1}/{num_actions}...")
            attempts = 0
            
            while attempts < max_attempts:
                prompt_parts = [base_prompt]
                
                if guidelines_context:
                    prompt_parts.append(guidelines_context)
                
                if existing_names:
                    names_list = ", ".join(existing_names[:30])
                    uniqueness_constraint = f"\n\nIMPORTANT: Do NOT use any of these existing fraudster action names: {names_list}\nGenerate a completely unique action name that is not in this list."
                    prompt_parts.append(uniqueness_constraint)
                
                prompt = "\n".join(prompt_parts)
                
                try:
                    response = requests.post(
                        f"{self.base_url}/api/generate",
                        json={
                            "model": self.model,
                            "prompt": prompt,
                            "stream": False,
                            "format": "json"
                        },
                        timeout=60
                    )
                    response.raise_for_status()
                    result = response.json()
                    
                    if "response" in result:
                        content = result["response"].strip()
                        if content.startswith("```"):
                            parts = content.split("```")
                            for part in parts:
                                part = part.strip()
                                if part.startswith("json"):
                                    part = part[4:].strip()
                                if part.startswith("{") and part.endswith("}"):
                                    content = part
                                    break
                        
                        parsed = json.loads(content)
                        # Handle different response formats
                        new_action = None
                        if "fraudster_actions" in parsed and len(parsed["fraudster_actions"]) > 0:
                            new_action = parsed["fraudster_actions"][0]
                        elif "name" in parsed:  # Single action format
                            new_action = parsed
                        
                        if new_action:
                            action_name = new_action.get("name", "").lower().strip()
                            
                            if action_name not in [a.get("name", "").lower().strip() for a in actions]:
                                actions.append(new_action)
                                existing_names.append(action_name)
                                print(f"    ✓ Generated unique fraudster action: {action_name}")
                                break
                            else:
                                print(f"    ✗ Duplicate detected: {action_name}, retrying...")
                                attempts += 1
                        else:
                            attempts += 1
                    else:
                        attempts += 1
                except Exception as e:
                    print(f"    ✗ Error: {e}")
                    attempts += 1
            
            if attempts >= max_attempts:
                print(f"    ✗ Failed to generate unique fraudster action after {max_attempts} attempts")
        
        # Save all actions to file
        output = {"fraudster_actions": actions}
        with open(filepath, "w") as f:
            json.dump(output, f, indent=4)
        
        print(f"\n Saved {len(actions)} total fraudster actions to {filepath}")
        print(f"  - New actions generated: {num_actions}")
        print(f"  - Total unique fraudster actions: {len(actions)}")
        
        return actions

    def generate_victim_actions_database(self, num_actions=25, filepath: str = "data/graph/victim_actions.json") -> List[Dict]:
        """
        Use LLM to generate victim actions with uniqueness checking
        Returns a list of victim actions
        
        Args:
            num_actions: Number of victim actions to generate
            filepath: Path to save victim_actions.json file
        
        Returns:
            List of victim action dictionaries
        """
        # Load existing actions to check for uniqueness
        existing_names = self.load_existing_victim_actions(filepath)
        print(f"Found {len(existing_names)} existing victim actions")
        
        # Load existing actions data if file exists
        existing_actions = []
        try:
            with open(filepath, "r") as f:
                data = json.load(f)
                if "victim_actions" in data:
                    existing_actions = data["victim_actions"]
        except FileNotFoundError:
            pass
        
        actions = existing_actions.copy()
        base_prompt = self.prompts.get("victim_actions")
        guidelines_context = self._format_guidelines_for_victim_actions()
        max_attempts = 10
        
        print(f"\nGenerating {num_actions} unique victim actions...")
        
        for i in range(num_actions):
            print(f"  Generating victim action {i+1}/{num_actions}...")
            attempts = 0
            
            while attempts < max_attempts:
                # Build prompt with guidelines and existing names
                prompt_parts = [base_prompt]
                
                if guidelines_context:
                    prompt_parts.append(guidelines_context)
                
                if existing_names:
                    names_list = ", ".join(existing_names[:30])
                    uniqueness_constraint = f"\n\nIMPORTANT: Do NOT use any of these existing victim action names: {names_list}\nGenerate a completely unique action name that is not in this list."
                    prompt_parts.append(uniqueness_constraint)
                
                prompt = "\n".join(prompt_parts)
                
                try:
                    response = requests.post(
                        f"{self.base_url}/api/generate",
                        json={
                            "model": self.model,
                            "prompt": prompt,
                            "stream": False,
                            "format": "json"
                        },
                        timeout=60
                    )
                    response.raise_for_status()
                    result = response.json()
                    
                    if "response" in result:
                        content = result["response"].strip()
                        if content.startswith("```"):
                            parts = content.split("```")
                            for part in parts:
                                part = part.strip()
                                if part.startswith("json"):
                                    part = part[4:].strip()
                                if part.startswith("{") and part.endswith("}"):
                                    content = part
                                    break
                        
                        parsed = json.loads(content)
                        # Handle different response formats
                        new_action = None
                        if "victim_actions" in parsed and len(parsed["victim_actions"]) > 0:
                            new_action = parsed["victim_actions"][0]
                        elif "name" in parsed:  # Single action format
                            new_action = parsed
                        
                        if new_action:
                            action_name = new_action.get("name", "").lower().strip()
                            
                            # Check uniqueness
                            if action_name not in [a.get("name", "").lower().strip() for a in actions]:
                                actions.append(new_action)
                                existing_names.append(action_name)
                                print(f"    ✓ Generated unique victim action: {action_name}")
                                break
                            else:
                                print(f"    ✗ Duplicate detected: {action_name}, retrying...")
                                attempts += 1
                        else:
                            attempts += 1
                    else:
                        attempts += 1
                except Exception as e:
                    print(f"    ✗ Error: {e}")
                    attempts += 1
            
            if attempts >= max_attempts:
                print(f"    ✗ Failed to generate unique victim action after {max_attempts} attempts")
        
        # Save all actions to file; preserve legit_actions if present
        prior = self._load_victim_actions_file(filepath)
        output = {
            "victim_actions": actions,
            "legit_actions": prior.get("legit_actions", []),
        }
        with open(filepath, "w") as f:
            json.dump(output, f, indent=4)
        
        print(f"\n Saved {len(actions)} total victim actions to {filepath}")
        print(f"  - New actions generated: {num_actions}")
        print(f"  - Total unique victim actions: {len(actions)}")
        
        return actions

    def generate_legit_actions_database(
        self, num_actions: int = 25, filepath: str = "data/graph/victim_actions.json"
    ) -> List[Dict]:
        """
        Use LLM to invent new legitimate banking actions; append to legit_actions in victim_actions.json.
        Preserves existing victim_actions (fraud) and any existing legit_actions.
        """
        existing_names = self.load_existing_legit_actions(filepath)
        print(f"Found {len(existing_names)} existing legit action names")

        data = self._load_victim_actions_file(filepath)
        existing_legit = list(data.get("legit_actions", []))
        actions = existing_legit.copy()
        base_prompt = self.prompts.get("legit_actions")
        max_attempts = 10

        print(f"\nGenerating {num_actions} unique legit (benign) actions...")

        for i in range(num_actions):
            category_hint = LEGIT_ACTION_CATEGORIES[i % len(LEGIT_ACTION_CATEGORIES)]
            print(f"  Generating legit action {i + 1}/{num_actions} (category hint: {category_hint})...")
            attempts = 0

            while attempts < max_attempts:
                prompt_parts = [
                    base_prompt,
                    f'\n\nFor this generation, set "category" to exactly "{category_hint}" (required). '
                    f'Invent a novel "name" (snake_case) that fits this category.',
                ]
                if existing_names:
                    names_list = ", ".join(existing_names[:40])
                    prompt_parts.append(
                        f"\n\nIMPORTANT: Do NOT use any of these existing legit action names: {names_list}\n"
                        "Generate a completely unique action name."
                    )
                prompt = "\n".join(prompt_parts)

                try:
                    response = requests.post(
                        f"{self.base_url}/api/generate",
                        json={
                            "model": self.model,
                            "prompt": prompt,
                            "stream": False,
                            "format": "json",
                        },
                        timeout=60,
                    )
                    response.raise_for_status()
                    result = response.json()

                    if "response" not in result:
                        attempts += 1
                        continue

                    content = result["response"].strip()
                    if content.startswith("```"):
                        parts = content.split("```")
                        for part in parts:
                            part = part.strip()
                            if part.startswith("json"):
                                part = part[4:].strip()
                            if part.startswith("{") and part.endswith("}"):
                                content = part
                                break

                    parsed = json.loads(content)
                    new_action = None
                    if "legit_actions" in parsed and len(parsed["legit_actions"]) > 0:
                        new_action = parsed["legit_actions"][0]
                    elif "name" in parsed:
                        new_action = parsed

                    if new_action and self._validate_legit_action(new_action):
                        action_name = new_action.get("name", "").lower().strip()
                        if action_name not in [
                            a.get("name", "").lower().strip() for a in actions
                        ]:
                            actions.append(new_action)
                            existing_names.append(action_name)
                            print(f"    ✓ Generated unique legit action: {action_name}")
                            break
                        print(f"    ✗ Duplicate detected: {action_name}, retrying...")
                        attempts += 1
                    else:
                        attempts += 1
                except Exception as e:
                    print(f"    ✗ Error: {e}")
                    attempts += 1

            if attempts >= max_attempts:
                print(
                    f"    ✗ Failed to generate unique legit action after {max_attempts} attempts"
                )

        victim_block = data.get("victim_actions", [])
        output = {"victim_actions": victim_block, "legit_actions": actions}
        with open(filepath, "w") as f:
            json.dump(output, f, indent=4)

        print(f"\n✓ Saved {len(actions)} total legit_actions entries to {filepath}")
        print(f"  - New actions requested this run: {num_actions}")
        print(f"  - Total legit actions in file: {len(actions)}")

        return actions

    
    

def main():
    """Main function to run the environment builder"""
    _repo_root = Path(__file__).resolve().parents[1]
    client = OllamaClient(
        "http://localhost:11434",
        "llama3.2",
        str(_repo_root / "llm_config"),
        guidelines_path=str(_repo_root / "data" / "llm" / "guidelines.json"),
    )

    # client.generate_victim_database()
    # client.generate_fraudster_database()
    # client.generate_fraudster_actions_database()
    # client.generate_victim_actions_database()
    client.generate_legit_actions_database(num_actions=25)

if __name__ == "__main__":
    main()