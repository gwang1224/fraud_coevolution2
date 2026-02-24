"""
Environment Builder for Fraudulent Transaction Sequences

This module generates deterministic fraud scenarios using LLM (Ollama) and stores
them in Neo4j graph database with proper nodes and relationships.
"""

import json
import random
from typing import Dict, List, Optional
from dataclasses import dataclass
from neo4j import GraphDatabase
import requests


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

SYS_PROMPT = """
You are designing a structured graph database for a FAST-payment fraud simulation.
You must return valid JSON only.
Do not include explanations or commentary.
"""


class OllamaClient:
    """Client for interacting with Ollama LLM"""
    
    def __init__(self, base_url: str = "http://localhost:11434", model: str = "llama3.2"):
        self.base_url = base_url
        self.model = model
    
    def generate_victims(self) -> Dict:
        """
        Use LLM to generate victims and accounts
        Returns a dictionary with 'individuals' and 'accounts' lists
        """

        prompt = """
        Create a structured nested JSON dictionary for a FAST-payment fraud simulation.

        You must generate:

        - 10 individuals (victims)
        - 1 account per individual

        Requirements:

        1. Individuals:
           - Each must have:
             - "name" (lowercase, snake_case, no spaces)

        2. Accounts:
           - Each must have:
             - "name" (format: acc_<individual_name>)
             - "owner" (must match individual name exactly)
             - "bank" (realistic bank name in lowercase snake_case)
             - "balance" (must be close to avg_balance value)

        Constraints:
        - Names must be lowercase snake_case.
        - All accounts must reference valid individuals.
        - Do NOT include markdown.
        - Do NOT include explanations.
        - Output valid JSON only.

        Output format:

        {
          "victims": [
            {"name": "...", 
                "account": {
                    "acc_name" : "acc_...",
                    "owner": "...",
                    "bank": "...",
                    "balance": 1000
                }
            }
        }
        """
        
        try:
            # Combine system prompt and user prompt
            full_prompt = f"{SYS_PROMPT}\n\n{prompt}"
            
            response = requests.post(
                f"{self.base_url}/api/generate",
                json={
                    "model": self.model,
                    "prompt": full_prompt,
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
                # Remove markdown code blocks if present
                if content.startswith("```"):
                    # Find the JSON part
                    parts = content.split("```")
                    for part in parts:
                        part = part.strip()
                        if part.startswith("json"):
                            part = part[4:].strip()
                        if part.startswith("{") and part.endswith("}"):
                            content = part
                            break
                
                parsed = json.loads(content)
                # Validate structure
                if "victims" in parsed:
                    with open("victims.json", "w") as f:
                        json.dump(parsed, f, indent=4)
                    print(f"Victims data saved to victims.json") 
                    return parsed
                else:
                    raise ValueError("Response missing 'individuals' or 'accounts' keys")
            else:
                raise ValueError("Unexpected response format from Ollama")
                
        except Exception as e:
            print(f"Error generating victims with Ollama: {e}")
        
    def generate_fraudsters(self) -> Dict:
        """
        Use LLM to generate fraudsters and accounts
        Returns a dictionary with 'individuals' and 'accounts' lists
        """

        prompt = """
        Create a structured nested JSON dictionary for a FAST-payment fraud simulation.

        You must generate:

        - 10 individuals (fraudsters)
        - 1 account per fraudster

        Requirements:

        1. Fraudster:
           - Each must have:
             - "name" (lowercase, snake_case, no spaces)

        2. Accounts:
           - Each must have:
             - "name" (format: acc_<individual_name>)
             - "owner" (must match individual name exactly)
             - "bank" (realistic bank name in lowercase snake_case)
             - "balance" (must be close to avg_balance value)

        Constraints:
        - Names must be lowercase snake_case.
        - All accounts must reference valid individuals.
        - Do NOT include markdown.
        - Do NOT include explanations.
        - Output valid JSON only.

        Output format:

        {
          "fraudsters": [
            {"name": "...", 
                "account": {
                    "acc_name" : "acc_...",
                    "owner": "...",
                    "bank": "...",
                    "balance": 1000
                }
            }
        }
        """
        
        try:
            # Combine system prompt and user prompt
            full_prompt = f"{SYS_PROMPT}\n\n{prompt}"
            
            response = requests.post(
                f"{self.base_url}/api/generate",
                json={
                    "model": self.model,
                    "prompt": full_prompt,
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
                # Remove markdown code blocks if present
                if content.startswith("```"):
                    # Find the JSON part
                    parts = content.split("```")
                    for part in parts:
                        part = part.strip()
                        if part.startswith("json"):
                            part = part[4:].strip()
                        if part.startswith("{") and part.endswith("}"):
                            content = part
                            break
                
                parsed = json.loads(content)
                # Validate structure
                if "fraudsters" in parsed:
                    with open("fraudsters.json", "w") as f:
                        json.dump(parsed, f, indent=4)
                    print(f"Fraudsters data saved to fraudsters.json") 
                    return parsed
                else:
                    raise ValueError("Response missing 'individuals' or 'accounts' keys")
            else:
                raise ValueError("Unexpected response format from Ollama")
                
        except Exception as e:
            print(f"Error generating victims with Ollama: {e}")


    def generate_actions(self) -> Dict:
        """
        Use LLM to generate fraudster and victim actions
        Returns a dictionary with 'fraudster_actions' and 'victim_actions' lists
        """
        prompt = """
        Create a structured nested JSON dictionary for a FAST-payment fraud simulation.

        You must generate:

        - 10 fraudster actions (e.g., phishing, impersonation, sim_swap, credential_theft, etc.)
        - 10 victim actions (e.g., submit_sensitive_information, transfer_money, provide_credentials, authorize_transaction, etc.)

        Requirements:

        1. Fraudster Actions:
           - Each must have:
             - "name" (lowercase, snake_case, no spaces, descriptive action name)
             - "common_channels" (array of strings: email, sms, call, app, website, in_person, etc.)
             - "description" (string: describe the action and what it does)
             - "stage" (string: e.g. "RECON", "TRUST_BUILDING", "CREDENTIAL_THEFT", "TRANSFER", etc.)
             - "initiator" (string: always "fraudster")
             - "target" (string: one of: victim, victim_account, bank, institution, merchant, government, telecom)
             - "compromises_account" (bool: True if this action leaks the account information)
             - "is_terminal" (bool: ONLY TRUE if money is transferred)

        2. Victim Actions:
           - Each must have:
             - "name" (lowercase, snake_case, no spaces, descriptive action name)
             - "common_channels" (array of strings: email, sms, call, app, website, in_person, etc.)
             - "description" (string: describe the action and what it does)
             - "stage" (string: e.g. "RECON", "TRUST_BUILDING", "CREDENTIAL_THEFT", "TRANSFER", etc.)
             - "initiator" (string: always "victim")
             - "target" (string: one of: fraudster, fraudster_account, bank, institution, merchant, government, telecom, victim_account)
             - "compromises_account" (bool: True if this action leaks the account information)
             - "is_terminal" (bool: ONLY TRUE if money is transferred)

        Constraints:
        - Names must be lowercase snake_case.
        - Common_channels must be an array of at least one channel.
        - Targets must be valid entity types.
        - Do NOT include markdown.
        - Do NOT include explanations.
        - Output valid JSON only.

        Output format:

        {
          "fraudster_actions": [
            {
              "name": "phishing",
              "common_channels": ["email", "sms", "website"],
              "initiator": "fraudster",
              "target": "victim"
            }
          ],
          "victim_actions": [
            {
              "name": "submit_sensitive_information",
              "common_channels": ["email", "website"],
              "initiator": "victim",
              "target": "fraudster"
            }
          ]
        }
        """
        
        try:
            print("Start generating actions...")
            # Combine system prompt and user prompt
            full_prompt = f"{SYS_PROMPT}\n\n{prompt}"
            
            response = requests.post(
                f"{self.base_url}/api/generate",
                json={
                    "model": self.model,
                    "prompt": full_prompt,
                    "stream": False,
                    "format": "json"
                },
            )
            response.raise_for_status()
            result = response.json()
            
            # Extract JSON from response
            if "response" in result:
                content = result["response"].strip()
                # Remove markdown code blocks if present
                if content.startswith("```"):
                    # Find the JSON part
                    parts = content.split("```")
                    for part in parts:
                        part = part.strip()
                        if part.startswith("json"):
                            part = part[4:].strip()
                        if part.startswith("{") and part.endswith("}"):
                            content = part
                            break
                
                parsed = json.loads(content)
                # Validate structure
                if "fraudster_actions" in parsed and "victim_actions" in parsed:
                    with open("actions.json", "w") as f:
                        json.dump(parsed, f, indent=4)
                    print(f"Actions data saved to actions.json")
                    print(f"  - Generated {len(parsed.get('fraudster_actions', []))} fraudster actions")
                    print(f"  - Generated {len(parsed.get('victim_actions', []))} victim actions")
                    return parsed
                else:
                    raise ValueError("Response missing 'fraudster_actions' or 'victim_actions' keys")
            else:
                raise ValueError("Unexpected response format from Ollama")
                
        except Exception as e:
            print(f"Error generating actions with Ollama: {e}")
            return None

    
    

def main():
    """Main function to run the environment builder"""
    ollama_client = OllamaClient()
    victims_data = ollama_client.generate_actions()
    # print(victims_data)

if __name__ == "__main__":
    main()