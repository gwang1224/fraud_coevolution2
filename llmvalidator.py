import json
from build_sequence import Neo4jSequenceBuilder
import os
from dotenv import load_dotenv
import requests

load_dotenv()

# NEO4J_URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")
# NEO4J_USERNAME = os.getenv("NEO4J_USERNAME", "neo4j")
# NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "password")
# Neo4jSequenceBuilder = Neo4jSequenceBuilder(NEO4J_URI, NEO4J_USERNAME, NEO4J_PASSWORD)

prompt = """
You are a fraud analyst validating whether a sequence of actions represents a realistic FAST-payment fraud scenario.

Fraud sequences typically follow a logical progression of stages:

RECON → TRUST_BUILDING → CREDENTIAL_THEFT → ACCOUNT_TAKEOVER → PAYMENT_MANIPULATION → TRANSFER

Rules for a valid fraud sequence:
1. The sequence should follow a logical progression of stages (not necessarily all stages, but it must make sense).
2. A TRANSFER stage (money movement) should only occur after the fraudster gains sufficient access or trust.
3. If money is transferred, the sequence must include actions that compromise credentials, accounts, or victim trust beforehand.
4. Actions should logically connect (e.g., phishing → victim provides credentials → fraudster accesses account → transfer).
5. If the sequence contains impossible or illogical transitions, it is invalid.

Sequence to evaluate:
{input_sequence}

Your task:
Determine whether this is a plausible fraud sequence.
Return only the word 'valid' if the sequence is valid, otherwise return 'invalid'.
Dont explain why the sequence is valid or invalid.
"""

with open('sequences_100.json', 'r') as file:
    sequences = json.load(file)['sequences']

valid_sequences = 0

for sequence in sequences:
    input_sequence = ""
    actions = sequence['actions']
    for index, action in enumerate(actions):
        input_sequence += f"{index + 1}. {action['entity1']} - {action['action']} -> {action['entity2']} via {action['channel']}\n"

    prompt = prompt.format(input_sequence=input_sequence)
    try:
        response = requests.post(
            "http://localhost:11434/api/generate",
            json={
                "model": "llama3.2",
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": 0
                },
            },
        )
        response.raise_for_status()
        result = response.json()['response']
        print(result)
        if result.lower() == "valid":
            valid_sequences += 1
    except Exception as e:
        print(f"Error validating sequence {sequence['sequence_id']}: {e}")
        continue

print(f"Valid sequences: {valid_sequences}")
print(f"Invalid sequences: {100 - valid_sequences}")