"""
Build a **mixed-label** dataset of fraud and legit sequences for coevolution experiments.

Alternates calls into ``Neo4jSequenceBuilder`` (fraud) and ``Neo4jLegitSequenceBuilder``
(legit) according to a random index split, then writes accumulated records to disk.
Intended to be run from the **repository root** so ``src.*`` imports resolve
(``python -m src.generate_dataset``).

Note: The default output filename uses a ``.csv`` extension but the writer uses
``json.dump``; treat the file as JSON or change the extension in code if you need CSV.

Requires: Neo4j, Ollama, and the same dependencies as the sequence builders.
"""

from src.generate_sequences.build_fraud_sequence import Neo4jSequenceBuilder
from src.generate_sequences.build_legit_sequence import Neo4jLegitSequenceBuilder
import random
import json
import time

fraud_builder = Neo4jSequenceBuilder()
legit_builder = Neo4jLegitSequenceBuilder()

def generate_sequences(data_len=50, num_fraud_seq=25):
    data = {}
    fraud_index = random.sample(range(0, data_len), num_fraud_seq)

    for i in range(data_len):
        time.sleep(30)
        print(f"\nSequence {i} ---------------------------------------------")
        label = "fraud" if i in fraud_index else "legit"

        seq = None

        if label == "fraud": 
            seq = fraud_builder.generate_one_sequence()
        else: 
            seq = legit_builder.generate_one_sequence()

        data[i] = {
            "label": label,
            "sequence": seq
        }
    
        with open("coev_dataset2.csv", 'w') as json_file:
            json.dump(data, json_file, indent=4)


if __name__ == "__main__":
    generate_sequences()
