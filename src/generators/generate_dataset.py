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

from src.generators.build_fraud_sequence import Neo4jSequenceBuilder
from src.generators.build_legit_sequence import Neo4jLegitSequenceBuilder
import random
import argparse
import json
import time
from pathlib import Path

fraud_builder = Neo4jSequenceBuilder()
legit_builder = Neo4jLegitSequenceBuilder()

def generate_sequences(output_path, data_len=4, num_fraud_seq=2):
    data = {}
    fraud_index = random.sample(range(0, data_len), num_fraud_seq)

    for i in range(data_len):
        time.sleep(10)
        print(f"\n------------------- Sequence {i} -------------------")
        label = "fraud" if i in fraud_index else "legit"

        data[i] = fraud_builder.generate_one_sequence() if label == "fraud" else legit_builder.generate_one_sequence()

    with open(output_path, 'w') as json_file:
        json.dump(data, json_file, indent=4)


if __name__ == "__main__":
    repo = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description="Generate a dataset of fraud and legit sequences")

    parser.add_argument("--output", type=Path, default= repo / "output" / "datasets" / "coev_dataset_example.json")
    parser.add_argument("--data-len", type=int, default=4)
    parser.add_argument("--num-fraud-seq", type=int, default=2)

    args = parser.parse_args()

    generate_sequences(args.output, args.data_len, args.num_fraud_seq)
