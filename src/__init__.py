"""
Fraud coevolution research package.

- ``generate_sequences`` — synthetic fraud and legit payment sequences (Neo4j + LLM).
- ``validators`` — LLM checks that sequences match fraud vs legit narratives.
- ``detectors`` — classifiers / evaluation (e.g. fraud vs legit on datasets).
- ``analysis`` — LLM-assisted mining of patterns from labeled data.

Top-level modules: ``build_graph`` (Neo4j seed from JSON), ``build_env`` (LLM JSON generation).

Static data layout (repo root): ``data/graph/*.json`` (Neo4j + ``victim_actions`` including
``legit_actions``), ``data/llm/guidelines.json``.
"""
