

### Prerequisites

1. **Neo4j Database**: Install and run Neo4j locally or use Neo4j Aura
   - Default connection: `bolt://localhost:7687`
   - Default credentials: `neo4j` / `password` (change in `build_env.py`)

2. **Ollama**: Install and run Ollama with a compatible model
   - Install from: https://ollama.ai
   - Default model: `llama2` (can be changed in `OllamaClient`)
   - Ensure Ollama is running on `http://localhost:11434`

### Installation

```bash
pip install -r requirements.txt
```

## Usage

### Basic Usage

```python
from build_env import Neo4jEnvironmentBuilder

# Initialize builder
builder = Neo4jEnvironmentBuilder(
    uri="bolt://localhost:7687",
    user="neo4j",
    password="your_password"
)

# Generate and build a single scenario
actions, transactions = builder.generate_and_build_scenario()

# Generate multiple scenarios
scenarios = builder.generate_multiple_scenarios(10)

# Don't forget to close the connection
builder.close()
```

### Running the Example

```bash
python build_env.py
```

## Data Model

### Nodes

- **:Actor** - People and fraudsters
  - Properties: `name`, `role` (victim, fraudster)

- **:Account** - Bank accounts
  - Properties: `name`, `owner`, `bank`, `balance`

- **:Institution** - Banks, telecoms, merchants, government
  - Properties: `name`, `type` (bank, telecom, merchant, government)

- **:ActionType** - Types of actions
  - Properties: `name` (e.g., phishing, impersonation, sim_swap, credential_submission)

- **:Channel** - Communication channels
  - Properties: `name` (e.g., email, sms, call, app, website)

### Relationships

- `(a:Actor)-[:OWNS]->(acc:Account)`
- `(acc:Account)-[:HELD_AT]->(i:Institution {type:"bank"})`
- `(fraud:Actor)-[:USES_ACTION]->(t:ActionType)`
- `(victim:Actor)-[:RESPONDS_WITH]->(t:ActionType)` (optional)
- `(t:ActionType)-[:COMMON_CHANNEL]->(c:Channel)`

## Sequence Format

The system generates sequences in two formats:

1. **Action Sequences**: `Action(entity1, action, entity2, channel)`
   - Example: `Action(John Smith, phishing, Alice Brown, email)`

2. **Transaction Sequences**: `Transaction(entity1, action, entity2, amount)`
   - Example: `Transaction(John Smith, transfer, ACC-1234, 5000.00)`

## Configuration

- **Ollama Model**: Change `self.model` in `OllamaClient.__init__()`
- **Ollama URL**: Change `base_url` parameter in `OllamaClient.__init__()`
- **Neo4j Connection**: Update URI, user, and password in `main()` or when creating `Neo4jEnvironmentBuilder`
