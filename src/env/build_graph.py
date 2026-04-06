"""
Populate Neo4j from JSON **data** files (victims, fraudsters, fraudster actions, victim actions).

``Neo4jApp`` reads JSON under ``data/graph/`` (``victims.json``, ``fraudsters.json``,
``fraudster_actions.json``, ``victim_actions.json``) relative to ``database_path``, creates person and
account nodes, and wires ``PERFORMS`` / ``TARGETS`` relationships for the fraud simulation
graph used by sequence generators.

Environment: ``NEO4J_URI``, ``NEO4J_USERNAME``, ``NEO4J_PASSWORD`` (via ``python-dotenv``).

This loader does **not** ingest ``legit_actions`` inside ``victim_actions.json``; those rows
are for ``build_legit_sequence`` only.
"""

from neo4j import GraphDatabase
import json
import os
from dotenv import load_dotenv
from pathlib import Path


load_dotenv()

NEO4J_URI = os.getenv("NEO4J_URI")
NEO4J_USERNAME = os.getenv("NEO4J_USERNAME")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD")

class Neo4jApp:
    def __init__(self, uri, user, password, database_path):
        self.driver = GraphDatabase.driver(uri, auth=(user, password))
        self.driver.verify_connectivity()
        self.db_path = Path(database_path)


    def close(self):
        """Close the driver connection when done."""
        self.driver.close()

    def delete_all_nodes(self):
        """Delete all nodes"""
        query = "MATCH (n) DETACH DELETE n"
        nodes_deleted_count = self.driver.execute_query(query)
        print(f"Deleted {nodes_deleted_count} nodes.")

    def add_entity_acc_nodes(self, file_name: str):
        """
        Add victim accounts to the graph database
        
        :param file: file path
        :param label: label of the node
        :type label: str
        """
        label = "victims" if "victim" in file_name.lower() else "fraudsters"
        with open(self.db_path / file_name, 'r') as file:
            data = json.load(file)
            for person in data[label]:
                query = f"""
                    CREATE (p:{label} {{name: $name}})
                        -[r:OWNS]->
                        (a:account {{
                            acc_name: $acc_name, 
                            owner: $owner, 
                            bank: $bank, 
                            balance: $balance
                        }})
                    RETURN p.name AS name, a.acc_name AS acc_name
                """

                result, summary, keys = self.driver.execute_query(
                    query,
                    name=person.get("name"),
                    acc_name=person.get("account").get("acc_name"),
                    owner=person.get("name"),
                    bank=person.get("account").get("bank"),
                    balance=person.get("account").get("balance"),
                    database_="neo4j",
                )
                print(result)
                print(f"Created Person: {result[0]['name']}, Created Account: {result[0]['acc_name']}")
    
    def add_fraudster_actions_nodes(self, file_name):
        with open(self.db_path / file_name, 'r') as file:
            data = json.load(file)['fraudster_actions']
            
            total_actions = 0

            for faction in data:
                query = """
                    CREATE (a:action {action: $action, channels: $channels, description: $description, stage: $stage, initiator: $initiator, target: $target, compromises_account: $compromises_account, is_terminal: $is_terminal})
                    WITH a
                    MATCH (x:fraudsters)
                    MATCH (y:victims)
                    MERGE (x) - [:PERFORMS] -> (a) - [:TARGETS] -> (y)
                    RETURN a.action as action
                """
                result, summary, keys = self.driver.execute_query(
                    query,
                    action=faction.get("name"),
                    channels=faction.get("common_channels"),
                    description=faction.get("description"),
                    stage=faction.get("stage"),
                    initiator=faction.get("initiator"),
                    target=faction.get("target"),
                    compromises_account=faction.get("compromises_account"),
                    is_terminal=faction.get("is_terminal")
                )
                total_actions += 1

            print("Total actions added to graph: " + str(total_actions))

    def add_victim_actions_nodes(self, file_name):
        with open(self.db_path / file_name, 'r') as file:
            data = json.load(file)["victim_actions"]
            
            total_actions = 0

            for vaction in data:
                query = """
                    CREATE (a:action {action: $action, channels: $channels, description: $description, stage: $stage, initiator: $initiator, target: $target, is_terminal: $is_terminal})
                    WITH a
                    MATCH (x:victims)
                    MATCH (y:fraudsters)
                    MERGE (x) - [:PERFORMS] -> (a) - [:TARGETS] -> (y)
                    RETURN a.action as action
                """
                result, summary, keys = self.driver.execute_query(
                    query,
                    action=vaction.get("name"),
                    channels=vaction.get("common_channels"),
                    description=vaction.get("description"),
                    stage=vaction.get("stage"),
                    initiator=vaction.get("initiator"),
                    target=vaction.get("target"),
                    is_terminal=vaction.get("is_terminal"),
                )
                total_actions += 1

            print("Total actions added to graph: " + str(total_actions))

            
            
    
def main():
    neo4j = Neo4jApp(NEO4J_URI, NEO4J_USERNAME, NEO4J_PASSWORD, "data/graph/")
    neo4j.delete_all_nodes()
    neo4j.add_entity_acc_nodes("victims.json")
    neo4j.add_entity_acc_nodes("fraudsters.json")
    neo4j.add_fraudster_actions_nodes("fraudster_actions.json")
    neo4j.add_victim_actions_nodes("victim_actions.json")


if __name__ == "__main__":
    main()