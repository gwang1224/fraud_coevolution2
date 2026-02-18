from neo4j import GraphDatabase
import json

NEO4J_URI="neo4j+s://40d1b041.databases.neo4j.io"
NEO4J_USERNAME="neo4j"
NEO4J_PASSWORD="rguy8ZrmKEM4eab03Auccidyrhv0ZBYb6ki3ZQkfdl8"

class Neo4jApp:
    def __init__(self, uri, user, password):
        self.driver = GraphDatabase.driver(uri, auth=(user, password))
        self.driver.verify_connectivity() # Ensures a working connection

    def close(self):
        """Close the driver connection when done."""
        self.driver.close()

    def delete_all_nodes(self):
        """Delete all nodes"""
        query = "MATCH (n) DETACH DELETE n"
        nodes_deleted_count = self.driver.execute_query(query)
        print(f"Deleted {nodes_deleted_count} nodes.")

    def add_victim_acc_nodes(self, file):
        with open(file, 'r') as file:
            data = json.load(file)
            for person in data['individuals']:
                query = """
                    CREATE (p:Person {name: $name})-[r:OWNS]->(a:Account {acc_name: $acc_name, owner: $owner, bank: $bank, balance: $balance})
                    RETURN p.name AS name, a.name AS acc_name
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
                print(f"Created Person: {result[0]['name']}, Created Account: {result[0]['acc_name']}")

    


neo4j = Neo4jApp(NEO4J_URI, NEO4J_USERNAME, NEO4J_PASSWORD)
neo4j.delete_all_nodes()
neo4j.add_victim_acc_nodes("victims.json")