# Graph Databases

Graph databases for relationship-rich data, with focus on Neo4j.

## When to Use Graph DBs

**Good Fit:**
- Complex relationships (social networks, fraud detection)
- Hierarchical data (org charts, taxonomies)
- Recommendation engines
- Knowledge graphs
- Path finding / network analysis

**Not Ideal:**
- Simple CRUD operations
- Time-series data
- Document storage
- High-volume writes

## Neo4j

### Core Concepts

- **Nodes**: Entities (like rows)
- **Labels**: Node types (like table names)
- **Relationships**: Connections between nodes
- **Properties**: Key-value pairs on nodes/relationships

### Cypher Query Language

```cypher
-- Create nodes
CREATE (p:Person {name: 'Alice', age: 30})
CREATE (c:Company {name: 'TechCorp', industry: 'Technology'})

-- Create relationship
MATCH (p:Person {name: 'Alice'})
MATCH (c:Company {name: 'TechCorp'})
CREATE (p)-[:WORKS_AT {since: 2020, role: 'Engineer'}]->(c)

-- Find nodes
MATCH (p:Person) WHERE p.age > 25 RETURN p

-- Find with relationships
MATCH (p:Person)-[r:WORKS_AT]->(c:Company)
RETURN p.name, r.role, c.name

-- Traversal (friends of friends)
MATCH (p:Person {name: 'Alice'})-[:FRIEND*2]->(fof:Person)
WHERE fof <> p
RETURN DISTINCT fof.name

-- Shortest path
MATCH path = shortestPath(
  (p1:Person {name: 'Alice'})-[*]-(p2:Person {name: 'Bob'})
)
RETURN path

-- Pattern matching
MATCH (p:Person)-[:WORKS_AT]->(c:Company)<-[:WORKS_AT]-(colleague:Person)
WHERE p.name = 'Alice' AND colleague <> p
RETURN colleague.name AS colleagues
```

### Python Integration

```python
from neo4j import GraphDatabase

class Neo4jConnection:
    def __init__(self, uri: str, user: str, password: str):
        self.driver = GraphDatabase.driver(uri, auth=(user, password))
    
    def close(self):
        self.driver.close()
    
    def query(self, query: str, parameters: dict = None):
        with self.driver.session() as session:
            result = session.run(query, parameters or {})
            return [record.data() for record in result]
    
    def write(self, query: str, parameters: dict = None):
        with self.driver.session() as session:
            result = session.write_transaction(
                lambda tx: tx.run(query, parameters or {})
            )
            return result

# Usage
conn = Neo4jConnection("bolt://localhost:7687", "neo4j", "password")

# Create data
conn.write("""
    CREATE (p:Person {name: $name, email: $email})
    RETURN p
""", {"name": "Alice", "email": "alice@example.com"})

# Query
results = conn.query("""
    MATCH (p:Person)-[:WORKS_AT]->(c:Company)
    WHERE c.name = $company
    RETURN p.name, p.email
""", {"company": "TechCorp"})

conn.close()
```

### Indexing

```cypher
-- Create index for faster lookups
CREATE INDEX person_name FOR (p:Person) ON (p.name)

-- Composite index
CREATE INDEX person_composite FOR (p:Person) ON (p.name, p.email)

-- Unique constraint (also creates index)
CREATE CONSTRAINT person_email_unique FOR (p:Person) REQUIRE p.email IS UNIQUE

-- Full-text index
CREATE FULLTEXT INDEX person_search FOR (p:Person) ON EACH [p.name, p.bio]

-- Query full-text
CALL db.index.fulltext.queryNodes('person_search', 'alice OR engineer') 
YIELD node, score
RETURN node.name, score
```

### Graph Algorithms

```python
# Using Neo4j Graph Data Science library
from graphdatascience import GraphDataScience

gds = GraphDataScience("bolt://localhost:7687", auth=("neo4j", "password"))

# Create in-memory graph projection
gds.graph.project(
    "social",
    "Person",
    "FRIEND"
)

# PageRank
result = gds.pageRank.stream("social")
print(result[["nodeId", "score"]])

# Community detection (Louvain)
result = gds.louvain.stream("social")
print(result[["nodeId", "communityId"]])

# Node similarity
result = gds.nodeSimilarity.stream("social")
print(result[["node1", "node2", "similarity"]])

# Cleanup
gds.graph.drop("social")
```

## Knowledge Graph Patterns

### Entity-Relationship Model

```cypher
-- Legal documents example
CREATE (law:Law {id: 'law_5237', title: 'Turkish Penal Code', year: 2004})
CREATE (article:Article {number: 157, title: 'Fraud', text: '...'})
CREATE (article)-[:PART_OF]->(law)

-- Cross-references
MATCH (a1:Article {number: 157})
MATCH (a2:Article {number: 158})
CREATE (a1)-[:REFERENCES {type: 'aggravation'}]->(a2)

-- Find all referenced articles
MATCH (a:Article {number: 157})-[:REFERENCES*1..3]->(referenced)
RETURN referenced.number, referenced.title
```

### Temporal Patterns

```cypher
-- Version tracking
CREATE (law:Law {id: 'law_5237'})
CREATE (v1:Version {number: 1, effective_date: date('2005-06-01')})
CREATE (v2:Version {number: 2, effective_date: date('2020-01-01')})
CREATE (law)-[:HAS_VERSION]->(v1)
CREATE (law)-[:HAS_VERSION]->(v2)
CREATE (v1)-[:SUPERSEDED_BY]->(v2)

-- Get current version
MATCH (law:Law {id: 'law_5237'})-[:HAS_VERSION]->(v:Version)
WHERE NOT (v)-[:SUPERSEDED_BY]->()
RETURN v
```

### LangChain Integration

```python
from langchain_community.graphs import Neo4jGraph
from langchain_openai import ChatOpenAI
from langchain.chains import GraphCypherQAChain

graph = Neo4jGraph(
    url="bolt://localhost:7687",
    username="neo4j",
    password="password"
)

llm = ChatOpenAI(model="gpt-4", temperature=0)

chain = GraphCypherQAChain.from_llm(
    llm=llm,
    graph=graph,
    verbose=True,
    allow_dangerous_requests=True  # Required for write operations
)

response = chain.invoke({
    "query": "Who works at TechCorp and what are their roles?"
})
print(response)
```

## Best Practices

### Data Modeling

```cypher
-- Good: Relationships as first-class citizens
MATCH (p:Person)-[r:PURCHASED {date: date(), amount: 100}]->(product:Product)

-- Avoid: Storing relationships as properties
-- Bad: (p:Person {purchased_products: ['prod1', 'prod2']})
```

### Query Optimization

```cypher
-- Use parameters (prevents query recompilation)
MATCH (p:Person {name: $name}) RETURN p

-- Profile queries
PROFILE MATCH (p:Person)-[:WORKS_AT]->(c:Company) RETURN p, c

-- Limit early
MATCH (p:Person) 
WHERE p.age > 25 
WITH p LIMIT 100
MATCH (p)-[:FRIEND]->(f)
RETURN p, f
```

### Transaction Patterns

```python
def transfer_friendship(tx, from_person, to_person, friend):
    tx.run("""
        MATCH (from:Person {name: $from})-[r:FRIEND]->(f:Person {name: $friend})
        DELETE r
        WITH f
        MATCH (to:Person {name: $to})
        CREATE (to)-[:FRIEND]->(f)
    """, from=from_person, to=to_person, friend=friend)

with driver.session() as session:
    session.execute_write(transfer_friendship, "Alice", "Bob", "Charlie")
```

## Related Resources

- [RAG Systems](../../3-ai-ml/rag-systems/README.md) - GraphRAG implementation
- [Vector DBs](../vector-dbs/README.md) - Hybrid graph + vector search
- [Legal RAG Blueprint](../../99-blueprints/legal-rag-graphdb/README.md) - Production graph use case
