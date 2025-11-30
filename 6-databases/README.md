# Databases

Vector, graph, and relational database patterns.

## Topics

| Type | Description |
|------|-------------|
| [Vector DBs](./vector-dbs/) | FAISS, Qdrant, ChromaDB for AI |
| [Graph DBs](./graph-dbs/) | Neo4j, Cypher, knowledge graphs |
| [Relational](./postgres-advanced/) | PostgreSQL, MySQL, advanced SQL |

## When to Use What

| Use Case | Database Type |
|----------|---------------|
| Semantic search | Vector (Qdrant, FAISS) |
| Relationships | Graph (Neo4j) |
| Transactions | Relational (PostgreSQL) |
| Analytics | OLAP (ClickHouse, DuckDB) |
| Documents | Document (MongoDB) |

## Quick Comparison

```
Vector DBs   → Similarity search, embeddings, RAG
Graph DBs    → Relationships, traversals, knowledge graphs
Relational   → ACID, complex queries, transactions
```

## Related Blueprints
- [Legal RAG](../99-blueprints/legal-rag-graphdb/) - Neo4j + Qdrant hybrid
- [RAG Systems](../3-ai-ml/rag-systems/) - Vector DB selection
