# Vector Databases

Vector storage and similarity search for AI applications.

## Vector DB Comparison

| Database | Type | Best For | Filtering | Scaling |
|----------|------|----------|-----------|---------|
| FAISS | Library | Prototyping, embedded | Limited | Single machine |
| ChromaDB | Embedded/Server | Simple apps | Good | Moderate |
| Qdrant | Server | Production, rich features | Excellent | Horizontal |
| Pinecone | Managed | Enterprise, zero-ops | Good | Automatic |
| Weaviate | Server | Hybrid search | Excellent | Horizontal |
| Milvus | Server | Large scale | Good | Excellent |

## FAISS

Facebook's library for efficient similarity search.

### Index Types

```python
import faiss
import numpy as np

dimension = 768
n_vectors = 100000
vectors = np.random.random((n_vectors, dimension)).astype('float32')

# Flat index (exact search, brute force)
index_flat = faiss.IndexFlatL2(dimension)  # L2 distance
index_flat = faiss.IndexFlatIP(dimension)  # Inner product (cosine if normalized)
index_flat.add(vectors)

# IVF index (faster, approximate)
nlist = 100  # Number of clusters
quantizer = faiss.IndexFlatL2(dimension)
index_ivf = faiss.IndexIVFFlat(quantizer, dimension, nlist)
index_ivf.train(vectors)  # Required for IVF
index_ivf.add(vectors)

# HNSW index (best recall/speed tradeoff)
index_hnsw = faiss.IndexHNSWFlat(dimension, 32)  # 32 = M parameter
index_hnsw.add(vectors)

# With quantization (memory efficient)
index_pq = faiss.IndexIVFPQ(quantizer, dimension, nlist, 16, 8)
index_pq.train(vectors)
index_pq.add(vectors)
```

### Search Operations

```python
# Search
query = np.random.random((1, dimension)).astype('float32')
k = 10

distances, indices = index_flat.search(query, k)

# Batch search
queries = np.random.random((100, dimension)).astype('float32')
distances, indices = index_flat.search(queries, k)

# Tune IVF search
index_ivf.nprobe = 10  # Search 10 clusters (default 1)

# Save/Load
faiss.write_index(index_flat, "index.faiss")
index = faiss.read_index("index.faiss")
```

### With GPU

```python
# GPU-accelerated search
res = faiss.StandardGpuResources()
gpu_index = faiss.index_cpu_to_gpu(res, 0, index_flat)

# Multi-GPU
gpu_index = faiss.index_cpu_to_all_gpus(index_flat)
```

## ChromaDB

Simple, developer-friendly vector database.

### Setup and Collection

```python
import chromadb
from chromadb.config import Settings

# Persistent client
client = chromadb.PersistentClient(path="./chroma_db")

# Or ephemeral
client = chromadb.Client()

# Create collection
collection = client.create_collection(
    name="documents",
    metadata={"hnsw:space": "cosine"}  # cosine, l2, or ip
)

# Or get existing
collection = client.get_or_create_collection("documents")
```

### CRUD Operations

```python
# Add documents (auto-embedding if embedding_function set)
collection.add(
    ids=["doc1", "doc2", "doc3"],
    documents=["First document", "Second document", "Third document"],
    metadatas=[
        {"source": "wiki", "category": "science"},
        {"source": "wiki", "category": "history"},
        {"source": "arxiv", "category": "science"}
    ],
    embeddings=[[0.1, 0.2, ...], [0.3, 0.4, ...], [0.5, 0.6, ...]]  # Optional
)

# Query
results = collection.query(
    query_embeddings=[[0.1, 0.2, ...]],
    n_results=5,
    where={"category": "science"},
    where_document={"$contains": "neural"}
)

# Update
collection.update(
    ids=["doc1"],
    documents=["Updated first document"],
    metadatas=[{"source": "wiki", "category": "tech"}]
)

# Delete
collection.delete(ids=["doc2"])
collection.delete(where={"category": "history"})
```

### With LangChain

```python
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings

embeddings = OpenAIEmbeddings()

# Create from documents
vectorstore = Chroma.from_documents(
    documents=docs,
    embedding=embeddings,
    persist_directory="./chroma_db"
)

# Search
results = vectorstore.similarity_search("query", k=5)
results_with_scores = vectorstore.similarity_search_with_score("query", k=5)
```

## Qdrant

Production-ready vector database with rich filtering.

### Setup

```python
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance, VectorParams, PointStruct,
    Filter, FieldCondition, MatchValue, Range
)

# Local
client = QdrantClient(path="./qdrant_db")

# Remote
client = QdrantClient(host="localhost", port=6333)

# Cloud
client = QdrantClient(
    url="https://xxx.cloud.qdrant.io",
    api_key="your-api-key"
)
```

### Collection Management

```python
# Create collection
client.create_collection(
    collection_name="documents",
    vectors_config=VectorParams(
        size=768,
        distance=Distance.COSINE
    )
)

# With multiple vector types
from qdrant_client.models import VectorParams

client.create_collection(
    collection_name="hybrid",
    vectors_config={
        "dense": VectorParams(size=768, distance=Distance.COSINE),
        "sparse": VectorParams(size=30000, distance=Distance.DOT)
    }
)
```

### CRUD Operations

```python
# Insert
points = [
    PointStruct(
        id=1,
        vector=[0.1, 0.2, ...],
        payload={
            "text": "Document content",
            "category": "science",
            "date": "2024-01-15",
            "score": 0.95
        }
    ),
    PointStruct(id=2, vector=[0.3, 0.4, ...], payload={...})
]

client.upsert(collection_name="documents", points=points)

# Search
results = client.search(
    collection_name="documents",
    query_vector=[0.1, 0.2, ...],
    limit=10
)

# Search with filtering
results = client.search(
    collection_name="documents",
    query_vector=[0.1, 0.2, ...],
    query_filter=Filter(
        must=[
            FieldCondition(key="category", match=MatchValue(value="science")),
            FieldCondition(key="score", range=Range(gte=0.8))
        ]
    ),
    limit=10
)
```

### Advanced Features

```python
# Hybrid search (dense + sparse)
from qdrant_client.models import NamedVector, NamedSparseVector

results = client.search(
    collection_name="hybrid",
    query_vector=NamedVector(
        name="dense",
        vector=[0.1, 0.2, ...]
    ),
    with_payload=True,
    limit=10
)

# Batch search
from qdrant_client.models import SearchRequest

results = client.search_batch(
    collection_name="documents",
    requests=[
        SearchRequest(vector=[0.1, 0.2, ...], limit=5),
        SearchRequest(vector=[0.3, 0.4, ...], limit=5)
    ]
)

# Scroll (iterate all points)
offset = None
while True:
    points, offset = client.scroll(
        collection_name="documents",
        limit=100,
        offset=offset,
        with_payload=True
    )
    if not points:
        break
    process(points)
```

## Embedding Strategies

### Choosing Embedding Models

| Model | Dimensions | Speed | Quality | Use Case |
|-------|------------|-------|---------|----------|
| text-embedding-3-small | 1536 | Fast | Good | General |
| text-embedding-3-large | 3072 | Medium | Best | High accuracy |
| all-MiniLM-L6-v2 | 384 | Very Fast | Good | Local/edge |
| bge-large-en-v1.5 | 1024 | Medium | Excellent | Open source |

### Dimensionality Reduction

```python
# Reduce dimensions for faster search
from sklearn.decomposition import PCA

pca = PCA(n_components=256)
reduced_vectors = pca.fit_transform(vectors)

# Matryoshka embeddings (OpenAI)
# text-embedding-3-* supports variable dimensions
response = client.embeddings.create(
    model="text-embedding-3-small",
    input="text",
    dimensions=256  # Reduce from 1536
)
```

## Production Patterns

### Caching Layer

```python
import hashlib
import redis

redis_client = redis.Redis()

def get_embedding_cached(text: str, model: str = "text-embedding-3-small"):
    cache_key = f"emb:{model}:{hashlib.md5(text.encode()).hexdigest()}"
    
    cached = redis_client.get(cache_key)
    if cached:
        return np.frombuffer(cached, dtype=np.float32)
    
    embedding = create_embedding(text, model)
    redis_client.set(cache_key, embedding.tobytes(), ex=86400)  # 24h TTL
    return embedding
```

### Hybrid Search

```python
from langchain.retrievers import EnsembleRetriever
from langchain_community.retrievers import BM25Retriever

# Combine keyword and semantic search
bm25 = BM25Retriever.from_documents(docs)
vector = vectorstore.as_retriever()

ensemble = EnsembleRetriever(
    retrievers=[bm25, vector],
    weights=[0.3, 0.7]
)

results = ensemble.get_relevant_documents("query")
```

## Related Resources

- [RAG Systems](../../3-ai-ml/rag-systems/README.md) - Using vector DBs in RAG
- [LLM Agents](../../3-ai-ml/llm-agents/README.md) - Agent memory with vectors
- [Legal RAG Blueprint](../../99-blueprints/legal-rag-graphdb/README.md) - Production implementation
