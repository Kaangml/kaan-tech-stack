# RAG Systems

Retrieval-Augmented Generation architectures for document Q&A, enterprise search, and knowledge management.

## Architecture Patterns

### Classic RAG (Naive RAG)

The simplest and most common pattern:

```
Query → Embedding → Vector Search → Top-K Retrieval → LLM Context → Response
```

**Best For:**
- Simple document Q&A
- Homogeneous document sets
- Quick prototyping

**Limitations:**
- No relationship awareness between documents
- Limited reasoning over multi-hop queries
- Context window constraints

### Advanced RAG

Improvements over naive RAG:

```python
# Example: Multi-stage retrieval pipeline
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import LLMChainExtractor

# Stage 1: Initial broad retrieval
base_retriever = vectorstore.as_retriever(search_kwargs={"k": 20})

# Stage 2: Re-ranking with cross-encoder
from sentence_transformers import CrossEncoder
reranker = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')

# Stage 3: Compression/filtering
compressor = LLMChainExtractor.from_llm(llm)
compression_retriever = ContextualCompressionRetriever(
    base_compressor=compressor,
    base_retriever=base_retriever
)
```

**Key Techniques:**
- Query expansion/rewriting
- Hybrid search (BM25 + semantic)
- Re-ranking with cross-encoders
- Chunk compression

### GraphRAG

Graph-enhanced retrieval for relationship-aware Q&A:

```
Documents → Entity Extraction → Knowledge Graph → Community Detection → 
Graph Traversal + Vector Search → Contextual Response
```

**Architecture:**
```python
# Neo4j + Vector Store hybrid
from langchain_community.graphs import Neo4jGraph
from langchain_community.vectorstores import Qdrant

class GraphRAG:
    def __init__(self):
        self.graph = Neo4jGraph(url, username, password)
        self.vectorstore = Qdrant.from_documents(docs, embeddings)
    
    def retrieve(self, query: str):
        # Vector similarity for initial context
        semantic_results = self.vectorstore.similarity_search(query, k=5)
        
        # Graph traversal for relationships
        entities = self.extract_entities(query)
        graph_context = self.graph.query("""
            MATCH (e:Entity)-[r*1..2]-(related)
            WHERE e.name IN $entities
            RETURN e, r, related
        """, {"entities": entities})
        
        return self.merge_contexts(semantic_results, graph_context)
```

**Best For:**
- Legal document analysis (laws referencing other laws)
- Multi-entity queries ("How is Company A related to Person B?")
- Temporal relationships

### LightRAG

Lightweight alternative focusing on efficiency:

**Key Differences:**
- Smaller embedding models
- Local inference
- Simpler retrieval without heavy re-ranking

```python
# Example: Local LightRAG setup
from sentence_transformers import SentenceTransformer

# Small, fast embedding model
embedder = SentenceTransformer('all-MiniLM-L6-v2')  # 80MB vs 400MB+

# Simple FAISS for vector search (no external DB needed)
import faiss
index = faiss.IndexFlatIP(384)  # Inner product for cosine similarity
```

## Chunking Strategies

### Document Type-Specific Chunking

| Document Type | Strategy | Chunk Size | Overlap |
|--------------|----------|------------|---------|
| Legal contracts | Section-based | Full section | Headers included |
| Technical docs | Markdown headers | 500-1000 tokens | 50-100 |
| Conversations | Turn-based | Per message | Context window |
| Code | Function/class | Logical units | Import statements |

### Semantic Chunking

```python
from langchain.text_splitter import SemanticChunker
from langchain_openai import OpenAIEmbeddings

# Chunks based on semantic similarity, not character count
splitter = SemanticChunker(
    OpenAIEmbeddings(),
    breakpoint_threshold_type="percentile",
    breakpoint_threshold_amount=95
)

chunks = splitter.split_documents(documents)
```

### Legal Document Chunking (My Approach)

```python
# Structure-aware chunking for legislation
def chunk_legal_document(doc: str) -> List[Chunk]:
    sections = []
    
    # Hierarchy: Article > Paragraph > Clause
    article_pattern = r"Madde\s+(\d+)"
    paragraph_pattern = r"\((\d+)\)"
    
    # Keep metadata for cross-references
    for article in re.split(article_pattern, doc):
        chunk = Chunk(
            content=article,
            metadata={
                "type": "article",
                "references": extract_references(article),
                "effective_date": extract_date(article)
            }
        )
        sections.append(chunk)
    
    return sections
```

## Embedding Model Selection

| Model | Dimensions | Speed | Quality | Use Case |
|-------|------------|-------|---------|----------|
| text-embedding-3-small | 1536 | Fast | Good | General purpose |
| text-embedding-3-large | 3072 | Medium | Best | High accuracy needs |
| all-MiniLM-L6-v2 | 384 | Very Fast | Good | Local/edge |
| bge-large-en-v1.5 | 1024 | Medium | Excellent | Open-source alternative |

## Vector Store Selection

### FAISS (Local)
```python
from langchain_community.vectorstores import FAISS

# Best for: Prototyping, smaller datasets (<1M vectors)
vectorstore = FAISS.from_documents(docs, embeddings)
vectorstore.save_local("faiss_index")
```

### Qdrant (Production)
```python
from qdrant_client import QdrantClient
from langchain_community.vectorstores import Qdrant

# Best for: Production, filtering, payloads
client = QdrantClient(url="http://localhost:6333")
vectorstore = Qdrant(
    client=client,
    collection_name="documents",
    embeddings=embeddings
)
```

### ChromaDB (Simple Production)
```python
import chromadb
from langchain_community.vectorstores import Chroma

# Best for: Simple persistence, quick setup
client = chromadb.PersistentClient(path="./chroma_db")
vectorstore = Chroma(
    client=client,
    collection_name="documents",
    embedding_function=embeddings
)
```

## Evaluation Metrics

```python
# RAG evaluation framework
from ragas import evaluate
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_precision,
    context_recall
)

result = evaluate(
    dataset,
    metrics=[
        faithfulness,      # Is answer grounded in context?
        answer_relevancy,  # Is answer relevant to question?
        context_precision, # Are retrieved docs relevant?
        context_recall     # Are all relevant docs retrieved?
    ]
)
```

## Production Considerations

### Caching Layer
```python
from langchain.cache import RedisSemanticCache

# Cache similar queries
langchain.llm_cache = RedisSemanticCache(
    redis_url="redis://localhost:6379",
    embedding=embeddings,
    score_threshold=0.95
)
```

### Hybrid Search
```python
from langchain.retrievers import EnsembleRetriever
from langchain_community.retrievers import BM25Retriever

# Combine keyword and semantic search
bm25_retriever = BM25Retriever.from_documents(docs)
vector_retriever = vectorstore.as_retriever()

ensemble = EnsembleRetriever(
    retrievers=[bm25_retriever, vector_retriever],
    weights=[0.3, 0.7]
)
```

## Related Resources

- [LLM Agents](../llm-agents/README.md) - Agent frameworks for RAG orchestration
- [Vector DBs](../../6-databases/vector-dbs/README.md) - Vector store deep dive
- [Legal RAG Blueprint](../../99-blueprints/legal-rag-graphdb/README.md) - Complete implementation
