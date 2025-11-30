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

---

## Agentic RAG

Agentic RAG extends traditional RAG with autonomous decision-making capabilities.

### Why Agentic RAG?

| Traditional RAG | Agentic RAG |
|-----------------|-------------|
| Fixed retrieval pipeline | Adaptive query routing |
| Single retrieval attempt | Self-correcting with retries |
| Static context | Dynamic context expansion |
| No multi-hop reasoning | Iterative knowledge gathering |

### Core Patterns

#### 1. Query Router

```python
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

class QueryRouter:
    """Route queries to appropriate retrieval strategies"""
    
    def __init__(self):
        self.llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
        self.router_prompt = ChatPromptTemplate.from_template("""
        Analyze this query and determine the best retrieval strategy.
        
        Query: {query}
        
        Available strategies:
        - "vector": Semantic similarity search for conceptual questions
        - "keyword": BM25/exact match for specific terms, codes, names
        - "hybrid": Combine both for complex queries
        - "graph": Knowledge graph traversal for relationship queries
        - "multi_hop": Sequential retrieval for questions needing multiple sources
        
        Respond with JSON: {{"strategy": "...", "reason": "..."}}
        """)
    
    async def route(self, query: str) -> str:
        response = await self.llm.ainvoke(
            self.router_prompt.format(query=query)
        )
        result = json.loads(response.content)
        return result["strategy"]

# Usage in RAG pipeline
async def adaptive_retrieve(query: str, retrievers: dict):
    router = QueryRouter()
    strategy = await router.route(query)
    
    retriever = retrievers[strategy]
    return await retriever.ainvoke(query)
```

#### 2. Self-Corrective RAG

```python
from langgraph.graph import StateGraph, END

class CRAGState(TypedDict):
    query: str
    documents: list
    generation: str
    grade: str
    retry_count: int

def create_crag_graph():
    """Corrective RAG with grading and fallback"""
    
    graph = StateGraph(CRAGState)
    
    # Nodes
    graph.add_node("retrieve", retrieve_documents)
    graph.add_node("grade_documents", grade_document_relevance)
    graph.add_node("generate", generate_answer)
    graph.add_node("grade_generation", grade_answer_quality)
    graph.add_node("rewrite_query", rewrite_query)
    graph.add_node("web_search", fallback_web_search)
    
    # Flow
    graph.set_entry_point("retrieve")
    graph.add_edge("retrieve", "grade_documents")
    
    # If documents are relevant, generate; else rewrite query
    graph.add_conditional_edges(
        "grade_documents",
        documents_relevant,
        {
            "relevant": "generate",
            "not_relevant": "rewrite_query"
        }
    )
    
    # After query rewrite, try web search as fallback
    graph.add_edge("rewrite_query", "web_search")
    graph.add_edge("web_search", "generate")
    
    # Grade the generation
    graph.add_edge("generate", "grade_generation")
    
    # If generation is good, end; else retry
    graph.add_conditional_edges(
        "grade_generation",
        generation_acceptable,
        {
            "acceptable": END,
            "retry": "rewrite_query"
        }
    )
    
    return graph.compile()

async def grade_document_relevance(state: CRAGState) -> dict:
    """Grade retrieved documents for relevance"""
    
    grader_prompt = """Grade the relevance of this document to the query.
    
    Query: {query}
    Document: {document}
    
    Score from 0-1 where 1 is highly relevant.
    Respond with JSON: {{"score": 0.X, "reason": "..."}}
    """
    
    graded_docs = []
    for doc in state["documents"]:
        response = await llm.ainvoke(
            grader_prompt.format(query=state["query"], document=doc.page_content)
        )
        grade = json.loads(response.content)
        if grade["score"] > 0.5:
            graded_docs.append(doc)
    
    return {"documents": graded_docs}

async def grade_answer_quality(state: CRAGState) -> dict:
    """Check if answer is grounded and addresses the query"""
    
    grader_prompt = """Evaluate this answer:
    
    Query: {query}
    Answer: {generation}
    Source Documents: {documents}
    
    Check:
    1. Is the answer grounded in the documents? (no hallucination)
    2. Does it fully address the query?
    
    Respond with JSON: {{"grounded": true/false, "addresses_query": true/false, "issues": "..."}}
    """
    
    response = await llm.ainvoke(
        grader_prompt.format(
            query=state["query"],
            generation=state["generation"],
            documents=state["documents"]
        )
    )
    
    result = json.loads(response.content)
    
    if result["grounded"] and result["addresses_query"]:
        return {"grade": "acceptable"}
    else:
        return {"grade": "retry", "retry_count": state["retry_count"] + 1}
```

#### 3. Adaptive Retrieval

```python
class AdaptiveRetriever:
    """Dynamically adjust retrieval based on results"""
    
    def __init__(self, vectorstore, k_initial: int = 5, k_max: int = 20):
        self.vectorstore = vectorstore
        self.k_initial = k_initial
        self.k_max = k_max
    
    async def retrieve(self, query: str) -> list:
        k = self.k_initial
        
        while k <= self.k_max:
            docs = await self.vectorstore.asimilarity_search(query, k=k)
            
            # Check coverage
            coverage = await self._assess_coverage(query, docs)
            
            if coverage["sufficient"]:
                return docs
            
            # Need more documents
            k = min(k * 2, self.k_max)
        
        return docs  # Return best effort
    
    async def _assess_coverage(self, query: str, docs: list) -> dict:
        prompt = f"""Does this context sufficiently answer the query?
        
        Query: {query}
        Context: {self._summarize_docs(docs)}
        
        Respond: {{"sufficient": true/false, "missing": "what's missing if any"}}
        """
        
        response = await llm.ainvoke(prompt)
        return json.loads(response.content)
```

#### 4. Multi-Hop Retrieval

```python
async def multi_hop_retrieve(query: str, vectorstore, max_hops: int = 3) -> list:
    """Iterative retrieval for complex multi-part questions"""
    
    all_docs = []
    current_query = query
    
    for hop in range(max_hops):
        # Retrieve for current query
        docs = await vectorstore.asimilarity_search(current_query, k=5)
        all_docs.extend(docs)
        
        # Determine if more hops needed
        next_query = await get_follow_up_query(query, current_query, docs)
        
        if next_query is None:
            break
        
        current_query = next_query
    
    # Deduplicate and rank
    return deduplicate_and_rank(all_docs, query)

async def get_follow_up_query(
    original_query: str, 
    current_query: str, 
    docs: list
) -> str | None:
    """Generate follow-up query if needed"""
    
    prompt = f"""Based on the original question and retrieved information,
    determine if additional retrieval is needed.
    
    Original Question: {original_query}
    Current Search: {current_query}
    Retrieved Info: {summarize(docs)}
    
    If the original question is fully answerable, respond: {{"done": true}}
    If more information is needed, respond: {{"done": false, "next_query": "..."}}
    """
    
    response = await llm.ainvoke(prompt)
    result = json.loads(response.content)
    
    return None if result["done"] else result["next_query"]
```

### Agentic RAG Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                           USER QUERY                                │
└─────────────────────────────────┬───────────────────────────────────┘
                                  │
                                  ▼
┌─────────────────────────────────────────────────────────────────────┐
│                         QUERY ROUTER                                │
│              Analyze → Route → vector/keyword/graph/multi-hop       │
└─────────────────────────────────┬───────────────────────────────────┘
                                  │
                                  ▼
┌─────────────────────────────────────────────────────────────────────┐
│                      ADAPTIVE RETRIEVER                             │
│         Retrieve → Grade → Expand if needed → Re-rank               │
└─────────────────────────────────┬───────────────────────────────────┘
                                  │
                                  ▼
┌─────────────────────────────────────────────────────────────────────┐
│                      CONTEXT PROCESSOR                              │
│          Compress → Reorder → Add metadata → Format                 │
└─────────────────────────────────┬───────────────────────────────────┘
                                  │
                                  ▼
┌─────────────────────────────────────────────────────────────────────┐
│                         GENERATOR                                   │
│              Generate answer with citations                         │
└─────────────────────────────────┬───────────────────────────────────┘
                                  │
                                  ▼
┌─────────────────────────────────────────────────────────────────────┐
│                     ANSWER GRADER                                   │
│      Check: Grounded? Complete? If not → Retry with feedback        │
└─────────────────────────────────┬───────────────────────────────────┘
                                  │
                                  ▼
┌─────────────────────────────────────────────────────────────────────┐
│                       FINAL ANSWER                                  │
└─────────────────────────────────────────────────────────────────────┘
```

---

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

