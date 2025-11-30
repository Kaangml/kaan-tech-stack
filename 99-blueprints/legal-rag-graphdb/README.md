# Legal RAG with GraphDB

Blueprint for building a legal document Q&A system using RAG with knowledge graphs.

## Architecture

```
┌──────────────────────────────────────────────────────────────────┐
│                        Query Interface                            │
│                    (FastAPI / Streamlit)                          │
└────────────────────────────┬─────────────────────────────────────┘
                             │
                             ▼
┌──────────────────────────────────────────────────────────────────┐
│                       Query Processing                            │
│                                                                   │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────────────┐  │
│  │   Query     │───►│   Hybrid    │───►│    Context          │  │
│  │   Rewrite   │    │   Retrieval │    │    Assembly         │  │
│  └─────────────┘    └─────────────┘    └─────────────────────┘  │
│                             │                                     │
│              ┌──────────────┴──────────────┐                     │
│              ▼                             ▼                     │
│       ┌───────────┐               ┌───────────────┐              │
│       │  Vector   │               │  Knowledge    │              │
│       │  Search   │               │    Graph      │              │
│       │  (Qdrant) │               │   (Neo4j)     │              │
│       └───────────┘               └───────────────┘              │
└──────────────────────────────────────────────────────────────────┘
                             │
                             ▼
┌──────────────────────────────────────────────────────────────────┐
│                        LLM Response                               │
│                     (GPT-4 / Claude)                              │
└──────────────────────────────────────────────────────────────────┘
```

## Data Model

### Knowledge Graph Schema

```cypher
// Core entities
CREATE (law:Law {
    id: "5237",
    title: "Turkish Penal Code",
    enacted_date: date("2004-09-26"),
    effective_date: date("2005-06-01"),
    status: "active"
})

CREATE (article:Article {
    number: 157,
    title: "Fraud",
    text: "Any person who...",
    law_id: "5237"
})

// Relationships
CREATE (article)-[:PART_OF]->(law)
CREATE (article)-[:REFERENCES {type: "related"}]->(other_article)
CREATE (article)-[:AMENDED_BY {date: date("2020-01-01")}]->(amendment)
CREATE (article)-[:SUPERSEDES]->(old_article)

// Court decisions
CREATE (decision:CourtDecision {
    id: "2023/1234",
    court: "Court of Cassation",
    chamber: "Criminal Chamber 6",
    date: date("2023-05-15"),
    summary: "..."
})

CREATE (decision)-[:INTERPRETS]->(article)
CREATE (decision)-[:CITES]->(other_decision)
```

### Document Processing Pipeline

```python
from dataclasses import dataclass
from typing import List, Optional
import re

@dataclass
class LegalChunk:
    id: str
    text: str
    law_id: str
    article_number: Optional[int]
    chunk_type: str  # "article", "paragraph", "definition"
    references: List[str]
    metadata: dict

class LegalDocumentProcessor:
    def __init__(self):
        self.article_pattern = re.compile(r"Madde\s+(\d+)")
        self.reference_pattern = re.compile(r"(\d+)\s+sayılı.*?(\d+)\.?\s*madde")
    
    def process_law(self, text: str, law_id: str) -> List[LegalChunk]:
        chunks = []
        articles = self._split_into_articles(text)
        
        for article in articles:
            # Extract article number
            match = self.article_pattern.search(article)
            article_num = int(match.group(1)) if match else None
            
            # Extract cross-references
            references = self._extract_references(article)
            
            # Create chunk
            chunk = LegalChunk(
                id=f"{law_id}_art_{article_num}",
                text=article,
                law_id=law_id,
                article_number=article_num,
                chunk_type="article",
                references=references,
                metadata={"source": f"Law {law_id}"}
            )
            chunks.append(chunk)
        
        return chunks
    
    def _extract_references(self, text: str) -> List[str]:
        refs = []
        for match in self.reference_pattern.finditer(text):
            law_num, article_num = match.groups()
            refs.append(f"{law_num}_art_{article_num}")
        return refs
```

## Hybrid Retrieval

### Vector + Graph Search

```python
from qdrant_client import QdrantClient
from neo4j import GraphDatabase

class HybridRetriever:
    def __init__(self, qdrant_url: str, neo4j_url: str):
        self.vector_client = QdrantClient(qdrant_url)
        self.graph_driver = GraphDatabase.driver(neo4j_url)
    
    async def retrieve(self, query: str, k: int = 10) -> List[dict]:
        # Parallel retrieval
        vector_results = await self._vector_search(query, k)
        
        # Extract entities from query for graph search
        entities = await self._extract_legal_entities(query)
        graph_results = await self._graph_search(entities)
        
        # Merge and rank results
        combined = self._merge_results(vector_results, graph_results)
        return combined[:k]
    
    async def _vector_search(self, query: str, k: int) -> List[dict]:
        embedding = await get_embedding(query)
        
        results = self.vector_client.search(
            collection_name="legal_docs",
            query_vector=embedding,
            limit=k,
            with_payload=True
        )
        
        return [
            {
                "id": r.id,
                "text": r.payload["text"],
                "score": r.score,
                "source": "vector",
                "metadata": r.payload
            }
            for r in results
        ]
    
    async def _graph_search(self, entities: List[str]) -> List[dict]:
        with self.graph_driver.session() as session:
            # Find articles and their relationships
            query = """
            MATCH (a:Article)
            WHERE a.id IN $article_ids
            
            // Get directly referenced articles
            OPTIONAL MATCH (a)-[:REFERENCES]->(ref:Article)
            
            // Get related court decisions
            OPTIONAL MATCH (d:CourtDecision)-[:INTERPRETS]->(a)
            
            RETURN a, collect(DISTINCT ref) as references, 
                   collect(DISTINCT d) as decisions
            """
            
            result = session.run(query, article_ids=entities)
            
            return [self._format_graph_result(r) for r in result]
    
    def _merge_results(
        self, 
        vector: List[dict], 
        graph: List[dict]
    ) -> List[dict]:
        # Weighted combination
        all_results = {}
        
        for r in vector:
            all_results[r["id"]] = {
                **r,
                "combined_score": r["score"] * 0.6
            }
        
        for r in graph:
            if r["id"] in all_results:
                all_results[r["id"]]["combined_score"] += 0.4
                all_results[r["id"]]["graph_context"] = r.get("context")
            else:
                all_results[r["id"]] = {
                    **r,
                    "combined_score": 0.4
                }
        
        return sorted(
            all_results.values(), 
            key=lambda x: x["combined_score"], 
            reverse=True
        )
```

## Query Processing

### Legal Query Rewriter

```python
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate

REWRITE_PROMPT = """You are a legal research assistant. Rewrite the user's question 
to improve retrieval from a legal document database.

Original question: {question}

Consider:
1. Expand legal terminology
2. Include related legal concepts
3. Specify relevant areas of law

Rewritten question:"""

class LegalQueryRewriter:
    def __init__(self):
        self.llm = ChatOpenAI(model="gpt-4o", temperature=0)
        self.prompt = ChatPromptTemplate.from_template(REWRITE_PROMPT)
    
    async def rewrite(self, question: str) -> str:
        messages = self.prompt.format_messages(question=question)
        response = await self.llm.ainvoke(messages)
        return response.content
```

### Context Assembly

```python
class ContextAssembler:
    def __init__(self, max_tokens: int = 4000):
        self.max_tokens = max_tokens
    
    def assemble(self, results: List[dict]) -> str:
        context_parts = []
        current_tokens = 0
        
        for result in results:
            # Add main article
            article_text = self._format_article(result)
            tokens = count_tokens(article_text)
            
            if current_tokens + tokens > self.max_tokens:
                break
            
            context_parts.append(article_text)
            current_tokens += tokens
            
            # Add referenced articles (summarized)
            if "references" in result:
                for ref in result["references"][:3]:
                    ref_text = self._format_reference(ref)
                    ref_tokens = count_tokens(ref_text)
                    if current_tokens + ref_tokens < self.max_tokens:
                        context_parts.append(ref_text)
                        current_tokens += ref_tokens
        
        return "\n\n---\n\n".join(context_parts)
    
    def _format_article(self, result: dict) -> str:
        return f"""
**Law {result['metadata']['law_id']}, Article {result['metadata']['article_number']}**

{result['text']}

Source: {result['metadata'].get('source', 'Unknown')}
"""
```

## RAG Chain

```python
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate

LEGAL_QA_PROMPT = """You are a legal research assistant specialized in Turkish law.
Answer the question based on the provided legal context.

Context:
{context}

Question: {question}

Instructions:
1. Base your answer strictly on the provided context
2. Cite specific articles and laws when possible
3. If the context doesn't contain enough information, say so
4. Use clear, professional language

Answer:"""

class LegalRAGChain:
    def __init__(self):
        self.retriever = HybridRetriever(QDRANT_URL, NEO4J_URL)
        self.rewriter = LegalQueryRewriter()
        self.assembler = ContextAssembler()
        self.llm = ChatOpenAI(model="gpt-4o", temperature=0)
        self.prompt = ChatPromptTemplate.from_template(LEGAL_QA_PROMPT)
    
    async def answer(self, question: str) -> dict:
        # Rewrite query
        rewritten = await self.rewriter.rewrite(question)
        
        # Retrieve
        results = await self.retriever.retrieve(rewritten)
        
        # Assemble context
        context = self.assembler.assemble(results)
        
        # Generate answer
        messages = self.prompt.format_messages(
            context=context,
            question=question
        )
        response = await self.llm.ainvoke(messages)
        
        return {
            "answer": response.content,
            "sources": [r["metadata"] for r in results[:5]],
            "rewritten_query": rewritten
        }
```

## API Endpoint

```python
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

app = FastAPI()
rag_chain = LegalRAGChain()

class QuestionRequest(BaseModel):
    question: str

class AnswerResponse(BaseModel):
    answer: str
    sources: list
    rewritten_query: str

@app.post("/ask", response_model=AnswerResponse)
async def ask_legal_question(request: QuestionRequest):
    try:
        result = await rag_chain.answer(request.question)
        return AnswerResponse(**result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/article/{law_id}/{article_number}")
async def get_article(law_id: str, article_number: int):
    # Direct article lookup
    with rag_chain.retriever.graph_driver.session() as session:
        result = session.run("""
            MATCH (a:Article {law_id: $law_id, number: $number})
            OPTIONAL MATCH (a)-[:REFERENCES]->(ref)
            RETURN a, collect(ref) as references
        """, law_id=law_id, number=article_number)
        
        record = result.single()
        if not record:
            raise HTTPException(status_code=404, detail="Article not found")
        
        return {
            "article": dict(record["a"]),
            "references": [dict(r) for r in record["references"]]
        }
```

## Related Resources

- [RAG Systems](../../3-ai-ml/rag-systems/README.md) - RAG patterns
- [Graph DBs](../../6-databases/graph-dbs/README.md) - Neo4j deep dive
- [Vector DBs](../../6-databases/vector-dbs/README.md) - Qdrant usage
- [PDF Processing](../../5-python-production/pdf-processing/README.md) - Document extraction
