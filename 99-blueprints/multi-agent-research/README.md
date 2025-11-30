# Multi-Agent Research System

A production-ready architecture for autonomous research using coordinated AI agents.

## Overview

This blueprint implements a **Supervisor-Worker** multi-agent system where specialized agents collaborate to conduct comprehensive research, analyze information, and produce synthesized reports.

```
┌────────────────────────────────────────────────────────────────────────────┐
│                              USER QUERY                                    │
│                    "Analyze the AI agent framework landscape"              │
└─────────────────────────────────┬──────────────────────────────────────────┘
                                  │
                                  ▼
┌────────────────────────────────────────────────────────────────────────────┐
│                           🎯 SUPERVISOR                                    │
│              Orchestrates workflow, delegates tasks, synthesizes           │
└─────────────────────────────────┬──────────────────────────────────────────┘
                                  │
          ┌───────────────────────┼───────────────────────┐
          │                       │                       │
          ▼                       ▼                       ▼
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│   🔍 RESEARCHER │     │   🔬 ANALYST    │     │   ✍️ WRITER     │
│   Web search,   │     │   Data analysis │     │   Report gen,   │
│   doc retrieval │     │   comparisons   │     │   formatting    │
└─────────────────┘     └─────────────────┘     └─────────────────┘
          │                       │                       │
          └───────────────────────┼───────────────────────┘
                                  │
                                  ▼
┌────────────────────────────────────────────────────────────────────────────┐
│                          📊 SHARED MEMORY                                  │
│              Research findings, facts, intermediate results                │
└────────────────────────────────────────────────────────────────────────────┘
```

## Technology Stack

| Component | Technology | Purpose |
|-----------|------------|---------|
| Orchestration | LangGraph | State machine, cyclic flows |
| Agents | LangChain | Tool-using agents |
| LLM | GPT-4o / Claude 3.5 | Reasoning, generation |
| Search | Tavily / SerpAPI | Web research |
| Memory | Redis + Qdrant | Working + semantic memory |
| Observability | LangSmith | Tracing, debugging |

## Architecture Components

### 1. State Schema

```python
from typing import TypedDict, Annotated, Literal
from operator import add
from pydantic import BaseModel

class ResearchState(TypedDict):
    # User input
    query: str
    research_type: Literal["market", "technical", "competitive", "general"]
    
    # Workflow control
    current_phase: Literal["planning", "researching", "analyzing", "writing", "review"]
    iteration: int
    max_iterations: int
    
    # Agent outputs (accumulate with add)
    research_findings: Annotated[list[dict], add]
    analysis_results: Annotated[list[dict], add]
    
    # Memory
    facts_learned: Annotated[list[str], add]
    sources_used: Annotated[list[str], add]
    
    # Final output
    report_draft: str
    final_report: str
    
    # Control
    needs_more_research: bool
    supervisor_notes: str
    error: str | None
```

### 2. Agent Definitions

```python
from langchain_openai import ChatOpenAI
from langchain.agents import create_tool_calling_agent, AgentExecutor
from langchain_core.prompts import ChatPromptTemplate

# === RESEARCHER AGENT ===
researcher_prompt = ChatPromptTemplate.from_messages([
    ("system", """You are a Research Specialist with expertise in finding accurate, 
    up-to-date information from diverse sources.
    
    Your responsibilities:
    - Search the web for relevant information
    - Retrieve and analyze documents
    - Extract key facts and cite sources
    - Identify gaps in knowledge that need further research
    
    Always prioritize:
    1. Accuracy over speed
    2. Primary sources over secondary
    3. Recent information over outdated
    
    Current research context:
    {research_context}
    
    Previously gathered facts:
    {facts_learned}
    """),
    ("human", "{task}"),
    ("placeholder", "{agent_scratchpad}")
])

researcher_tools = [
    TavilySearchTool(),
    WebScraperTool(),
    ArxivSearchTool(),
    DocumentRetrieverTool()
]

researcher_agent = create_tool_calling_agent(
    llm=ChatOpenAI(model="gpt-4o", temperature=0),
    tools=researcher_tools,
    prompt=researcher_prompt
)

researcher_executor = AgentExecutor(
    agent=researcher_agent,
    tools=researcher_tools,
    max_iterations=10,
    return_intermediate_steps=True
)

# === ANALYST AGENT ===
analyst_prompt = ChatPromptTemplate.from_messages([
    ("system", """You are a Senior Data Analyst specializing in synthesizing 
    complex information into actionable insights.
    
    Your responsibilities:
    - Analyze research findings for patterns and trends
    - Create comparisons and frameworks
    - Identify strengths, weaknesses, opportunities, threats
    - Generate data visualizations (as text/tables)
    
    Analysis guidelines:
    - Be objective and data-driven
    - Acknowledge limitations and biases
    - Provide confidence levels for conclusions
    
    Research findings to analyze:
    {research_findings}
    """),
    ("human", "{task}"),
    ("placeholder", "{agent_scratchpad}")
])

analyst_tools = [
    PythonCodeExecutor(),  # For data analysis
    ChartGenerator(),
    ComparisonFramework()
]

# === WRITER AGENT ===
writer_prompt = ChatPromptTemplate.from_messages([
    ("system", """You are a Technical Writer who transforms complex research 
    and analysis into clear, engaging reports.
    
    Your responsibilities:
    - Structure information logically
    - Write clear executive summaries
    - Create detailed sections with proper citations
    - Format for readability (headers, bullets, tables)
    
    Writing style:
    - Professional but accessible
    - Active voice preferred
    - Concrete examples over abstract statements
    - Always cite sources inline
    
    Available content:
    Research: {research_findings}
    Analysis: {analysis_results}
    """),
    ("human", "{task}"),
    ("placeholder", "{agent_scratchpad}")
])
```

### 3. Supervisor Logic

```python
from langgraph.graph import StateGraph, END

class Supervisor:
    def __init__(self, llm):
        self.llm = llm
        self.decision_prompt = """You are the Supervisor of a research team.
        
        Current state:
        - Query: {query}
        - Phase: {current_phase}
        - Iteration: {iteration}/{max_iterations}
        - Research findings: {num_findings}
        - Analysis complete: {analysis_done}
        
        Recent findings summary:
        {recent_findings}
        
        Decide the next action:
        1. "research" - Need more information
        2. "analyze" - Have enough data, need analysis
        3. "write" - Analysis complete, generate report
        4. "review" - Report drafted, needs review
        5. "complete" - Report is satisfactory
        
        Consider:
        - Is the research comprehensive enough?
        - Are there gaps in the analysis?
        - Does the report meet quality standards?
        
        Respond with JSON: {{"next": "action", "reason": "explanation", "notes": "guidance for next agent"}}
        """
    
    async def decide(self, state: ResearchState) -> dict:
        response = await self.llm.ainvoke(
            self.decision_prompt.format(
                query=state["query"],
                current_phase=state["current_phase"],
                iteration=state["iteration"],
                max_iterations=state["max_iterations"],
                num_findings=len(state["research_findings"]),
                analysis_done=bool(state["analysis_results"]),
                recent_findings=self._summarize_recent(state["research_findings"])
            )
        )
        
        decision = json.loads(response.content)
        
        return {
            "current_phase": decision["next"],
            "supervisor_notes": decision["notes"],
            "iteration": state["iteration"] + 1
        }
    
    def _summarize_recent(self, findings: list, n: int = 5) -> str:
        recent = findings[-n:] if len(findings) > n else findings
        return "\n".join([f"- {f.get('summary', str(f))}" for f in recent])
```

### 4. LangGraph Workflow

```python
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver

def create_research_graph() -> StateGraph:
    # Initialize components
    supervisor = Supervisor(ChatOpenAI(model="gpt-4o"))
    researcher = ResearcherNode(researcher_executor)
    analyst = AnalystNode(analyst_executor)
    writer = WriterNode(writer_executor)
    reviewer = ReviewerNode()
    
    # Create graph
    graph = StateGraph(ResearchState)
    
    # Add nodes
    graph.add_node("supervisor", supervisor.decide)
    graph.add_node("researcher", researcher.execute)
    graph.add_node("analyst", analyst.execute)
    graph.add_node("writer", writer.execute)
    graph.add_node("reviewer", reviewer.execute)
    
    # Entry point
    graph.set_entry_point("supervisor")
    
    # Supervisor routes to appropriate agent
    graph.add_conditional_edges(
        "supervisor",
        route_by_phase,
        {
            "research": "researcher",
            "analyze": "analyst",
            "write": "writer",
            "review": "reviewer",
            "complete": END
        }
    )
    
    # All agents return to supervisor
    for node in ["researcher", "analyst", "writer", "reviewer"]:
        graph.add_edge(node, "supervisor")
    
    # Add memory checkpoint
    memory = MemorySaver()
    
    return graph.compile(checkpointer=memory)

def route_by_phase(state: ResearchState) -> str:
    phase = state["current_phase"]
    
    # Check iteration limit
    if state["iteration"] >= state["max_iterations"]:
        return "complete"
    
    # Check for errors
    if state.get("error"):
        return "complete"  # Exit on error
    
    return phase


# === AGENT NODE IMPLEMENTATIONS ===

class ResearcherNode:
    def __init__(self, executor: AgentExecutor):
        self.executor = executor
    
    async def execute(self, state: ResearchState) -> dict:
        # Prepare context for researcher
        context = {
            "research_context": state["query"],
            "facts_learned": state["facts_learned"],
            "task": state["supervisor_notes"] or f"Research: {state['query']}"
        }
        
        try:
            result = await self.executor.ainvoke(context)
            
            # Extract findings and sources
            findings = self._parse_findings(result)
            sources = self._extract_sources(result)
            
            return {
                "research_findings": findings,
                "sources_used": sources,
                "facts_learned": [f["key_fact"] for f in findings if f.get("key_fact")]
            }
        
        except Exception as e:
            return {"error": f"Research failed: {str(e)}"}
    
    def _parse_findings(self, result) -> list[dict]:
        # Parse agent output into structured findings
        return [{
            "summary": result["output"],
            "timestamp": datetime.now().isoformat(),
            "confidence": 0.8
        }]


class AnalystNode:
    def __init__(self, executor: AgentExecutor):
        self.executor = executor
    
    async def execute(self, state: ResearchState) -> dict:
        findings_text = self._format_findings(state["research_findings"])
        
        result = await self.executor.ainvoke({
            "research_findings": findings_text,
            "task": state["supervisor_notes"] or "Analyze the research findings"
        })
        
        return {
            "analysis_results": [{
                "analysis": result["output"],
                "type": state["research_type"],
                "timestamp": datetime.now().isoformat()
            }]
        }


class WriterNode:
    def __init__(self, executor: AgentExecutor):
        self.executor = executor
    
    async def execute(self, state: ResearchState) -> dict:
        result = await self.executor.ainvoke({
            "research_findings": state["research_findings"],
            "analysis_results": state["analysis_results"],
            "task": state["supervisor_notes"] or "Write a comprehensive report"
        })
        
        return {
            "report_draft": result["output"]
        }


class ReviewerNode:
    async def execute(self, state: ResearchState) -> dict:
        # Quality check the draft
        review_prompt = f"""Review this research report for:
        1. Accuracy of claims (cross-reference with sources)
        2. Completeness (does it answer the original query?)
        3. Clarity and structure
        4. Missing information or gaps
        
        Original Query: {state['query']}
        
        Report Draft:
        {state['report_draft']}
        
        Sources Used:
        {state['sources_used']}
        
        Provide: approval (yes/no), feedback, and if approved, the final polished report.
        """
        
        llm = ChatOpenAI(model="gpt-4o", temperature=0)
        response = await llm.ainvoke(review_prompt)
        
        review = self._parse_review(response.content)
        
        if review["approved"]:
            return {
                "final_report": review["final_report"],
                "current_phase": "complete"
            }
        else:
            return {
                "needs_more_research": True,
                "supervisor_notes": review["feedback"]
            }
```

### 5. Shared Memory System

```python
import redis
from qdrant_client import QdrantClient
from langchain_openai import OpenAIEmbeddings

class SharedMemory:
    def __init__(self):
        self.redis = redis.Redis()
        self.qdrant = QdrantClient(host="localhost", port=6333)
        self.embeddings = OpenAIEmbeddings()
        self.collection_name = "research_memory"
    
    # === WORKING MEMORY (Redis) ===
    def store_session(self, session_id: str, state: dict):
        self.redis.hset(f"session:{session_id}", mapping={
            "state": json.dumps(state),
            "updated_at": datetime.now().isoformat()
        })
    
    def get_session(self, session_id: str) -> dict:
        data = self.redis.hgetall(f"session:{session_id}")
        return json.loads(data.get("state", "{}"))
    
    def append_finding(self, session_id: str, finding: dict):
        self.redis.rpush(f"findings:{session_id}", json.dumps(finding))
    
    # === LONG-TERM MEMORY (Qdrant) ===
    def store_research(self, query: str, report: str, metadata: dict):
        """Store completed research for future reference"""
        text = f"Query: {query}\n\nReport: {report}"
        embedding = self.embeddings.embed_query(text)
        
        self.qdrant.upsert(
            collection_name=self.collection_name,
            points=[{
                "id": str(uuid.uuid4()),
                "vector": embedding,
                "payload": {
                    "query": query,
                    "report": report,
                    "created_at": datetime.now().isoformat(),
                    **metadata
                }
            }]
        )
    
    def find_similar_research(self, query: str, limit: int = 3) -> list[dict]:
        """Find previous research on similar topics"""
        embedding = self.embeddings.embed_query(query)
        
        results = self.qdrant.search(
            collection_name=self.collection_name,
            query_vector=embedding,
            limit=limit
        )
        
        return [
            {
                "query": r.payload["query"],
                "report": r.payload["report"],
                "similarity": r.score
            }
            for r in results
        ]
```

### 6. Main Entry Point

```python
async def run_research(
    query: str,
    research_type: str = "general",
    max_iterations: int = 10
) -> str:
    """
    Main entry point for the multi-agent research system.
    
    Args:
        query: The research question or topic
        research_type: "market", "technical", "competitive", or "general"
        max_iterations: Maximum supervisor decision cycles
    
    Returns:
        Final research report
    """
    # Check for similar past research
    memory = SharedMemory()
    similar = memory.find_similar_research(query)
    
    # Initialize state
    initial_state: ResearchState = {
        "query": query,
        "research_type": research_type,
        "current_phase": "planning",
        "iteration": 0,
        "max_iterations": max_iterations,
        "research_findings": [],
        "analysis_results": [],
        "facts_learned": [],
        "sources_used": [],
        "report_draft": "",
        "final_report": "",
        "needs_more_research": False,
        "supervisor_notes": "",
        "error": None
    }
    
    # Add context from similar research
    if similar:
        initial_state["supervisor_notes"] = (
            f"Found {len(similar)} similar past research. "
            f"Most relevant: {similar[0]['query']}"
        )
    
    # Create and run graph
    graph = create_research_graph()
    config = {"configurable": {"thread_id": str(uuid.uuid4())}}
    
    # Stream execution
    async for event in graph.astream(initial_state, config):
        # Log progress
        if "supervisor" in event:
            print(f"Phase: {event['supervisor'].get('current_phase')}")
    
    # Get final state
    final_state = graph.get_state(config)
    
    # Store for future reference
    if final_state.values["final_report"]:
        memory.store_research(
            query=query,
            report=final_state.values["final_report"],
            metadata={
                "type": research_type,
                "iterations": final_state.values["iteration"],
                "sources": final_state.values["sources_used"]
            }
        )
    
    return final_state.values["final_report"]


# === USAGE ===
if __name__ == "__main__":
    import asyncio
    
    report = asyncio.run(run_research(
        query="Compare LangGraph, AutoGen, and CrewAI for building multi-agent systems",
        research_type="technical",
        max_iterations=15
    ))
    
    print(report)
```

---

## Deployment

### Docker Compose

```yaml
version: '3.8'

services:
  research-agent:
    build: .
    environment:
      - OPENAI_API_KEY=${OPENAI_API_KEY}
      - TAVILY_API_KEY=${TAVILY_API_KEY}
      - REDIS_URL=redis://redis:6379
      - QDRANT_URL=http://qdrant:6333
    depends_on:
      - redis
      - qdrant
    ports:
      - "8000:8000"

  redis:
    image: redis:7-alpine
    volumes:
      - redis_data:/data

  qdrant:
    image: qdrant/qdrant
    volumes:
      - qdrant_data:/qdrant/storage
    ports:
      - "6333:6333"

volumes:
  redis_data:
  qdrant_data:
```

### FastAPI Wrapper

```python
from fastapi import FastAPI, BackgroundTasks
from pydantic import BaseModel

app = FastAPI(title="Multi-Agent Research API")

class ResearchRequest(BaseModel):
    query: str
    research_type: str = "general"
    max_iterations: int = 10

class ResearchResponse(BaseModel):
    task_id: str
    status: str

@app.post("/research", response_model=ResearchResponse)
async def start_research(
    request: ResearchRequest,
    background_tasks: BackgroundTasks
):
    task_id = str(uuid.uuid4())
    
    background_tasks.add_task(
        run_research,
        request.query,
        request.research_type,
        request.max_iterations,
        task_id
    )
    
    return {"task_id": task_id, "status": "started"}

@app.get("/research/{task_id}")
async def get_research_status(task_id: str):
    memory = SharedMemory()
    session = memory.get_session(task_id)
    
    return {
        "status": session.get("current_phase", "unknown"),
        "iteration": session.get("iteration", 0),
        "report": session.get("final_report")
    }
```

---

## Observability

### LangSmith Integration

```python
import os
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_PROJECT"] = "multi-agent-research"

# All agent invocations are now traced
```

### Custom Metrics

```python
from prometheus_client import Counter, Histogram, start_http_server

research_requests = Counter('research_requests_total', 'Total research requests')
research_duration = Histogram('research_duration_seconds', 'Time spent on research')
agent_invocations = Counter('agent_invocations_total', 'Agent invocations', ['agent_type'])

async def run_research_with_metrics(query: str, **kwargs):
    research_requests.inc()
    
    with research_duration.time():
        result = await run_research(query, **kwargs)
    
    return result
```

---

## Lessons Learned

### What Works Well

1. **Supervisor pattern** provides clear control flow and debugging
2. **Shared memory** enables context persistence across agents
3. **LangGraph checkpoints** allow resume from failures
4. **Specialized agents** outperform single general-purpose agents

### Pitfalls to Avoid

1. **Too many agents** → coordination overhead exceeds benefits
2. **Unbounded iterations** → set max_iterations to prevent infinite loops
3. **Large context windows** → summarize intermediate results
4. **No error handling** → agents will fail; plan for graceful degradation

### Cost Optimization

- Use GPT-4o-mini for researcher (high volume, lower complexity)
- Use GPT-4o for supervisor and analyst (critical decisions)
- Cache embeddings and common queries
- Batch similar research tasks

---

## Related Resources

- [LLM Agents](../../3-ai-ml/llm-agents/) - Agent patterns and frameworks
- [RAG Systems](../../3-ai-ml/rag-systems/) - Document retrieval for agents
- [Event-Driven Systems](../../4-automation/event-driven/) - Async agent coordination
