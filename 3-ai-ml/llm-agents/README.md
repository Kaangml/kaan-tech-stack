# LLM-Powered Autonomous Agents

Production architectures for building autonomous AI agents that reason, plan, and execute complex tasks.

## Table of Contents

- [Framework Selection](#framework-selection)
- [Agent Architectures](#agent-architectures)
- [Memory Systems](#memory-systems)
- [Tool Integration](#tool-integration)
- [Planning Strategies](#planning-strategies)
- [Multi-Agent Systems](#multi-agent-systems)
- [MCP (Model Context Protocol)](#mcp-model-context-protocol)
- [Error Recovery & Self-Correction](#error-recovery--self-correction)
- [Observability & Debugging](#observability--debugging)
- [Cost Optimization](#cost-optimization)

---

## Framework Selection

### Decision Matrix

| Framework | Architecture | Multi-Agent | Learning Curve | Best For |
|-----------|--------------|-------------|----------------|----------|
| **LangGraph** | Graph (Cyclic) | Structured | Medium | Complex flows, state machines |
| **AutoGen** | Conversation | Native | Medium | Team simulation, coding |
| **CrewAI** | Role-based | Native | Low | Task delegation, workflows |
| **LangChain** | Chain (DAG) | Basic | Low | RAG, simple agents |
| **Semantic Kernel** | Plugins | Basic | Medium | Enterprise, .NET integration |

### LangGraph

**Core Focus:** Stateful, cyclic workflows with precise control flow.

```python
from langgraph.graph import StateGraph, END
from typing import TypedDict, Annotated, Literal
from operator import add

class AgentState(TypedDict):
    messages: Annotated[list, add]
    current_step: str
    retry_count: int
    final_answer: str | None

def create_agent_graph():
    graph = StateGraph(AgentState)
    
    # Add nodes
    graph.add_node("analyze", analyze_task)
    graph.add_node("plan", create_plan)
    graph.add_node("execute", execute_action)
    graph.add_node("reflect", reflect_on_result)
    graph.add_node("respond", generate_response)
    
    # Entry point
    graph.set_entry_point("analyze")
    
    # Edges with conditional routing
    graph.add_edge("analyze", "plan")
    graph.add_edge("plan", "execute")
    graph.add_conditional_edges(
        "execute",
        should_retry_or_continue,
        {
            "retry": "plan",      # Loop back for retry
            "reflect": "reflect",
            "error": "respond"
        }
    )
    graph.add_conditional_edges(
        "reflect",
        is_task_complete,
        {
            "complete": "respond",
            "continue": "plan"    # Cyclic: go back to planning
        }
    )
    graph.add_edge("respond", END)
    
    return graph.compile()
```

**When to use:**
- ✅ Need loops and cycles (retry logic, iterative refinement)
- ✅ Complex state management across steps
- ✅ Human-in-the-loop workflows
- ✅ Long-running, multi-step tasks

### AutoGen

**Core Focus:** Multi-agent conversations with role-based collaboration.

```python
from autogen import AssistantAgent, UserProxyAgent, GroupChat, GroupChatManager

# Define specialized agents
researcher = AssistantAgent(
    name="Researcher",
    system_message="""You are a research specialist. 
    Search for information and provide detailed findings.
    Always cite your sources.""",
    llm_config=llm_config
)

critic = AssistantAgent(
    name="Critic",
    system_message="""You critically evaluate research findings.
    Point out gaps, biases, and areas needing more investigation.""",
    llm_config=llm_config
)

writer = AssistantAgent(
    name="Writer",
    system_message="""You synthesize research into clear reports.
    Structure information logically with executive summaries.""",
    llm_config=llm_config
)

# User proxy for human interaction
user_proxy = UserProxyAgent(
    name="User",
    human_input_mode="TERMINATE",
    code_execution_config={"work_dir": "workspace"}
)

# Create group chat
group_chat = GroupChat(
    agents=[user_proxy, researcher, critic, writer],
    messages=[],
    max_round=12
)

manager = GroupChatManager(groupchat=group_chat, llm_config=llm_config)

# Start conversation
user_proxy.initiate_chat(
    manager,
    message="Research the latest developments in quantum computing"
)
```

**When to use:**
- ✅ Simulating team dynamics
- ✅ Code generation with execution
- ✅ Research and analysis workflows
- ✅ Debate and critique patterns

### CrewAI

**Core Focus:** Role-based task delegation with clear hierarchies.

```python
from crewai import Agent, Task, Crew, Process

# Define agents with roles
researcher = Agent(
    role="Senior Research Analyst",
    goal="Uncover cutting-edge developments in AI",
    backstory="You're a veteran researcher with 20 years of experience",
    tools=[search_tool, scrape_tool],
    verbose=True
)

writer = Agent(
    role="Tech Content Strategist",
    goal="Craft compelling content about tech innovations",
    backstory="You're a renowned content strategist",
    tools=[],
    verbose=True
)

# Define tasks
research_task = Task(
    description="Research the latest AI agent frameworks in 2024",
    expected_output="Detailed report with comparisons",
    agent=researcher
)

write_task = Task(
    description="Write a blog post based on the research",
    expected_output="Engaging blog post, 1000 words",
    agent=writer,
    context=[research_task]  # Depends on research
)

# Create crew
crew = Crew(
    agents=[researcher, writer],
    tasks=[research_task, write_task],
    process=Process.sequential,
    verbose=2
)

result = crew.kickoff()
```

---

## Agent Architectures

### Single Agent Patterns

#### ReAct (Reasoning + Acting)

The foundational pattern: Think → Act → Observe → Repeat

```python
from langchain.agents import AgentExecutor, create_react_agent
from langchain_openai import ChatOpenAI
from langchain.tools import Tool

REACT_PROMPT = """Answer the following questions as best you can.

You have access to the following tools:
{tools}

Use the following format:

Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question

Begin!

Question: {input}
Thought:{agent_scratchpad}"""

llm = ChatOpenAI(model="gpt-4o", temperature=0)
tools = [search_tool, calculator_tool, code_executor]

agent = create_react_agent(llm, tools, REACT_PROMPT)
executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

result = executor.invoke({"input": "What is the population of Tokyo multiplied by 2?"})
```

#### Plan-and-Execute

Separate planning from execution for complex tasks:

```python
from langgraph.graph import StateGraph

class PlanExecuteState(TypedDict):
    objective: str
    plan: list[str]  # List of steps
    current_step: int
    results: list[str]
    final_output: str | None

async def planner(state: PlanExecuteState) -> dict:
    """Create a step-by-step plan"""
    response = await llm.ainvoke(
        f"""Create a detailed plan to achieve this objective:
        {state['objective']}
        
        Return a numbered list of specific, actionable steps."""
    )
    steps = parse_plan(response.content)
    return {"plan": steps, "current_step": 0}

async def executor(state: PlanExecuteState) -> dict:
    """Execute the current step"""
    current_step = state["plan"][state["current_step"]]
    
    result = await execute_step(current_step, state["results"])
    
    return {
        "results": [result],
        "current_step": state["current_step"] + 1
    }

async def replanner(state: PlanExecuteState) -> dict:
    """Adjust plan based on results"""
    if needs_replanning(state):
        new_plan = await replan(state)
        return {"plan": new_plan}
    return {}
```

#### ReWOO (Reasoning WithOut Observation)

Plan all actions upfront, then execute in batch:

```python
# Step 1: Plan all actions without execution
plan_prompt = """For the question: {question}
Create a plan with steps. For each step, specify:
- The tool to use
- The input to the tool
- What variable to store the result in

Format:
#E1 = Tool1[input1]
#E2 = Tool2[input using #E1]
..."""

# Step 2: Execute all planned actions
async def execute_plan(plan: list[dict]) -> dict:
    results = {}
    for step in plan:
        # Replace variable references with actual values
        input_resolved = resolve_variables(step["input"], results)
        result = await tools[step["tool"]].ainvoke(input_resolved)
        results[step["variable"]] = result
    return results

# Step 3: Synthesize final answer
def synthesize(question: str, results: dict) -> str:
    return llm.invoke(f"Question: {question}\nEvidence: {results}\nAnswer:")
```

---

## Memory Systems

### Memory Types

```
┌─────────────────────────────────────────────────────────────────┐
│                        AGENT MEMORY                             │
├─────────────────┬─────────────────┬─────────────────────────────┤
│   Short-Term    │   Long-Term     │      Episodic               │
│   (Working)     │   (Persistent)  │      (Experience)           │
├─────────────────┼─────────────────┼─────────────────────────────┤
│ • Current task  │ • User prefs    │ • Past task traces          │
│ • Recent msgs   │ • Facts learned │ • Success/failure logs      │
│ • Scratchpad    │ • Relationships │ • Similar problem solutions │
└─────────────────┴─────────────────┴─────────────────────────────┘
```

### Implementation

```python
from datetime import datetime
from typing import Optional
import json

class AgentMemory:
    def __init__(self, vector_store, redis_client):
        self.vector_store = vector_store
        self.redis = redis_client
        self.session_id = None
        
    # === SHORT-TERM MEMORY ===
    def init_session(self, session_id: str):
        self.session_id = session_id
        self.redis.delete(f"memory:{session_id}:buffer")
    
    def add_to_buffer(self, message: dict):
        """Add message to conversation buffer (last N messages)"""
        key = f"memory:{self.session_id}:buffer"
        self.redis.lpush(key, json.dumps(message))
        self.redis.ltrim(key, 0, 19)  # Keep last 20
    
    def get_buffer(self) -> list[dict]:
        """Get recent conversation context"""
        key = f"memory:{self.session_id}:buffer"
        messages = self.redis.lrange(key, 0, -1)
        return [json.loads(m) for m in messages]
    
    # === LONG-TERM MEMORY ===
    def store_fact(self, fact: str, metadata: dict = None):
        """Store a learned fact in vector store"""
        self.vector_store.add_texts(
            texts=[fact],
            metadatas=[{
                "type": "fact",
                "timestamp": datetime.now().isoformat(),
                **(metadata or {})
            }]
        )
    
    def recall_facts(self, query: str, k: int = 5) -> list[str]:
        """Retrieve relevant facts"""
        docs = self.vector_store.similarity_search(
            query, 
            k=k,
            filter={"type": "fact"}
        )
        return [doc.page_content for doc in docs]
    
    # === EPISODIC MEMORY ===
    def store_episode(self, task: str, steps: list, outcome: str, success: bool):
        """Store a complete task episode"""
        episode = {
            "task": task,
            "steps": steps,
            "outcome": outcome,
            "success": success,
            "timestamp": datetime.now().isoformat()
        }
        
        # Store as searchable vector
        episode_text = f"Task: {task}\nOutcome: {outcome}\nSuccess: {success}"
        self.vector_store.add_texts(
            texts=[episode_text],
            metadatas=[{"type": "episode", "full_episode": json.dumps(episode)}]
        )
    
    def recall_similar_episodes(self, task: str, k: int = 3) -> list[dict]:
        """Find similar past experiences"""
        docs = self.vector_store.similarity_search(
            task,
            k=k,
            filter={"type": "episode"}
        )
        return [json.loads(doc.metadata["full_episode"]) for doc in docs]
    
    # === ENTITY MEMORY ===
    def update_entity(self, entity_name: str, attributes: dict):
        """Update knowledge about an entity"""
        key = f"entity:{entity_name}"
        existing = self.redis.hgetall(key)
        existing.update(attributes)
        self.redis.hset(key, mapping=existing)
    
    def get_entity(self, entity_name: str) -> dict:
        return self.redis.hgetall(f"entity:{entity_name}")
```

### Memory-Augmented Agent

```python
class MemoryAugmentedAgent:
    def __init__(self, llm, memory: AgentMemory, tools: list):
        self.llm = llm
        self.memory = memory
        self.tools = tools
    
    async def run(self, task: str) -> str:
        # 1. Recall relevant context
        similar_episodes = self.memory.recall_similar_episodes(task)
        relevant_facts = self.memory.recall_facts(task)
        recent_context = self.memory.get_buffer()
        
        # 2. Build enriched prompt
        context = self._build_context(similar_episodes, relevant_facts, recent_context)
        
        # 3. Execute with context
        steps = []
        result = await self._execute_with_context(task, context, steps)
        
        # 4. Store episode for future reference
        self.memory.store_episode(
            task=task,
            steps=steps,
            outcome=result,
            success=self._evaluate_success(result)
        )
        
        return result
```

---

## Tool Integration

### Function Calling (OpenAI Style)

```python
from openai import OpenAI
from pydantic import BaseModel
import json

# Define tools as Pydantic models
class SearchWeb(BaseModel):
    """Search the web for information"""
    query: str
    num_results: int = 5

class ExecuteCode(BaseModel):
    """Execute Python code in a sandbox"""
    code: str
    timeout: int = 30

class SendEmail(BaseModel):
    """Send an email"""
    to: str
    subject: str
    body: str

# Convert to OpenAI tool format
tools = [
    {
        "type": "function",
        "function": {
            "name": "search_web",
            "description": "Search the web for information",
            "parameters": SearchWeb.model_json_schema()
        }
    },
    {
        "type": "function", 
        "function": {
            "name": "execute_code",
            "description": "Execute Python code in a sandbox",
            "parameters": ExecuteCode.model_json_schema()
        }
    }
]

# Tool executor
TOOL_EXECUTORS = {
    "search_web": lambda args: search_engine.search(**args),
    "execute_code": lambda args: sandbox.run(**args),
    "send_email": lambda args: email_client.send(**args)
}

async def run_with_tools(messages: list) -> str:
    client = OpenAI()
    
    while True:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=messages,
            tools=tools,
            tool_choice="auto"
        )
        
        message = response.choices[0].message
        
        if message.tool_calls:
            messages.append(message)
            
            for tool_call in message.tool_calls:
                func_name = tool_call.function.name
                args = json.loads(tool_call.function.arguments)
                
                result = TOOL_EXECUTORS[func_name](args)
                
                messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": str(result)
                })
        else:
            return message.content
```

### Dynamic Tool Loading

```python
from typing import Callable, Any
import inspect

class ToolRegistry:
    def __init__(self):
        self._tools: dict[str, Callable] = {}
        self._schemas: dict[str, dict] = {}
    
    def register(self, name: str = None, description: str = None):
        """Decorator to register a function as a tool"""
        def decorator(func: Callable):
            tool_name = name or func.__name__
            tool_desc = description or func.__doc__ or ""
            
            # Auto-generate schema from type hints
            sig = inspect.signature(func)
            params = {}
            required = []
            
            for param_name, param in sig.parameters.items():
                if param.annotation != inspect.Parameter.empty:
                    params[param_name] = self._type_to_schema(param.annotation)
                    if param.default == inspect.Parameter.empty:
                        required.append(param_name)
            
            self._tools[tool_name] = func
            self._schemas[tool_name] = {
                "type": "function",
                "function": {
                    "name": tool_name,
                    "description": tool_desc,
                    "parameters": {
                        "type": "object",
                        "properties": params,
                        "required": required
                    }
                }
            }
            return func
        return decorator
    
    def get_tools_for_context(self, context: str) -> list[dict]:
        """Dynamically select relevant tools based on context"""
        # Use embeddings to find relevant tools
        relevant = self._find_relevant_tools(context)
        return [self._schemas[name] for name in relevant]

# Usage
registry = ToolRegistry()

@registry.register(description="Search for flights between cities")
def search_flights(origin: str, destination: str, date: str) -> list[dict]:
    ...

@registry.register(description="Book a hotel room")  
def book_hotel(city: str, check_in: str, check_out: str, guests: int = 1) -> dict:
    ...
```

---

## Planning Strategies

### Tree of Thoughts (ToT)

Explore multiple reasoning paths:

```python
from dataclasses import dataclass
from typing import List

@dataclass
class Thought:
    content: str
    score: float
    children: List["Thought"] = None

class TreeOfThoughts:
    def __init__(self, llm, branching_factor: int = 3, max_depth: int = 3):
        self.llm = llm
        self.branching_factor = branching_factor
        self.max_depth = max_depth
    
    async def solve(self, problem: str) -> str:
        root = await self._generate_thoughts(problem, None)
        best_path = await self._search(root)
        return self._extract_solution(best_path)
    
    async def _generate_thoughts(self, problem: str, parent: Thought | None) -> list[Thought]:
        context = parent.content if parent else ""
        
        prompt = f"""Problem: {problem}
        Current thinking: {context}
        
        Generate {self.branching_factor} different next steps in reasoning.
        For each, provide a score (0-1) for how promising it is."""
        
        response = await self.llm.ainvoke(prompt)
        thoughts = self._parse_thoughts(response.content)
        
        return thoughts
    
    async def _search(self, thoughts: list[Thought], depth: int = 0) -> list[Thought]:
        if depth >= self.max_depth:
            return [max(thoughts, key=lambda t: t.score)]
        
        # BFS with pruning - keep top k
        sorted_thoughts = sorted(thoughts, key=lambda t: t.score, reverse=True)
        top_thoughts = sorted_thoughts[:self.branching_factor]
        
        # Expand top thoughts
        all_children = []
        for thought in top_thoughts:
            children = await self._generate_thoughts(thought.content, thought)
            thought.children = children
            all_children.extend(children)
        
        return await self._search(all_children, depth + 1)
```

### Hierarchical Task Decomposition

```python
class HierarchicalPlanner:
    def __init__(self, llm):
        self.llm = llm
    
    async def decompose(self, goal: str, max_depth: int = 3) -> dict:
        """Recursively decompose a goal into sub-tasks"""
        return await self._decompose_recursive(goal, depth=0, max_depth=max_depth)
    
    async def _decompose_recursive(self, task: str, depth: int, max_depth: int) -> dict:
        if depth >= max_depth:
            return {"task": task, "subtasks": [], "is_atomic": True}
        
        # Check if task is atomic (can be done in one step)
        is_atomic = await self._is_atomic(task)
        
        if is_atomic:
            return {"task": task, "subtasks": [], "is_atomic": True}
        
        # Decompose into subtasks
        subtasks = await self._get_subtasks(task)
        
        # Recursively decompose subtasks
        decomposed_subtasks = []
        for subtask in subtasks:
            decomposed = await self._decompose_recursive(subtask, depth + 1, max_depth)
            decomposed_subtasks.append(decomposed)
        
        return {
            "task": task,
            "subtasks": decomposed_subtasks,
            "is_atomic": False
        }
    
    async def _is_atomic(self, task: str) -> bool:
        response = await self.llm.ainvoke(
            f"Can this task be completed in a single action? Task: {task}\nAnswer YES or NO."
        )
        return "YES" in response.content.upper()
    
    async def _get_subtasks(self, task: str) -> list[str]:
        response = await self.llm.ainvoke(
            f"Break down this task into 3-5 subtasks:\n{task}\n\nSubtasks:"
        )
        return self._parse_subtasks(response.content)
```

---

## Multi-Agent Systems

### Supervisor Architecture

```python
from langgraph.graph import StateGraph, END
from typing import Literal

class SupervisorState(TypedDict):
    messages: list
    next_agent: str
    task_complete: bool

# Define worker agents
workers = {
    "researcher": create_researcher_agent(),
    "coder": create_coder_agent(),
    "writer": create_writer_agent()
}

# Supervisor decides which agent to call
async def supervisor(state: SupervisorState) -> dict:
    response = await supervisor_llm.ainvoke(
        f"""You are a supervisor managing a team of agents.
        
        Available agents:
        - researcher: Searches and gathers information
        - coder: Writes and executes code
        - writer: Creates documentation and reports
        
        Based on the conversation, which agent should act next?
        Or is the task complete?
        
        Conversation: {state['messages']}
        
        Respond with: researcher, coder, writer, or FINISH"""
    )
    
    next_agent = response.content.strip().lower()
    
    return {
        "next_agent": next_agent,
        "task_complete": next_agent == "finish"
    }

# Build graph
def create_supervisor_graph():
    graph = StateGraph(SupervisorState)
    
    graph.add_node("supervisor", supervisor)
    for name, agent in workers.items():
        graph.add_node(name, agent)
    
    # Supervisor routes to workers
    graph.add_conditional_edges(
        "supervisor",
        lambda s: s["next_agent"] if not s["task_complete"] else "end",
        {**{name: name for name in workers}, "end": END}
    )
    
    # Workers return to supervisor
    for name in workers:
        graph.add_edge(name, "supervisor")
    
    graph.set_entry_point("supervisor")
    
    return graph.compile()
```

### Debate Architecture

```python
class DebateSystem:
    def __init__(self, llm):
        self.proposer = self._create_agent("proposer", 
            "You propose solutions and defend your ideas.")
        self.critic = self._create_agent("critic",
            "You find flaws and challenge proposals.")
        self.judge = self._create_agent("judge",
            "You evaluate arguments and make final decisions.")
    
    async def debate(self, problem: str, rounds: int = 3) -> str:
        proposal = await self.proposer.propose(problem)
        
        for round in range(rounds):
            # Critic challenges
            critique = await self.critic.critique(proposal)
            
            # Proposer defends/revises
            proposal = await self.proposer.revise(proposal, critique)
        
        # Judge makes final decision
        verdict = await self.judge.decide(problem, proposal)
        
        return verdict
```

---

## MCP (Model Context Protocol)

### What is MCP?

MCP is Anthropic's open protocol for connecting AI models to external tools, data sources, and services.

```
┌─────────────┐     MCP Protocol     ┌─────────────┐
│   Claude    │◄───────────────────►│  MCP Server │
│   (Client)  │    JSON-RPC 2.0     │   (Tools)   │
└─────────────┘                      └─────────────┘
                                            │
                    ┌───────────────────────┼───────────────────────┐
                    │                       │                       │
              ┌─────▼─────┐          ┌─────▼─────┐          ┌──────▼─────┐
              │ Database  │          │   APIs    │          │   Files    │
              └───────────┘          └───────────┘          └────────────┘
```

### Building an MCP Server

```python
# mcp_server.py
from mcp.server import Server
from mcp.types import Tool, TextContent
import mcp.server.stdio

# Create server
server = Server("my-tools")

# Define tools
@server.tool()
async def search_database(query: str) -> str:
    """Search the company database for information"""
    results = await db.search(query)
    return f"Found {len(results)} results: {results}"

@server.tool()
async def create_ticket(title: str, description: str, priority: str = "medium") -> str:
    """Create a support ticket in the ticketing system"""
    ticket_id = await ticketing_system.create(
        title=title,
        description=description,
        priority=priority
    )
    return f"Created ticket #{ticket_id}"

@server.tool()
async def get_weather(city: str) -> str:
    """Get current weather for a city"""
    weather = await weather_api.get(city)
    return f"{city}: {weather['temp']}°C, {weather['condition']}"

# Run server
async def main():
    async with mcp.server.stdio.stdio_server() as (read_stream, write_stream):
        await server.run(
            read_stream,
            write_stream,
            server.create_initialization_options()
        )

if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
```

### MCP Server Configuration

```json
// claude_desktop_config.json
{
  "mcpServers": {
    "my-tools": {
      "command": "python",
      "args": ["/path/to/mcp_server.py"],
      "env": {
        "DATABASE_URL": "postgresql://...",
        "API_KEY": "..."
      }
    },
    "filesystem": {
      "command": "npx",
      "args": ["-y", "@modelcontextprotocol/server-filesystem", "/allowed/path"]
    }
  }
}
```

### Connecting MCP to LangGraph

```python
from mcp import ClientSession
from mcp.client.stdio import stdio_client

class MCPToolProvider:
    def __init__(self, server_path: str):
        self.server_path = server_path
        self.session: ClientSession = None
    
    async def connect(self):
        transport = await stdio_client(self.server_path)
        self.session = ClientSession(*transport)
        await self.session.initialize()
    
    async def get_tools(self) -> list:
        """Get available tools as LangChain-compatible format"""
        tools = await self.session.list_tools()
        return [self._convert_to_langchain(t) for t in tools]
    
    async def call_tool(self, name: str, arguments: dict) -> str:
        result = await self.session.call_tool(name, arguments)
        return result.content[0].text

# Usage in LangGraph
mcp_provider = MCPToolProvider("python mcp_server.py")
await mcp_provider.connect()

# Add MCP tools to your agent
tools = await mcp_provider.get_tools()
agent = create_agent(llm, tools)
```

---

## Error Recovery & Self-Correction

### Retry with Reflection

```python
class SelfCorrectingAgent:
    def __init__(self, llm, max_retries: int = 3):
        self.llm = llm
        self.max_retries = max_retries
    
    async def execute_with_retry(self, task: str) -> str:
        errors = []
        
        for attempt in range(self.max_retries):
            try:
                # Attempt execution
                result = await self._execute(task, errors)
                
                # Validate result
                validation = await self._validate(task, result)
                
                if validation["is_valid"]:
                    return result
                else:
                    errors.append({
                        "attempt": attempt + 1,
                        "result": result,
                        "issue": validation["issue"]
                    })
            
            except Exception as e:
                errors.append({
                    "attempt": attempt + 1,
                    "error": str(e),
                    "type": type(e).__name__
                })
        
        # Final attempt with all error context
        return await self._final_attempt(task, errors)
    
    async def _execute(self, task: str, previous_errors: list) -> str:
        error_context = ""
        if previous_errors:
            error_context = f"""
            Previous attempts failed:
            {json.dumps(previous_errors, indent=2)}
            
            Learn from these errors and try a different approach.
            """
        
        response = await self.llm.ainvoke(
            f"Task: {task}\n{error_context}\nExecute the task:"
        )
        return response.content
    
    async def _validate(self, task: str, result: str) -> dict:
        response = await self.llm.ainvoke(
            f"""Task: {task}
            Result: {result}
            
            Is this result correct and complete?
            Respond with JSON: {{"is_valid": true/false, "issue": "description if invalid"}}"""
        )
        return json.loads(response.content)
```

### Graceful Degradation

```python
class ResilientAgent:
    def __init__(self, primary_llm, fallback_llm, tools: list):
        self.primary = primary_llm
        self.fallback = fallback_llm
        self.tools = tools
    
    async def run(self, task: str) -> str:
        try:
            # Try primary (e.g., GPT-4o)
            return await self._run_with_llm(self.primary, task, full_tools=True)
        
        except RateLimitError:
            # Fallback to secondary (e.g., GPT-3.5)
            return await self._run_with_llm(self.fallback, task, full_tools=False)
        
        except ToolExecutionError as e:
            # Run without the failing tool
            safe_tools = [t for t in self.tools if t.name != e.tool_name]
            return await self._run_with_llm(self.primary, task, tools=safe_tools)
        
        except Exception as e:
            # Last resort: simple completion without tools
            return await self.fallback.ainvoke(
                f"Answer this as best you can without tools: {task}"
            )
```

---

## Observability & Debugging

### Structured Logging

```python
import structlog
from opentelemetry import trace

logger = structlog.get_logger()
tracer = trace.get_tracer(__name__)

class ObservableAgent:
    async def run(self, task: str) -> str:
        with tracer.start_as_current_span("agent_run") as span:
            span.set_attribute("task", task)
            
            run_id = str(uuid.uuid4())
            logger.info("agent_started", run_id=run_id, task=task)
            
            try:
                # Track each step
                async for step in self._execute_steps(task):
                    logger.info("step_completed",
                        run_id=run_id,
                        step=step["name"],
                        duration_ms=step["duration"],
                        tokens_used=step["tokens"]
                    )
                    span.add_event(step["name"], {"tokens": step["tokens"]})
                
                result = self._get_final_result()
                
                logger.info("agent_completed",
                    run_id=run_id,
                    success=True,
                    total_steps=len(self.steps),
                    total_tokens=self.total_tokens
                )
                
                return result
                
            except Exception as e:
                logger.error("agent_failed",
                    run_id=run_id,
                    error=str(e),
                    step=self.current_step
                )
                span.record_exception(e)
                raise
```

### Integration with LangSmith

```python
from langsmith import traceable
from langsmith.wrappers import wrap_openai

# Wrap OpenAI client for automatic tracing
client = wrap_openai(OpenAI())

@traceable(name="research_agent")
async def research(query: str) -> str:
    # All LLM calls inside are automatically traced
    plan = await create_plan(query)
    
    results = []
    for step in plan:
        result = await execute_step(step)
        results.append(result)
    
    return synthesize(results)

# Custom evaluation
@traceable(name="evaluate_response")
def evaluate_response(response: str, expected: str) -> dict:
    return {
        "accuracy": calculate_accuracy(response, expected),
        "completeness": calculate_completeness(response, expected),
        "relevance": calculate_relevance(response, expected)
    }
```

---

## Cost Optimization

### Token Management

```python
import tiktoken

class TokenAwareAgent:
    def __init__(self, llm, max_tokens_per_run: int = 10000):
        self.llm = llm
        self.max_tokens = max_tokens_per_run
        self.encoder = tiktoken.encoding_for_model("gpt-4o")
        self.tokens_used = 0
    
    def count_tokens(self, text: str) -> int:
        return len(self.encoder.encode(text))
    
    async def run(self, task: str) -> str:
        while self.tokens_used < self.max_tokens:
            # Check remaining budget
            remaining = self.max_tokens - self.tokens_used
            
            if remaining < 500:
                # Force conclusion
                return await self._force_conclude()
            
            # Run with budget awareness
            result = await self._step(remaining)
            
            if result["done"]:
                return result["answer"]
        
        raise TokenBudgetExceeded(f"Used {self.tokens_used} tokens")
```

### Model Routing

```python
class ModelRouter:
    def __init__(self):
        self.models = {
            "simple": "gpt-3.5-turbo",      # $0.002/1K
            "standard": "gpt-4o-mini",       # $0.01/1K
            "complex": "gpt-4o",             # $0.03/1K
            "reasoning": "o1-preview"        # $0.06/1K
        }
    
    async def route(self, task: str) -> str:
        # Classify task complexity
        complexity = await self._classify_complexity(task)
        
        model = self.models[complexity]
        
        return await self._call_model(model, task)
    
    async def _classify_complexity(self, task: str) -> str:
        # Quick classification with cheap model
        response = await cheap_llm.ainvoke(
            f"""Classify this task's complexity:
            - simple: factual questions, formatting
            - standard: analysis, summarization
            - complex: multi-step reasoning, code generation
            - reasoning: math, logic puzzles
            
            Task: {task}
            
            Respond with one word: simple/standard/complex/reasoning"""
        )
        return response.content.strip().lower()
```

### Caching

```python
from functools import lru_cache
import hashlib

class CachedAgent:
    def __init__(self, llm, redis_client):
        self.llm = llm
        self.redis = redis_client
        self.cache_ttl = 3600  # 1 hour
    
    async def run(self, task: str, use_cache: bool = True) -> str:
        cache_key = self._make_key(task)
        
        if use_cache:
            cached = self.redis.get(cache_key)
            if cached:
                return cached.decode()
        
        result = await self._execute(task)
        
        # Cache successful results
        self.redis.setex(cache_key, self.cache_ttl, result)
        
        return result
    
    def _make_key(self, task: str) -> str:
        # Hash task for cache key
        task_hash = hashlib.sha256(task.encode()).hexdigest()[:16]
        return f"agent:cache:{task_hash}"
```

---

## Related Resources

- [Autonomous Browser Agent Blueprint](../../99-blueprints/autonomous-browser-agent/) - Full implementation
- [RAG Systems](../rag-systems/) - Retrieval for agent context
- [Multi-Agent Research Blueprint](../../99-blueprints/multi-agent-research/) - Team-based agents
