# Autonomous Browser Agent

Blueprint for building AI-powered browser automation agents.

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                         Orchestrator                            │
│                    (LangGraph / AutoGen)                        │
└────────────────────────────┬────────────────────────────────────┘
                             │
          ┌──────────────────┼──────────────────┐
          │                  │                  │
          ▼                  ▼                  ▼
    ┌───────────┐     ┌───────────┐     ┌───────────┐
    │  Planner  │     │  Browser  │     │  Memory   │
    │   Agent   │     │   Agent   │     │   Store   │
    └───────────┘     └───────────┘     └───────────┘
          │                  │                  │
          │                  ▼                  │
          │          ┌───────────┐              │
          └─────────►│ Playwright│◄─────────────┘
                     └───────────┘
```

## Core Components

### State Management

```python
from typing import TypedDict, Annotated, List
from langgraph.graph import StateGraph
from operator import add

class BrowserAgentState(TypedDict):
    goal: str
    current_url: str
    page_content: str
    action_history: Annotated[List[dict], add]
    extracted_data: dict
    error: str | None
    completed: bool

def create_initial_state(goal: str) -> BrowserAgentState:
    return {
        "goal": goal,
        "current_url": "",
        "page_content": "",
        "action_history": [],
        "extracted_data": {},
        "error": None,
        "completed": False
    }
```

### Browser Tool Interface

```python
from playwright.async_api import async_playwright, Page
from pydantic import BaseModel
from typing import Literal

class BrowserAction(BaseModel):
    action: Literal["navigate", "click", "type", "scroll", "extract", "screenshot"]
    selector: str | None = None
    value: str | None = None
    url: str | None = None

class BrowserController:
    def __init__(self):
        self.browser = None
        self.page: Page = None
    
    async def initialize(self):
        p = await async_playwright().start()
        self.browser = await p.chromium.launch(headless=True)
        self.page = await self.browser.new_page()
    
    async def execute(self, action: BrowserAction) -> dict:
        match action.action:
            case "navigate":
                await self.page.goto(action.url)
                return {"status": "navigated", "url": action.url}
            
            case "click":
                await self.page.click(action.selector)
                return {"status": "clicked", "selector": action.selector}
            
            case "type":
                await self.page.fill(action.selector, action.value)
                return {"status": "typed", "selector": action.selector}
            
            case "scroll":
                await self.page.evaluate("window.scrollBy(0, 500)")
                return {"status": "scrolled"}
            
            case "extract":
                elements = await self.page.query_selector_all(action.selector)
                data = [await el.inner_text() for el in elements]
                return {"status": "extracted", "data": data}
            
            case "screenshot":
                path = f"screenshot_{datetime.now().timestamp()}.png"
                await self.page.screenshot(path=path)
                return {"status": "screenshot", "path": path}
    
    async def get_page_state(self) -> dict:
        return {
            "url": self.page.url,
            "title": await self.page.title(),
            "content": await self.page.content()
        }
    
    async def close(self):
        await self.browser.close()
```

### Planner Agent

```python
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate

PLANNER_PROMPT = """You are a browser automation planner. Given a goal and current page state, 
decide the next action to take.

Goal: {goal}
Current URL: {current_url}
Page Summary: {page_summary}
Previous Actions: {action_history}

Available actions:
- navigate(url): Go to a URL
- click(selector): Click an element
- type(selector, value): Type into an input
- scroll(): Scroll down the page
- extract(selector): Extract text from elements
- done(): Task is complete

Respond with the next action in JSON format:
{{"action": "...", "selector": "...", "value": "...", "url": "...", "reasoning": "..."}}
"""

class PlannerAgent:
    def __init__(self):
        self.llm = ChatOpenAI(model="gpt-4o", temperature=0)
        self.prompt = ChatPromptTemplate.from_template(PLANNER_PROMPT)
    
    async def plan_next_action(self, state: BrowserAgentState) -> BrowserAction:
        page_summary = self._summarize_page(state["page_content"])
        
        messages = self.prompt.format_messages(
            goal=state["goal"],
            current_url=state["current_url"],
            page_summary=page_summary,
            action_history=state["action_history"][-5:]  # Last 5 actions
        )
        
        response = await self.llm.ainvoke(messages)
        action_data = json.loads(response.content)
        
        return BrowserAction(**action_data)
    
    def _summarize_page(self, html: str) -> str:
        soup = BeautifulSoup(html, "lxml")
        # Extract key elements
        links = [a.text for a in soup.find_all("a")[:10]]
        buttons = [b.text for b in soup.find_all("button")[:10]]
        inputs = [i.get("name", i.get("id", "")) for i in soup.find_all("input")[:10]]
        
        return f"""
        Links: {links}
        Buttons: {buttons}
        Inputs: {inputs}
        """
```

### LangGraph Workflow

```python
from langgraph.graph import StateGraph, END

def create_browser_agent_graph():
    graph = StateGraph(BrowserAgentState)
    
    # Add nodes
    graph.add_node("plan", plan_action)
    graph.add_node("execute", execute_action)
    graph.add_node("observe", observe_result)
    graph.add_node("complete", mark_complete)
    
    # Add edges
    graph.add_edge("plan", "execute")
    graph.add_edge("execute", "observe")
    
    # Conditional routing
    graph.add_conditional_edges(
        "observe",
        should_continue,
        {
            "continue": "plan",
            "done": "complete",
            "error": END
        }
    )
    
    graph.add_edge("complete", END)
    graph.set_entry_point("plan")
    
    return graph.compile()

async def plan_action(state: BrowserAgentState) -> BrowserAgentState:
    planner = PlannerAgent()
    action = await planner.plan_next_action(state)
    
    if action.action == "done":
        return {**state, "completed": True}
    
    return {**state, "next_action": action}

async def execute_action(state: BrowserAgentState) -> BrowserAgentState:
    controller = get_browser_controller()
    result = await controller.execute(state["next_action"])
    
    return {
        **state,
        "action_history": [
            {"action": state["next_action"].dict(), "result": result}
        ]
    }

async def observe_result(state: BrowserAgentState) -> BrowserAgentState:
    controller = get_browser_controller()
    page_state = await controller.get_page_state()
    
    return {
        **state,
        "current_url": page_state["url"],
        "page_content": page_state["content"]
    }

def should_continue(state: BrowserAgentState) -> str:
    if state.get("completed"):
        return "done"
    if state.get("error"):
        return "error"
    if len(state["action_history"]) > 50:  # Safety limit
        return "error"
    return "continue"
```

## Use Case: E-commerce Price Tracker

```python
async def track_product_prices(product_urls: list[str]) -> list[dict]:
    agent = create_browser_agent_graph()
    results = []
    
    for url in product_urls:
        state = create_initial_state(
            goal=f"Navigate to {url}, extract the product name, price, and availability"
        )
        
        final_state = await agent.ainvoke(state)
        results.append(final_state["extracted_data"])
    
    return results

# Run
prices = await track_product_prices([
    "https://store.example.com/product/123",
    "https://store.example.com/product/456"
])
```

## Use Case: Form Automation

```python
async def fill_application_form(form_data: dict):
    goal = f"""
    1. Navigate to https://example.com/apply
    2. Fill the form with:
       - Name: {form_data['name']}
       - Email: {form_data['email']}
       - Message: {form_data['message']}
    3. Click submit
    4. Extract the confirmation message
    """
    
    agent = create_browser_agent_graph()
    state = create_initial_state(goal=goal)
    
    final_state = await agent.ainvoke(state)
    return final_state["extracted_data"]
```

## Error Handling

```python
class BrowserAgentError(Exception):
    pass

async def safe_execute(controller: BrowserController, action: BrowserAction) -> dict:
    max_retries = 3
    
    for attempt in range(max_retries):
        try:
            return await controller.execute(action)
        except TimeoutError:
            if attempt == max_retries - 1:
                raise BrowserAgentError(f"Timeout after {max_retries} attempts")
            await asyncio.sleep(1)
        except Exception as e:
            # Take screenshot for debugging
            await controller.page.screenshot(path=f"error_{datetime.now()}.png")
            raise BrowserAgentError(f"Action failed: {e}")
```

## Memory and Context

```python
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings

class AgentMemory:
    def __init__(self):
        self.embeddings = OpenAIEmbeddings()
        self.vectorstore = None
        self.action_history = []
    
    def add_page_visit(self, url: str, content: str, summary: str):
        if self.vectorstore is None:
            self.vectorstore = FAISS.from_texts(
                [summary],
                self.embeddings,
                metadatas=[{"url": url}]
            )
        else:
            self.vectorstore.add_texts(
                [summary],
                metadatas=[{"url": url}]
            )
    
    def recall_similar_pages(self, query: str, k: int = 3) -> list[dict]:
        if self.vectorstore is None:
            return []
        
        docs = self.vectorstore.similarity_search(query, k=k)
        return [{"content": d.page_content, **d.metadata} for d in docs]
```

## Related Resources

- [LLM Agents](../../3-ai-ml/llm-agents/README.md) - Agent framework deep dive
- [Browser Automation](../../4-automation/browser-automation/README.md) - Playwright patterns
- [Scraping Tools](../../2-data-engineering/etl-pipelines/scraping-tools/README.md) - Data extraction
