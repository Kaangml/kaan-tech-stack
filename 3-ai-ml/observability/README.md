# LLM Observability & Evaluation

Monitoring, tracing, evaluation, and cost tracking for LLM applications in production.

## Why Observability Matters

```
Without Observability                With Observability
─────────────────────                ──────────────────
"The AI said something wrong"   →    "Token 847 in chain step 3 hallucinated"
"It's slow sometimes"           →    "P99 latency spike on embeddings at 3PM"
"Costs are high"                →    "GPT-4 calls for simple queries waste $200/day"
"Users don't like it"           →    "62% thumbs-down on legal QA queries"
```

## Table of Contents

- [Tracing Platforms](#tracing-platforms)
- [LangSmith Deep Dive](#langsmith-deep-dive)
- [Evaluation Frameworks](#evaluation-frameworks)
- [Cost Tracking](#cost-tracking)
- [Prompt Management](#prompt-management)
- [Alerting & Dashboards](#alerting--dashboards)

---

## Tracing Platforms

### Comparison

| Platform | Best For | Pricing | Key Features |
|----------|----------|---------|--------------|
| **LangSmith** | LangChain users | Free tier + paid | Native LC integration, datasets |
| **LangFuse** | Open-source option | Self-host free | EU hosting, prompt mgmt |
| **Phoenix (Arize)** | Local debugging | Free | In-notebook, no account |
| **Weights & Biases** | ML teams | Per seat | Full ML lifecycle |
| **Helicone** | Cost focus | Usage-based | Proxy-based, easy setup |

### Quick Setup Comparison

```python
# LangSmith
import os
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_API_KEY"] = "..."
# That's it - all LangChain calls are traced

# LangFuse
from langfuse.callback import CallbackHandler
handler = CallbackHandler()
chain.invoke({"input": "..."}, config={"callbacks": [handler]})

# Phoenix (local)
import phoenix as px
px.launch_app()
from phoenix.trace.langchain import LangChainInstrumentor
LangChainInstrumentor().instrument()

# Helicone (proxy)
from openai import OpenAI
client = OpenAI(
    base_url="https://oai.helicone.ai/v1",
    default_headers={"Helicone-Auth": f"Bearer {HELICONE_API_KEY}"}
)
```

---

## LangSmith Deep Dive

### Setup

```python
import os

# Environment variables
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_API_KEY"] = "ls__..."
os.environ["LANGCHAIN_PROJECT"] = "my-production-app"

# All LangChain operations are now traced automatically
```

### Tracing Custom Functions

```python
from langsmith import traceable
from langsmith.wrappers import wrap_openai

# Wrap OpenAI client
client = wrap_openai(OpenAI())

@traceable(name="process_document")
def process_document(doc: str) -> dict:
    # This function and all LLM calls inside are traced
    
    # Nested trace
    summary = summarize(doc)
    entities = extract_entities(doc)
    
    return {"summary": summary, "entities": entities}

@traceable(name="summarize", run_type="llm")
def summarize(text: str) -> str:
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": f"Summarize: {text}"}]
    )
    return response.choices[0].message.content

@traceable(name="extract_entities", run_type="chain")
def extract_entities(text: str) -> list:
    # Custom logic + LLM calls
    ...
```

### Logging Feedback

```python
from langsmith import Client

client = Client()

def log_user_feedback(run_id: str, score: float, comment: str = None):
    """Log user feedback for a specific run"""
    client.create_feedback(
        run_id=run_id,
        key="user-rating",
        score=score,  # 0-1
        comment=comment
    )

# In your app
@traceable
def answer_question(question: str) -> dict:
    answer = rag_chain.invoke(question)
    
    # Get run ID for feedback
    from langsmith.run_helpers import get_current_run_tree
    run_id = get_current_run_tree().id
    
    return {"answer": answer, "run_id": run_id}

# After user rates the answer
log_user_feedback(run_id="...", score=0.8, comment="Accurate but verbose")
```

### Creating Datasets

```python
from langsmith import Client

client = Client()

# Create dataset from production traces
dataset = client.create_dataset("qa-golden-set")

# Add examples manually
client.create_example(
    inputs={"question": "What is RAG?"},
    outputs={"answer": "Retrieval-Augmented Generation..."},
    dataset_id=dataset.id
)

# Add examples from runs (filtered)
runs = client.list_runs(
    project_name="production",
    filter='and(eq(feedback_key, "user-rating"), gte(feedback_score, 0.9))'
)

for run in runs:
    client.create_example(
        inputs=run.inputs,
        outputs=run.outputs,
        dataset_id=dataset.id
    )
```

### Running Evaluations

```python
from langsmith.evaluation import evaluate
from langsmith.schemas import Example, Run

# Define evaluators
def correctness(run: Run, example: Example) -> dict:
    """Check if answer matches expected"""
    predicted = run.outputs.get("answer", "")
    expected = example.outputs.get("answer", "")
    
    # Use LLM to judge
    prompt = f"""Compare these answers:
    Predicted: {predicted}
    Expected: {expected}
    
    Score 0-1 for correctness. Respond with just the number."""
    
    score = float(llm.invoke(prompt).content)
    return {"key": "correctness", "score": score}

def relevance(run: Run, example: Example) -> dict:
    """Check if answer is relevant to question"""
    question = example.inputs.get("question", "")
    answer = run.outputs.get("answer", "")
    
    prompt = f"""Is this answer relevant to the question?
    Question: {question}
    Answer: {answer}
    
    Score 0-1. Respond with just the number."""
    
    score = float(llm.invoke(prompt).content)
    return {"key": "relevance", "score": score}

# Run evaluation
results = evaluate(
    lambda inputs: my_chain.invoke(inputs),
    data="qa-golden-set",
    evaluators=[correctness, relevance],
    experiment_prefix="v2.1",
    max_concurrency=4
)

print(results.summary)
```

---

## Evaluation Frameworks

### RAGAS (RAG Assessment)

```python
from ragas import evaluate
from ragas.metrics import (
    faithfulness,        # Is answer grounded in context?
    answer_relevancy,    # Is answer relevant to question?
    context_precision,   # Are retrieved docs relevant?
    context_recall,      # Are all relevant docs retrieved?
    answer_correctness,  # Factual accuracy vs ground truth
)
from datasets import Dataset

# Prepare evaluation data
eval_data = {
    "question": ["What is RAG?", ...],
    "answer": ["RAG is...", ...],        # Generated answers
    "contexts": [["doc1", "doc2"], ...], # Retrieved contexts
    "ground_truth": ["RAG is...", ...]   # Expected answers
}

dataset = Dataset.from_dict(eval_data)

# Run evaluation
result = evaluate(
    dataset,
    metrics=[faithfulness, answer_relevancy, context_precision]
)

print(result)
# {'faithfulness': 0.87, 'answer_relevancy': 0.92, 'context_precision': 0.78}
```

### DeepEval

```python
from deepeval import evaluate
from deepeval.metrics import (
    AnswerRelevancyMetric,
    FaithfulnessMetric,
    ContextualPrecisionMetric,
    GEval
)
from deepeval.test_case import LLMTestCase

# Create test cases
test_case = LLMTestCase(
    input="What is RAG?",
    actual_output="RAG is Retrieval-Augmented Generation...",
    expected_output="RAG is a technique that...",
    retrieval_context=["Document 1 content...", "Document 2 content..."]
)

# Define metrics
metrics = [
    AnswerRelevancyMetric(threshold=0.7),
    FaithfulnessMetric(threshold=0.8),
    GEval(
        name="Conciseness",
        criteria="Is the answer concise without losing important information?",
        evaluation_params=[LLMTestCaseParams.ACTUAL_OUTPUT]
    )
]

# Run evaluation
result = evaluate([test_case], metrics)
```

### Custom Evaluation Pipeline

```python
from dataclasses import dataclass
from typing import Callable
import asyncio

@dataclass
class EvalResult:
    metric: str
    score: float
    explanation: str

class Evaluator:
    def __init__(self, llm):
        self.llm = llm
        self.metrics: dict[str, Callable] = {}
    
    def register_metric(self, name: str, prompt_template: str):
        async def evaluate(question: str, answer: str, context: str = None) -> EvalResult:
            prompt = prompt_template.format(
                question=question,
                answer=answer,
                context=context or ""
            )
            
            response = await self.llm.ainvoke(prompt)
            result = json.loads(response.content)
            
            return EvalResult(
                metric=name,
                score=result["score"],
                explanation=result["explanation"]
            )
        
        self.metrics[name] = evaluate
    
    async def evaluate_all(self, question: str, answer: str, context: str = None) -> list[EvalResult]:
        tasks = [
            metric(question, answer, context)
            for metric in self.metrics.values()
        ]
        return await asyncio.gather(*tasks)

# Usage
evaluator = Evaluator(llm)

evaluator.register_metric("accuracy", """
Evaluate the factual accuracy of this answer.
Question: {question}
Answer: {answer}
Context: {context}

Respond with JSON: {{"score": 0.0-1.0, "explanation": "..."}}
""")

evaluator.register_metric("helpfulness", """
How helpful is this answer to the user?
Question: {question}
Answer: {answer}

Respond with JSON: {{"score": 0.0-1.0, "explanation": "..."}}
""")

results = await evaluator.evaluate_all(
    question="What is RAG?",
    answer="RAG is...",
    context="Retrieved documents..."
)
```

---

## Cost Tracking

### Token Counting

```python
import tiktoken
from dataclasses import dataclass
from typing import Dict

@dataclass
class TokenUsage:
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    estimated_cost: float

class CostTracker:
    # Pricing per 1M tokens (as of 2024)
    PRICING = {
        "gpt-4o": {"input": 2.50, "output": 10.00},
        "gpt-4o-mini": {"input": 0.15, "output": 0.60},
        "gpt-4-turbo": {"input": 10.00, "output": 30.00},
        "claude-3-5-sonnet": {"input": 3.00, "output": 15.00},
        "claude-3-haiku": {"input": 0.25, "output": 1.25},
        "text-embedding-3-small": {"input": 0.02, "output": 0},
        "text-embedding-3-large": {"input": 0.13, "output": 0},
    }
    
    def __init__(self):
        self.usage_log: list[dict] = []
        self.encoders: Dict[str, tiktoken.Encoding] = {}
    
    def get_encoder(self, model: str) -> tiktoken.Encoding:
        if model not in self.encoders:
            try:
                self.encoders[model] = tiktoken.encoding_for_model(model)
            except KeyError:
                self.encoders[model] = tiktoken.get_encoding("cl100k_base")
        return self.encoders[model]
    
    def count_tokens(self, text: str, model: str = "gpt-4o") -> int:
        encoder = self.get_encoder(model)
        return len(encoder.encode(text))
    
    def calculate_cost(self, model: str, prompt_tokens: int, completion_tokens: int) -> float:
        if model not in self.PRICING:
            return 0.0
        
        pricing = self.PRICING[model]
        cost = (
            (prompt_tokens / 1_000_000) * pricing["input"] +
            (completion_tokens / 1_000_000) * pricing["output"]
        )
        return cost
    
    def log_usage(self, model: str, prompt_tokens: int, completion_tokens: int, metadata: dict = None):
        cost = self.calculate_cost(model, prompt_tokens, completion_tokens)
        
        entry = {
            "timestamp": datetime.now().isoformat(),
            "model": model,
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
            "cost": cost,
            **(metadata or {})
        }
        
        self.usage_log.append(entry)
        return entry
    
    def get_daily_summary(self) -> dict:
        today = datetime.now().date().isoformat()
        today_logs = [l for l in self.usage_log if l["timestamp"].startswith(today)]
        
        return {
            "date": today,
            "total_requests": len(today_logs),
            "total_tokens": sum(l["total_tokens"] for l in today_logs),
            "total_cost": sum(l["cost"] for l in today_logs),
            "by_model": self._group_by_model(today_logs)
        }
    
    def _group_by_model(self, logs: list) -> dict:
        result = {}
        for log in logs:
            model = log["model"]
            if model not in result:
                result[model] = {"requests": 0, "tokens": 0, "cost": 0}
            result[model]["requests"] += 1
            result[model]["tokens"] += log["total_tokens"]
            result[model]["cost"] += log["cost"]
        return result

# Integration with OpenAI
from openai import OpenAI

tracker = CostTracker()
client = OpenAI()

def tracked_completion(messages: list, model: str = "gpt-4o", **kwargs):
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        **kwargs
    )
    
    tracker.log_usage(
        model=model,
        prompt_tokens=response.usage.prompt_tokens,
        completion_tokens=response.usage.completion_tokens,
        metadata={"endpoint": "chat.completions"}
    )
    
    return response
```

### Budget Alerts

```python
import smtplib
from email.message import EmailMessage

class BudgetManager:
    def __init__(self, daily_budget: float, monthly_budget: float):
        self.daily_budget = daily_budget
        self.monthly_budget = monthly_budget
        self.tracker = CostTracker()
        self.alerts_sent = set()
    
    def check_budget(self):
        daily = self.tracker.get_daily_summary()
        monthly = self._get_monthly_total()
        
        alerts = []
        
        # Daily alerts at 80% and 100%
        if daily["total_cost"] >= self.daily_budget:
            alerts.append(("daily_exceeded", daily["total_cost"], self.daily_budget))
        elif daily["total_cost"] >= self.daily_budget * 0.8:
            alerts.append(("daily_warning", daily["total_cost"], self.daily_budget))
        
        # Monthly alerts
        if monthly >= self.monthly_budget:
            alerts.append(("monthly_exceeded", monthly, self.monthly_budget))
        elif monthly >= self.monthly_budget * 0.8:
            alerts.append(("monthly_warning", monthly, self.monthly_budget))
        
        for alert in alerts:
            self._send_alert(*alert)
    
    def _send_alert(self, alert_type: str, current: float, limit: float):
        if alert_type in self.alerts_sent:
            return
        
        message = f"""
        LLM Cost Alert: {alert_type}
        
        Current spend: ${current:.2f}
        Budget limit: ${limit:.2f}
        Usage: {(current/limit)*100:.1f}%
        """
        
        # Send via email, Slack, etc.
        self._notify(message)
        self.alerts_sent.add(alert_type)
```

---

## Prompt Management

### Version Control for Prompts

```python
from pydantic import BaseModel
from typing import Optional
import hashlib

class PromptVersion(BaseModel):
    name: str
    version: str
    template: str
    variables: list[str]
    created_at: str
    metadata: dict = {}
    
    @property
    def hash(self) -> str:
        return hashlib.sha256(self.template.encode()).hexdigest()[:8]

class PromptRegistry:
    def __init__(self, storage_backend):
        self.storage = storage_backend
        self.cache: dict[str, PromptVersion] = {}
    
    def register(self, name: str, template: str, metadata: dict = None) -> PromptVersion:
        # Extract variables
        import re
        variables = re.findall(r'\{(\w+)\}', template)
        
        # Get next version
        existing = self.get_all_versions(name)
        version = f"v{len(existing) + 1}"
        
        prompt = PromptVersion(
            name=name,
            version=version,
            template=template,
            variables=variables,
            created_at=datetime.now().isoformat(),
            metadata=metadata or {}
        )
        
        self.storage.save(prompt)
        return prompt
    
    def get(self, name: str, version: str = "latest") -> PromptVersion:
        cache_key = f"{name}:{version}"
        
        if cache_key not in self.cache:
            if version == "latest":
                versions = self.get_all_versions(name)
                prompt = versions[-1] if versions else None
            else:
                prompt = self.storage.get(name, version)
            
            if prompt:
                self.cache[cache_key] = prompt
        
        return self.cache.get(cache_key)
    
    def render(self, name: str, version: str = "latest", **kwargs) -> str:
        prompt = self.get(name, version)
        return prompt.template.format(**kwargs)

# Usage
registry = PromptRegistry(storage)

registry.register(
    name="summarizer",
    template="""Summarize the following text in {num_sentences} sentences.

Text: {text}

Summary:""",
    metadata={"use_case": "document_processing", "author": "kaan"}
)

# Get and use
prompt = registry.render("summarizer", text="...", num_sentences=3)
```

### A/B Testing Prompts

```python
import random
from collections import defaultdict

class PromptExperiment:
    def __init__(self, name: str, variants: dict[str, str], traffic_split: dict[str, float] = None):
        self.name = name
        self.variants = variants
        self.traffic_split = traffic_split or {k: 1/len(variants) for k in variants}
        self.results = defaultdict(lambda: {"count": 0, "scores": []})
    
    def get_variant(self, user_id: str = None) -> tuple[str, str]:
        """Get a variant, optionally deterministic per user"""
        if user_id:
            # Consistent assignment per user
            hash_val = int(hashlib.md5(f"{self.name}:{user_id}".encode()).hexdigest(), 16)
            rand = (hash_val % 1000) / 1000
        else:
            rand = random.random()
        
        cumulative = 0
        for variant_name, weight in self.traffic_split.items():
            cumulative += weight
            if rand < cumulative:
                return variant_name, self.variants[variant_name]
        
        # Fallback
        return list(self.variants.items())[0]
    
    def log_result(self, variant: str, score: float):
        self.results[variant]["count"] += 1
        self.results[variant]["scores"].append(score)
    
    def get_statistics(self) -> dict:
        stats = {}
        for variant, data in self.results.items():
            if data["scores"]:
                scores = data["scores"]
                stats[variant] = {
                    "count": data["count"],
                    "mean": sum(scores) / len(scores),
                    "min": min(scores),
                    "max": max(scores)
                }
        return stats

# Usage
experiment = PromptExperiment(
    name="qa-prompt-v2",
    variants={
        "control": "Answer this question: {question}",
        "detailed": "Provide a detailed answer with examples: {question}",
        "concise": "Answer in 2-3 sentences: {question}"
    },
    traffic_split={"control": 0.34, "detailed": 0.33, "concise": 0.33}
)

# In production
variant_name, prompt_template = experiment.get_variant(user_id="user123")
response = llm.invoke(prompt_template.format(question=user_question))

# After getting feedback
experiment.log_result(variant_name, user_rating)

# Analyze
print(experiment.get_statistics())
```

---

## Alerting & Dashboards

### Prometheus Metrics

```python
from prometheus_client import Counter, Histogram, Gauge, start_http_server

# Define metrics
llm_requests = Counter(
    'llm_requests_total',
    'Total LLM requests',
    ['model', 'endpoint', 'status']
)

llm_latency = Histogram(
    'llm_latency_seconds',
    'LLM request latency',
    ['model'],
    buckets=[0.1, 0.5, 1, 2, 5, 10, 30]
)

llm_tokens = Counter(
    'llm_tokens_total',
    'Total tokens used',
    ['model', 'type']  # type: input/output
)

llm_cost = Counter(
    'llm_cost_dollars',
    'Total cost in dollars',
    ['model']
)

active_requests = Gauge(
    'llm_active_requests',
    'Currently active LLM requests'
)

# Instrumented client
import time

class InstrumentedLLM:
    def __init__(self, client):
        self.client = client
    
    async def invoke(self, model: str, messages: list) -> dict:
        active_requests.inc()
        start = time.time()
        
        try:
            response = await self.client.chat.completions.create(
                model=model,
                messages=messages
            )
            
            # Record metrics
            llm_requests.labels(model=model, endpoint="chat", status="success").inc()
            llm_latency.labels(model=model).observe(time.time() - start)
            llm_tokens.labels(model=model, type="input").inc(response.usage.prompt_tokens)
            llm_tokens.labels(model=model, type="output").inc(response.usage.completion_tokens)
            
            cost = CostTracker.calculate_cost(model, response.usage.prompt_tokens, response.usage.completion_tokens)
            llm_cost.labels(model=model).inc(cost)
            
            return response
        
        except Exception as e:
            llm_requests.labels(model=model, endpoint="chat", status="error").inc()
            raise
        
        finally:
            active_requests.dec()

# Start metrics server
start_http_server(8000)
```

### Grafana Dashboard JSON

```json
{
  "title": "LLM Observability",
  "panels": [
    {
      "title": "Request Rate",
      "type": "graph",
      "targets": [
        {
          "expr": "rate(llm_requests_total[5m])",
          "legendFormat": "{{model}} - {{status}}"
        }
      ]
    },
    {
      "title": "Latency P50/P95/P99",
      "type": "graph",
      "targets": [
        {
          "expr": "histogram_quantile(0.50, rate(llm_latency_seconds_bucket[5m]))",
          "legendFormat": "P50"
        },
        {
          "expr": "histogram_quantile(0.95, rate(llm_latency_seconds_bucket[5m]))",
          "legendFormat": "P95"
        },
        {
          "expr": "histogram_quantile(0.99, rate(llm_latency_seconds_bucket[5m]))",
          "legendFormat": "P99"
        }
      ]
    },
    {
      "title": "Daily Cost",
      "type": "stat",
      "targets": [
        {
          "expr": "sum(increase(llm_cost_dollars[24h]))",
          "legendFormat": "Cost"
        }
      ]
    },
    {
      "title": "Tokens by Model",
      "type": "piechart",
      "targets": [
        {
          "expr": "sum by (model) (increase(llm_tokens_total[24h]))"
        }
      ]
    }
  ]
}
```

---

## Best Practices

### 1. Log Everything, Sample Wisely

```python
# In production, sample traces to reduce costs
import random

def should_trace() -> bool:
    # 10% sampling in production
    return random.random() < 0.1 or os.environ.get("FORCE_TRACE")

# Always trace errors
if error or should_trace():
    log_full_trace(run)
```

### 2. Structured Feedback Collection

```python
# Don't just collect thumbs up/down
feedback_schema = {
    "relevance": 0.0-1.0,    # Was it relevant to the question?
    "accuracy": 0.0-1.0,     # Was it factually correct?
    "helpfulness": 0.0-1.0,  # Did it help solve the problem?
    "comment": "string",     # Free-form feedback
    "issue_type": ["hallucination", "incomplete", "wrong_format", "other"]
}
```

### 3. Baseline Before Optimizing

```python
# Always establish baseline metrics before changes
baseline = {
    "latency_p95": 2.5,
    "accuracy": 0.85,
    "cost_per_query": 0.02,
    "user_satisfaction": 0.78
}

# After changes
new_metrics = run_evaluation()
improvement = {k: (new_metrics[k] - v) / v * 100 for k, v in baseline.items()}
```

---

## Related Resources

- [LLM Agents](../llm-agents/) - Observability for agent workflows
- [RAG Systems](../rag-systems/) - RAG-specific evaluation
- [AWS Serverless](../../7-infrastructure/aws-serverless/) - CloudWatch integration
