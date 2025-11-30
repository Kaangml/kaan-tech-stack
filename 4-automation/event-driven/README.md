# Event-Driven Systems

Asynchronous, decoupled architectures with message queues.

## Patterns

### Pub/Sub
```
Publisher → Topic → Subscriber 1
                 → Subscriber 2
                 → Subscriber N
```

### Message Queue
```
Producer → Queue → Consumer (one receives)
```

### Event Sourcing
```
Command → Event Store → Projections → Read Models
```

## Redis Pub/Sub

```python
import redis
import json

r = redis.Redis()

# Publisher
def publish_event(channel: str, event: dict):
    r.publish(channel, json.dumps(event))

# Subscriber
def subscribe(channel: str):
    pubsub = r.pubsub()
    pubsub.subscribe(channel)
    
    for message in pubsub.listen():
        if message["type"] == "message":
            event = json.loads(message["data"])
            process_event(event)
```

## Redis Streams (Better for persistence)

```python
# Add to stream
r.xadd("events", {"type": "order.created", "data": json.dumps(order)})

# Read from stream (consumer group)
r.xreadgroup(
    groupname="workers",
    consumername="worker-1",
    streams={"events": ">"},
    count=10
)
```

## FastAPI + Background Tasks

```python
from fastapi import FastAPI, BackgroundTasks

app = FastAPI()

async def send_notification(user_id: str, message: str):
    # Async notification logic
    pass

@app.post("/orders")
async def create_order(order: Order, background_tasks: BackgroundTasks):
    # Create order synchronously
    result = await db.create_order(order)
    
    # Queue notification asynchronously
    background_tasks.add_task(send_notification, order.user_id, "Order created")
    
    return result
```

## Celery (Distributed Tasks)

```python
from celery import Celery

app = Celery('tasks', broker='redis://localhost:6379')

@app.task
def process_document(doc_id: str):
    # Long-running task
    return {"status": "processed", "doc_id": doc_id}

# Call task
result = process_document.delay("doc_123")
```

## Related Resources
- [AWS Serverless](../../7-infrastructure/aws-serverless/) - SQS, Lambda triggers
- [Scalable Scraping](../../99-blueprints/scalable-scraping/) - Job queues
