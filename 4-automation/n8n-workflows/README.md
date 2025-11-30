# n8n Workflows

Low-code workflow automation platform.

## Core Concepts

### Workflow Structure
```
Trigger → Node 1 → Node 2 → ... → Output
              ↓
         Branching → Alternative path
```

### Common Triggers
- Webhook (HTTP requests)
- Cron (scheduled)
- Database changes
- Email received
- File system events

## Use Cases

| Scenario | Workflow |
|----------|----------|
| Lead processing | Webhook → Enrich → CRM → Notify |
| Content sync | Schedule → Fetch → Transform → Push |
| Alert pipeline | Monitor → Filter → Slack/Email |
| Data backup | Cron → Export → S3 upload |

## Integration with Python

```python
# Call n8n webhook from Python
import httpx

async def trigger_workflow(data: dict):
    async with httpx.AsyncClient() as client:
        response = await client.post(
            "https://n8n.example.com/webhook/my-workflow",
            json=data
        )
        return response.json()
```

## n8n + FastAPI

```python
from fastapi import FastAPI, Request

app = FastAPI()

@app.post("/n8n-callback")
async def n8n_callback(request: Request):
    """Receive processed data from n8n workflow."""
    data = await request.json()
    # Process result
    return {"status": "received"}
```

## Self-Hosting

```yaml
# docker-compose.yml
version: '3.8'
services:
  n8n:
    image: n8nio/n8n
    ports:
      - "5678:5678"
    environment:
      - N8N_BASIC_AUTH_ACTIVE=true
      - N8N_BASIC_AUTH_USER=admin
      - N8N_BASIC_AUTH_PASSWORD=password
    volumes:
      - n8n_data:/home/node/.n8n

volumes:
  n8n_data:
```

## Related Resources
- [Event-Driven](../event-driven/) - Message queue patterns
- [AWS Serverless](../../7-infrastructure/aws-serverless/) - Lambda triggers
