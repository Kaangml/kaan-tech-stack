# Infrastructure

Docker, AWS, and deployment patterns.

## Topics

| Area | Description |
|------|-------------|
| [Docker](./docker/) | Containers, Compose, multi-stage builds |
| [AWS Serverless](./aws-serverless/) | Lambda, S3, Glue, API Gateway |

## Docker Quick Start

```dockerfile
FROM python:3.11-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .
CMD ["uvicorn", "main:app", "--host", "0.0.0.0"]
```

## AWS Services

| Service | Use Case |
|---------|----------|
| Lambda | Event-driven compute |
| S3 | Object storage |
| API Gateway | REST/HTTP APIs |
| Glue | ETL jobs |
| SQS | Message queues |

## Deployment Patterns

```
Local Dev → Docker Compose → CI/CD → AWS/Cloud
```

## Related Blueprints
- [Data Pipelines](../99-blueprints/data-pipelines/) - AWS Glue patterns
- [Scalable Scraping](../99-blueprints/scalable-scraping/) - Docker deployment
