# Docker & Containerization

Container basics, multi-stage builds, and orchestration.

## Dockerfile Best Practices

### Multi-Stage Builds

```dockerfile
# Stage 1: Build
FROM python:3.11-slim as builder

WORKDIR /app

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip wheel --no-cache-dir --no-deps --wheel-dir /app/wheels -r requirements.txt

# Stage 2: Production
FROM python:3.11-slim as production

WORKDIR /app

# Create non-root user
RUN groupadd -r appgroup && useradd -r -g appgroup appuser

# Copy wheels from builder
COPY --from=builder /app/wheels /wheels
COPY --from=builder /app/requirements.txt .

# Install dependencies
RUN pip install --no-cache /wheels/*

# Copy application
COPY --chown=appuser:appgroup . .

# Switch to non-root user
USER appuser

# Run
EXPOSE 8000
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
```

### Layer Optimization

```dockerfile
# Bad: Invalidates cache on code change
COPY . .
RUN pip install -r requirements.txt

# Good: Dependencies cached separately
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
```

### Security Practices

```dockerfile
# Use specific version tags
FROM python:3.11.7-slim-bookworm

# Don't run as root
RUN useradd -m -s /bin/bash appuser
USER appuser

# Remove unnecessary packages
RUN apt-get purge -y --auto-remove -o APT::AutoRemove::RecommendsImportant=false

# Read-only filesystem (use with --read-only flag)
# Scan for vulnerabilities
# docker scout cves myimage:latest
```

## Docker Compose

### Development Setup

```yaml
# docker-compose.yml
version: '3.8'

services:
  api:
    build:
      context: .
      dockerfile: Dockerfile
      target: development  # Use dev stage
    volumes:
      - .:/app
      - /app/.venv  # Exclude venv from mount
    ports:
      - "8000:8000"
    environment:
      - DATABASE_URL=postgresql://postgres:password@db:5432/mydb
      - REDIS_URL=redis://redis:6379
    depends_on:
      db:
        condition: service_healthy
      redis:
        condition: service_started
    command: uvicorn main:app --host 0.0.0.0 --reload

  db:
    image: postgres:15-alpine
    volumes:
      - postgres_data:/var/lib/postgresql/data
    environment:
      - POSTGRES_USER=postgres
      - POSTGRES_PASSWORD=password
      - POSTGRES_DB=mydb
    healthcheck:
      test: ["CMD-SHELL", "pg_isready -U postgres"]
      interval: 5s
      timeout: 5s
      retries: 5

  redis:
    image: redis:7-alpine
    volumes:
      - redis_data:/data

  worker:
    build: .
    environment:
      - DATABASE_URL=postgresql://postgres:password@db:5432/mydb
      - REDIS_URL=redis://redis:6379
    depends_on:
      - db
      - redis
    command: celery -A tasks worker --loglevel=info

volumes:
  postgres_data:
  redis_data:
```

### Production Compose

```yaml
# docker-compose.prod.yml
version: '3.8'

services:
  api:
    image: myregistry/myapp:${VERSION:-latest}
    deploy:
      replicas: 3
      resources:
        limits:
          cpus: '0.5'
          memory: 512M
        reservations:
          cpus: '0.25'
          memory: 256M
      restart_policy:
        condition: on-failure
        delay: 5s
        max_attempts: 3
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/health"]
      interval: 30s
      timeout: 10s
      retries: 3
    logging:
      driver: "json-file"
      options:
        max-size: "10m"
        max-file: "3"
```

## Useful Commands

### Build & Run

```bash
# Build
docker build -t myapp:latest .
docker build -t myapp:latest --target production .
docker build --no-cache -t myapp:latest .

# Run
docker run -d --name myapp -p 8000:8000 myapp:latest
docker run -it --rm myapp:latest /bin/bash  # Interactive
docker run -d -v $(pwd):/app myapp:latest   # Volume mount

# Compose
docker compose up -d
docker compose up --build  # Rebuild
docker compose down -v     # Remove volumes
docker compose logs -f api # Follow logs
```

### Debugging

```bash
# Enter running container
docker exec -it myapp /bin/bash

# View logs
docker logs myapp
docker logs --tail 100 -f myapp

# Inspect
docker inspect myapp
docker stats  # Resource usage

# Copy files
docker cp myapp:/app/logs ./logs
docker cp ./config.json myapp:/app/
```

### Cleanup

```bash
# Remove stopped containers
docker container prune

# Remove unused images
docker image prune -a

# Remove all unused resources
docker system prune -a --volumes

# Remove specific
docker rm container_name
docker rmi image_name
```

## Networking

### Network Types

```bash
# Bridge (default)
docker network create mynetwork
docker run --network mynetwork myapp

# Host (Linux only, no port mapping needed)
docker run --network host myapp

# None (isolated)
docker run --network none myapp
```

### Service Discovery

```yaml
# docker-compose.yml
services:
  api:
    networks:
      - frontend
      - backend
  
  db:
    networks:
      - backend

networks:
  frontend:
  backend:
```

```python
# Services can reach each other by name
# From api container:
import psycopg2
conn = psycopg2.connect(host="db", ...)  # "db" resolves to db container
```

## Python-Specific Patterns

### Poetry with Docker

```dockerfile
FROM python:3.11-slim as builder

ENV POETRY_HOME="/opt/poetry" \
    POETRY_VIRTUALENVS_IN_PROJECT=true \
    POETRY_NO_INTERACTION=1

RUN pip install poetry

WORKDIR /app
COPY pyproject.toml poetry.lock ./
RUN poetry install --only=main --no-root

FROM python:3.11-slim

WORKDIR /app
COPY --from=builder /app/.venv /app/.venv
ENV PATH="/app/.venv/bin:$PATH"

COPY . .
CMD ["python", "main.py"]
```

### FastAPI with Gunicorn

```dockerfile
FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Gunicorn with Uvicorn workers
CMD ["gunicorn", "main:app", \
     "-w", "4", \
     "-k", "uvicorn.workers.UvicornWorker", \
     "-b", "0.0.0.0:8000", \
     "--access-logfile", "-"]
```

### ML Model Serving

```dockerfile
FROM python:3.11-slim

# Install system dependencies for ML
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy model files
COPY models/ /app/models/
COPY src/ /app/src/

# Preload model on startup
ENV MODEL_PATH=/app/models/model.pkl

CMD ["python", "-m", "src.serve"]
```

## Health Checks

```dockerfile
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1
```

```python
# FastAPI health endpoint
@app.get("/health")
async def health_check():
    return {"status": "healthy", "version": "1.0.0"}
```

## Related Resources

- [AWS Serverless](../aws-serverless/README.md) - Container deployment on AWS
- [Web Frameworks](../../5-python-production/web-frameworks/README.md) - FastAPI containerization
- [Scalable Architectures](../../99-blueprints/README.md) - Production patterns
