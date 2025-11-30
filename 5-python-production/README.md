# Python Production

Production-grade Python tools and frameworks.

## Topics

| Area | Description |
|------|-------------|
| [Web Frameworks](./web-frameworks/) | FastAPI, Flask, SQLAlchemy |
| [PDF Processing](./pdf-processing/) | PyMuPDF, pdfplumber, Docling |
| [Data Stack](./data-stack/) | Pandas, Polars, NumPy, visualization |
| Testing | pytest, coverage, mocking |

## FastAPI Quick Start

```python
from fastapi import FastAPI, Depends
from pydantic import BaseModel

app = FastAPI()

class Item(BaseModel):
    name: str
    price: float

@app.post("/items")
async def create_item(item: Item):
    return {"id": 1, **item.dict()}
```

## Key Libraries

| Category | Libraries |
|----------|-----------|
| Web | FastAPI, Flask, Uvicorn |
| ORM | SQLAlchemy, Alembic |
| Data | Pandas, Polars, NumPy |
| PDF | PyMuPDF, pdfplumber |
| Viz | Matplotlib, Seaborn |

## Related Resources
- [Clean Architecture](../1-architecture-patterns/clean-architecture/) - Project structure
- [Docker](../7-infrastructure/docker/) - Containerizing Python apps
