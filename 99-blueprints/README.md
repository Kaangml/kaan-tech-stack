# 99 Blueprints

Production-ready architectures and end-to-end implementations.

## What's a Blueprint?

Blueprints are **complete, working architectures** that combine multiple technologies from this knowledge base into production systems. Each blueprint includes:

- Architecture diagrams
- Code examples with context
- Technology selection rationale
- Deployment considerations
- Lessons learned

## Available Blueprints

| Blueprint | Technologies | Use Case |
|-----------|--------------|----------|
| [Autonomous Browser Agent](./autonomous-browser-agent/) | LangGraph, Playwright, GPT-4o | AI agent that navigates web autonomously |
| [Legal RAG with GraphDB](./legal-rag-graphdb/) | Neo4j, Qdrant, PyMuPDF | Hybrid search for legal documents |
| [Geospatial Pipeline](./geospatial-pipeline/) | PostGIS, Open3D, Dask | Large-scale 3D point cloud processing |
| [Scalable Scraping](./scalable-scraping/) | Playwright, Redis, FastAPI | High-volume web scraping architecture |
| [Data Pipelines](./data-pipelines/) | Medallion, dbt, Airflow | Enterprise data pipeline patterns |

## Blueprint Selection Guide

```
Need an AI that browses the web?
└── autonomous-browser-agent/

Building document Q&A with relationships?
└── legal-rag-graphdb/

Processing 3D/geospatial data at scale?
└── geospatial-pipeline/

Scraping 1000+ pages reliably?
└── scalable-scraping/

Designing data warehouse ETL?
└── data-pipelines/
```

## How to Use Blueprints

1. **Start with the README** - Understand the architecture and components
2. **Check prerequisites** - Ensure you have required tools/accounts
3. **Adapt, don't copy** - Blueprints are templates, not turnkey solutions
4. **Combine patterns** - Many blueprints share patterns you can mix

## Related Sections

- [AI/ML](../3-ai-ml/) - Core AI components used in blueprints
- [Data Engineering](../2-data-engineering/) - Pipeline patterns
- [Infrastructure](../7-infrastructure/) - Deployment patterns
