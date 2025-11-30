# Architecture Patterns

Software architecture principles and patterns for maintainable systems.

## Topics

| Pattern | Description |
|---------|-------------|
| [Clean Architecture](./clean-architecture/) | Layered architecture with dependency inversion |
| Design Patterns | GoF patterns and modern alternatives |
| System Design | Distributed systems and scalability |
| Testing | Unit, integration, and E2E testing strategies |

## Key Principles

### SOLID
- **S**ingle Responsibility - One reason to change
- **O**pen/Closed - Open for extension, closed for modification
- **L**iskov Substitution - Subtypes must be substitutable
- **I**nterface Segregation - Many specific interfaces over one general
- **D**ependency Inversion - Depend on abstractions

### Clean Architecture Layers
```
Domain (Entities) → Application (Use Cases) → Interface (Controllers) → Infrastructure (DB, APIs)
```

## Related Blueprints
- [Legal RAG System](../99-blueprints/legal-rag-graphdb/) - Clean architecture in practice
- [Browser Agent](../99-blueprints/autonomous-browser-agent/) - Agent architecture patterns
