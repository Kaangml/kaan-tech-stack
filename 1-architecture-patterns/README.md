# Architecture Patterns

Software architecture principles and patterns for maintainable systems.

## Topics

| Pattern | Description |
|---------|-------------|
| [Clean Architecture](./clean-architecture/) | Layered architecture with dependency inversion |
| [Clean Code](./clean-code/) | Readability, naming, function design |
| [Design Patterns](./design-patterns/) | GoF patterns and modern alternatives |
| [System Design](./system-design/) | Distributed systems and scalability |
| [Algorithms](./algorithms/) | Data structures and complexity analysis |

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
