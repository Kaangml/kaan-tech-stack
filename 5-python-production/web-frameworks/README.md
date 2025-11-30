# Web Frameworks

Python web frameworks for API development, with focus on FastAPI and Flask.

## FastAPI

Modern, fast, async-first framework with automatic OpenAPI documentation.

### Basic Application Structure

```python
from fastapi import FastAPI, HTTPException, Depends, Query
from pydantic import BaseModel, Field
from typing import Optional, List
import uvicorn

app = FastAPI(
    title="My API",
    version="1.0.0",
    description="Production-ready API"
)

# Pydantic models for validation
class ItemCreate(BaseModel):
    name: str = Field(..., min_length=1, max_length=100)
    price: float = Field(..., gt=0)
    description: Optional[str] = None
    
    class Config:
        json_schema_extra = {
            "example": {
                "name": "Widget",
                "price": 9.99,
                "description": "A useful widget"
            }
        }

class ItemResponse(ItemCreate):
    id: int

# Routes
@app.get("/items", response_model=List[ItemResponse])
async def list_items(
    skip: int = Query(0, ge=0),
    limit: int = Query(10, ge=1, le=100)
):
    return await db.get_items(skip=skip, limit=limit)

@app.post("/items", response_model=ItemResponse, status_code=201)
async def create_item(item: ItemCreate):
    return await db.create_item(item)

@app.get("/items/{item_id}", response_model=ItemResponse)
async def get_item(item_id: int):
    item = await db.get_item(item_id)
    if not item:
        raise HTTPException(status_code=404, detail="Item not found")
    return item
```

### Dependency Injection

```python
from fastapi import Depends
from sqlalchemy.ext.asyncio import AsyncSession

# Database session dependency
async def get_db() -> AsyncSession:
    async with async_session() as session:
        try:
            yield session
        finally:
            await session.close()

# Authentication dependency
async def get_current_user(
    token: str = Depends(oauth2_scheme),
    db: AsyncSession = Depends(get_db)
) -> User:
    user = await verify_token(token, db)
    if not user:
        raise HTTPException(status_code=401, detail="Invalid token")
    return user

# Using dependencies
@app.get("/me")
async def read_users_me(current_user: User = Depends(get_current_user)):
    return current_user
```

### Background Tasks

```python
from fastapi import BackgroundTasks

async def send_notification(email: str, message: str):
    # Async email sending
    await email_service.send(email, message)

@app.post("/orders")
async def create_order(
    order: OrderCreate,
    background_tasks: BackgroundTasks
):
    result = await db.create_order(order)
    background_tasks.add_task(
        send_notification, 
        order.email, 
        f"Order {result.id} confirmed"
    )
    return result
```

### Middleware

```python
from fastapi.middleware.cors import CORSMiddleware
import time

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://example.com"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Custom middleware
@app.middleware("http")
async def add_process_time_header(request, call_next):
    start_time = time.time()
    response = await call_next(request)
    process_time = time.time() - start_time
    response.headers["X-Process-Time"] = str(process_time)
    return response
```

## Flask

Lightweight, flexible framework for simpler applications.

### Basic Application

```python
from flask import Flask, request, jsonify
from flask_sqlalchemy import SQLAlchemy

app = Flask(__name__)
app.config['SQLALCHEMY_DATABASE_URI'] = 'postgresql://user:pass@localhost/db'
db = SQLAlchemy(app)

# Models
class Item(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(100), nullable=False)
    price = db.Column(db.Float, nullable=False)

# Routes
@app.route('/items', methods=['GET'])
def list_items():
    items = Item.query.all()
    return jsonify([{'id': i.id, 'name': i.name, 'price': i.price} for i in items])

@app.route('/items', methods=['POST'])
def create_item():
    data = request.get_json()
    item = Item(name=data['name'], price=data['price'])
    db.session.add(item)
    db.session.commit()
    return jsonify({'id': item.id, 'name': item.name, 'price': item.price}), 201

@app.route('/items/<int:item_id>', methods=['GET'])
def get_item(item_id):
    item = Item.query.get_or_404(item_id)
    return jsonify({'id': item.id, 'name': item.name, 'price': item.price})
```

### Blueprints (Modular Structure)

```python
# blueprints/items.py
from flask import Blueprint, jsonify

items_bp = Blueprint('items', __name__, url_prefix='/items')

@items_bp.route('/')
def list_items():
    return jsonify([])

# app.py
from blueprints.items import items_bp

app = Flask(__name__)
app.register_blueprint(items_bp)
```

## SQLAlchemy

ORM for database operations, works with both Flask and FastAPI.

### Model Definition

```python
from sqlalchemy import Column, Integer, String, Float, ForeignKey, DateTime
from sqlalchemy.orm import relationship, declarative_base
from datetime import datetime

Base = declarative_base()

class User(Base):
    __tablename__ = 'users'
    
    id = Column(Integer, primary_key=True)
    email = Column(String(255), unique=True, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    
    orders = relationship("Order", back_populates="user")

class Order(Base):
    __tablename__ = 'orders'
    
    id = Column(Integer, primary_key=True)
    user_id = Column(Integer, ForeignKey('users.id'), nullable=False)
    total = Column(Float, nullable=False)
    
    user = relationship("User", back_populates="orders")
    items = relationship("OrderItem", back_populates="order")
```

### Async SQLAlchemy (FastAPI)

```python
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import sessionmaker

# Async engine
engine = create_async_engine(
    "postgresql+asyncpg://user:pass@localhost/db",
    echo=True
)

async_session = sessionmaker(
    engine, class_=AsyncSession, expire_on_commit=False
)

# Async queries
async def get_user_with_orders(db: AsyncSession, user_id: int):
    result = await db.execute(
        select(User)
        .options(selectinload(User.orders))
        .where(User.id == user_id)
    )
    return result.scalar_one_or_none()
```

## Alembic (Database Migrations)

### Setup

```bash
# Initialize
alembic init alembic

# Configure alembic.ini
sqlalchemy.url = postgresql://user:pass@localhost/db
```

### Migration Workflow

```python
# alembic/env.py
from models import Base
target_metadata = Base.metadata
```

```bash
# Create migration
alembic revision --autogenerate -m "add users table"

# Apply migration
alembic upgrade head

# Rollback
alembic downgrade -1

# Show history
alembic history
```

### Migration Script Example

```python
# alembic/versions/xxx_add_users_table.py
def upgrade():
    op.create_table(
        'users',
        sa.Column('id', sa.Integer(), primary_key=True),
        sa.Column('email', sa.String(255), nullable=False),
        sa.Column('created_at', sa.DateTime(), default=sa.func.now())
    )
    op.create_index('ix_users_email', 'users', ['email'], unique=True)

def downgrade():
    op.drop_index('ix_users_email')
    op.drop_table('users')
```

## FastAPI vs Flask

| Feature | FastAPI | Flask |
|---------|---------|-------|
| Performance | High (async) | Moderate |
| Type Hints | Required | Optional |
| OpenAPI Docs | Automatic | Manual |
| Learning Curve | Steeper | Gentle |
| Best For | APIs, microservices | Simple apps, prototypes |
| Async Support | Native | Extension needed |

## Production Checklist

```python
# FastAPI production config
app = FastAPI(
    docs_url=None if PRODUCTION else "/docs",  # Disable in prod
    redoc_url=None if PRODUCTION else "/redoc"
)

# Structured logging
import structlog
logger = structlog.get_logger()

# Health check
@app.get("/health")
async def health_check():
    return {"status": "healthy", "version": "1.0.0"}

# Uvicorn with workers
# uvicorn main:app --host 0.0.0.0 --port 8000 --workers 4
```

## Related Resources

- [Docker](../../7-infrastructure/docker/README.md) - Containerizing web applications
- [PostgreSQL](../../6-databases/postgres-advanced/README.md) - Database configuration
- [Clean Architecture](../../1-architecture-patterns/clean-architecture/README.md) - Project structure
