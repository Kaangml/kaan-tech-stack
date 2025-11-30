# AWS Serverless

Production patterns for Lambda, API Gateway, Step Functions, and serverless architectures.

## Table of Contents

- [API Gateway](#api-gateway)
- [Lambda Patterns](#lambda-patterns)
- [FastAPI + Lambda (Mangum)](#fastapi--lambda-mangum)
- [Step Functions](#step-functions)
- [EventBridge](#eventbridge)
- [SAM (Serverless Application Model)](#sam-serverless-application-model)
- [S3 & DynamoDB](#s3--dynamodb)
- [AWS Glue](#aws-glue)
- [Cost Optimization](#cost-optimization)
- [Observability](#observability)

---

## API Gateway

### REST API vs HTTP API

| Feature | REST API | HTTP API |
|---------|----------|----------|
| Cost | $3.50/million | $1.00/million |
| Latency | ~10ms | ~5ms |
| WebSocket | Yes | No |
| Usage Plans | Yes | No |
| Request Validation | Yes | Basic |
| Caching | Yes | No |
| Private Endpoints | Yes | Yes |

**Rule of thumb:** Use HTTP API unless you need REST-specific features.

### HTTP API with Lambda (Terraform)

```hcl
# API Gateway HTTP API
resource "aws_apigatewayv2_api" "main" {
  name          = "${var.project}-api"
  protocol_type = "HTTP"
  
  cors_configuration {
    allow_origins = var.allowed_origins
    allow_methods = ["GET", "POST", "PUT", "DELETE", "OPTIONS"]
    allow_headers = ["Content-Type", "Authorization"]
    max_age       = 3600
  }
}

# Stage
resource "aws_apigatewayv2_stage" "main" {
  api_id      = aws_apigatewayv2_api.main.id
  name        = var.environment
  auto_deploy = true
  
  access_log_settings {
    destination_arn = aws_cloudwatch_log_group.api.arn
    format = jsonencode({
      requestId      = "$context.requestId"
      ip             = "$context.identity.sourceIp"
      method         = "$context.httpMethod"
      path           = "$context.path"
      status         = "$context.status"
      latency        = "$context.responseLatency"
      integrationErr = "$context.integrationErrorMessage"
    })
  }
  
  default_route_settings {
    throttling_burst_limit = 1000
    throttling_rate_limit  = 500
  }
}

# Lambda Integration
resource "aws_apigatewayv2_integration" "lambda" {
  api_id                 = aws_apigatewayv2_api.main.id
  integration_type       = "AWS_PROXY"
  integration_uri        = aws_lambda_function.api.invoke_arn
  payload_format_version = "2.0"
}

# Routes
resource "aws_apigatewayv2_route" "get_items" {
  api_id    = aws_apigatewayv2_api.main.id
  route_key = "GET /items"
  target    = "integrations/${aws_apigatewayv2_integration.lambda.id}"
}

resource "aws_apigatewayv2_route" "post_items" {
  api_id    = aws_apigatewayv2_api.main.id
  route_key = "POST /items"
  target    = "integrations/${aws_apigatewayv2_integration.lambda.id}"
}

resource "aws_apigatewayv2_route" "get_item" {
  api_id    = aws_apigatewayv2_api.main.id
  route_key = "GET /items/{id}"
  target    = "integrations/${aws_apigatewayv2_integration.lambda.id}"
}

# Lambda permission
resource "aws_lambda_permission" "api" {
  statement_id  = "AllowAPIGateway"
  action        = "lambda:InvokeFunction"
  function_name = aws_lambda_function.api.function_name
  principal     = "apigateway.amazonaws.com"
  source_arn    = "${aws_apigatewayv2_api.main.execution_arn}/*/*"
}
```

### REST API with Authorizers

```hcl
# REST API
resource "aws_api_gateway_rest_api" "main" {
  name = "${var.project}-rest-api"
  
  endpoint_configuration {
    types = ["REGIONAL"]
  }
}

# Cognito Authorizer
resource "aws_api_gateway_authorizer" "cognito" {
  name          = "cognito-authorizer"
  rest_api_id   = aws_api_gateway_rest_api.main.id
  type          = "COGNITO_USER_POOLS"
  provider_arns = [aws_cognito_user_pool.main.arn]
}

# Lambda Authorizer (Custom)
resource "aws_api_gateway_authorizer" "lambda" {
  name                   = "jwt-authorizer"
  rest_api_id            = aws_api_gateway_rest_api.main.id
  type                   = "TOKEN"
  authorizer_uri         = aws_lambda_function.authorizer.invoke_arn
  authorizer_credentials = aws_iam_role.authorizer.arn
  
  identity_validation_expression = "^Bearer [-0-9a-zA-Z._]+$"
  authorizer_result_ttl_in_seconds = 300
}

# Lambda Authorizer Function
# authorizer.py
def handler(event, context):
    token = event['authorizationToken'].replace('Bearer ', '')
    
    try:
        # Verify JWT
        payload = jwt.decode(token, SECRET_KEY, algorithms=['HS256'])
        
        return {
            'principalId': payload['sub'],
            'policyDocument': {
                'Version': '2012-10-17',
                'Statement': [{
                    'Action': 'execute-api:Invoke',
                    'Effect': 'Allow',
                    'Resource': event['methodArn']
                }]
            },
            'context': {
                'userId': payload['sub'],
                'email': payload.get('email', '')
            }
        }
    except jwt.InvalidTokenError:
        raise Exception('Unauthorized')
```

### Request/Response Mapping

```python
# API Gateway v2 (HTTP API) event format
def http_api_handler(event, context):
    """
    event structure:
    {
        "version": "2.0",
        "routeKey": "GET /items/{id}",
        "rawPath": "/items/123",
        "rawQueryString": "limit=10&offset=0",
        "headers": {"authorization": "Bearer xxx", ...},
        "queryStringParameters": {"limit": "10", "offset": "0"},
        "pathParameters": {"id": "123"},
        "body": "...",  # JSON string
        "isBase64Encoded": false,
        "requestContext": {
            "accountId": "123456789",
            "apiId": "abc123",
            "authorizer": {"jwt": {"claims": {...}}},
            "http": {"method": "GET", "path": "/items/123"}
        }
    }
    """
    
    # Extract parameters
    item_id = event.get('pathParameters', {}).get('id')
    query_params = event.get('queryStringParameters', {}) or {}
    limit = int(query_params.get('limit', 10))
    
    # Get auth context
    claims = event.get('requestContext', {}).get('authorizer', {}).get('jwt', {}).get('claims', {})
    user_id = claims.get('sub')
    
    # Parse body (POST/PUT)
    body = None
    if event.get('body'):
        body = json.loads(event['body'])
    
    # Response format
    return {
        "statusCode": 200,
        "headers": {
            "Content-Type": "application/json",
            "Access-Control-Allow-Origin": "*"
        },
        "body": json.dumps({"id": item_id, "data": "..."})
    }
```

---

## Lambda Patterns

### Cold Start Optimization

```python
# WRONG: Import inside handler (repeated on every invocation)
def handler(event, context):
    import boto3  # Cold start penalty
    import pandas  # Heavy import
    
    s3 = boto3.client('s3')  # Connection created every time
    ...

# CORRECT: Module-level initialization
import boto3
import pandas as pd
from functools import lru_cache

# Initialized once, reused across invocations
s3 = boto3.client('s3')
dynamodb = boto3.resource('dynamodb')
table = dynamodb.Table(os.environ['TABLE_NAME'])

# Cache expensive computations
@lru_cache(maxsize=100)
def get_config(config_key: str) -> dict:
    response = s3.get_object(Bucket='config-bucket', Key=config_key)
    return json.loads(response['Body'].read())

def handler(event, context):
    # Use pre-initialized clients
    config = get_config('app-config.json')
    ...
```

### Provisioned Concurrency

```hcl
# For latency-critical APIs
resource "aws_lambda_function" "api" {
  function_name = "api-handler"
  ...
}

resource "aws_lambda_alias" "live" {
  name             = "live"
  function_name    = aws_lambda_function.api.function_name
  function_version = aws_lambda_function.api.version
}

resource "aws_lambda_provisioned_concurrency_config" "api" {
  function_name                     = aws_lambda_alias.live.function_name
  qualifier                         = aws_lambda_alias.live.name
  provisioned_concurrent_executions = 10
}
```

### Async Processing with SQS

```python
from aws_lambda_powertools import Logger, Tracer
from aws_lambda_powertools.utilities.batch import BatchProcessor, EventType

logger = Logger()
tracer = Tracer()
processor = BatchProcessor(event_type=EventType.SQS)

@tracer.capture_method
def process_record(record: dict):
    body = json.loads(record['body'])
    
    # Process message
    result = heavy_processing(body)
    
    # If processing fails, message returns to queue
    if not result['success']:
        raise ProcessingError(result['error'])
    
    return result

@logger.inject_lambda_context
@tracer.capture_lambda_handler
def handler(event, context):
    batch = event['Records']
    
    with processor(records=batch, handler=process_record):
        processed = processor.process()
    
    # Return partial batch failures
    return processor.response()
```

---

## FastAPI + Lambda (Mangum)

### Why Mangum?

Mangum wraps ASGI applications (FastAPI, Starlette) to run on Lambda.

```
┌─────────────┐      ┌──────────────┐      ┌─────────────┐
│ API Gateway │─────▶│    Mangum    │─────▶│   FastAPI   │
└─────────────┘      │  (Adapter)   │      │ Application │
                     └──────────────┘      └─────────────┘
```

### Basic Setup

```python
# app/main.py
from fastapi import FastAPI, Depends, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from mangum import Mangum
from pydantic import BaseModel

app = FastAPI(
    title="My Serverless API",
    version="1.0.0",
    root_path="/prod"  # API Gateway stage
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"]
)

class Item(BaseModel):
    name: str
    price: float
    description: str | None = None

@app.get("/health")
async def health():
    return {"status": "healthy"}

@app.post("/items", response_model=Item)
async def create_item(item: Item):
    # DynamoDB operation
    return item

@app.get("/items/{item_id}")
async def get_item(item_id: str):
    item = dynamodb_get_item(item_id)
    if not item:
        raise HTTPException(status_code=404, detail="Item not found")
    return item

# Mangum handler for Lambda
handler = Mangum(app, lifespan="off")
```

### Production Mangum Setup

```python
# app/main.py
from fastapi import FastAPI
from mangum import Mangum
from contextlib import asynccontextmanager
import boto3

# Module-level clients (reused across invocations)
dynamodb = boto3.resource('dynamodb')
s3 = boto3.client('s3')

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: Initialize connections
    app.state.db = dynamodb.Table(os.environ['TABLE_NAME'])
    yield
    # Shutdown: Cleanup (rarely runs in Lambda)

app = FastAPI(lifespan=lifespan)

# Include routers
from app.routers import items, users
app.include_router(items.router, prefix="/items", tags=["items"])
app.include_router(users.router, prefix="/users", tags=["users"])

# Mangum with configuration
handler = Mangum(
    app,
    lifespan="off",  # Lambda doesn't support ASGI lifespan
    api_gateway_base_path="/prod"
)
```

### Project Structure

```
serverless-fastapi/
├── app/
│   ├── __init__.py
│   ├── main.py
│   ├── config.py
│   ├── dependencies.py
│   ├── routers/
│   │   ├── __init__.py
│   │   ├── items.py
│   │   └── users.py
│   ├── models/
│   │   ├── __init__.py
│   │   └── item.py
│   └── services/
│       ├── __init__.py
│       └── dynamodb.py
├── tests/
├── requirements.txt
├── template.yaml          # SAM template
└── samconfig.toml
```

### SAM Template for FastAPI

```yaml
# template.yaml
AWSTemplateFormatVersion: '2010-09-09'
Transform: AWS::Serverless-2016-10-31

Globals:
  Function:
    Timeout: 30
    MemorySize: 256
    Runtime: python3.11
    Environment:
      Variables:
        TABLE_NAME: !Ref ItemsTable
        LOG_LEVEL: INFO

Resources:
  ApiFunction:
    Type: AWS::Serverless::Function
    Properties:
      Handler: app.main.handler
      CodeUri: .
      Description: FastAPI application
      Architectures:
        - arm64  # Graviton: 20% cheaper, faster
      Events:
        Api:
          Type: HttpApi
          Properties:
            ApiId: !Ref HttpApi
      Policies:
        - DynamoDBCrudPolicy:
            TableName: !Ref ItemsTable
    Metadata:
      BuildMethod: python3.11

  HttpApi:
    Type: AWS::Serverless::HttpApi
    Properties:
      StageName: prod
      CorsConfiguration:
        AllowOrigins:
          - "*"
        AllowMethods:
          - GET
          - POST
          - PUT
          - DELETE
        AllowHeaders:
          - Content-Type
          - Authorization

  ItemsTable:
    Type: AWS::DynamoDB::Table
    Properties:
      TableName: !Sub ${AWS::StackName}-items
      BillingMode: PAY_PER_REQUEST
      AttributeDefinitions:
        - AttributeName: pk
          AttributeType: S
        - AttributeName: sk
          AttributeType: S
      KeySchema:
        - AttributeName: pk
          KeyType: HASH
        - AttributeName: sk
          KeyType: RANGE

Outputs:
  ApiUrl:
    Description: API Gateway URL
    Value: !Sub "https://${HttpApi}.execute-api.${AWS::Region}.amazonaws.com/prod"
```

---

## Step Functions

### When to Use Step Functions

| Use Case | Step Functions | Lambda Chaining |
|----------|----------------|-----------------|
| Long workflows (>15 min) | ✅ | ❌ |
| Complex branching | ✅ | ❌ |
| Retry/error handling | ✅ (built-in) | Manual |
| Visual debugging | ✅ | ❌ |
| State management | ✅ | Manual |
| Cost (simple flows) | Higher | Lower |

### Standard vs Express

| Feature | Standard | Express |
|---------|----------|---------|
| Max duration | 1 year | 5 minutes |
| Execution rate | 2,000/sec | 100,000/sec |
| Pricing | Per transition | Per duration |
| Best for | Long workflows | High-volume, short |

### State Machine Definition (ASL)

```json
{
  "Comment": "Order Processing Pipeline",
  "StartAt": "ValidateOrder",
  "States": {
    "ValidateOrder": {
      "Type": "Task",
      "Resource": "${ValidateOrderFunction}",
      "Next": "CheckInventory",
      "Catch": [{
        "ErrorEquals": ["ValidationError"],
        "Next": "OrderFailed"
      }]
    },
    
    "CheckInventory": {
      "Type": "Task",
      "Resource": "${CheckInventoryFunction}",
      "Next": "InventoryChoice"
    },
    
    "InventoryChoice": {
      "Type": "Choice",
      "Choices": [
        {
          "Variable": "$.inventoryAvailable",
          "BooleanEquals": true,
          "Next": "ProcessPayment"
        }
      ],
      "Default": "WaitForInventory"
    },
    
    "WaitForInventory": {
      "Type": "Wait",
      "Seconds": 300,
      "Next": "CheckInventory"
    },
    
    "ProcessPayment": {
      "Type": "Task",
      "Resource": "${ProcessPaymentFunction}",
      "Retry": [{
        "ErrorEquals": ["PaymentGatewayError"],
        "IntervalSeconds": 2,
        "MaxAttempts": 3,
        "BackoffRate": 2.0
      }],
      "Next": "ParallelFulfillment"
    },
    
    "ParallelFulfillment": {
      "Type": "Parallel",
      "Branches": [
        {
          "StartAt": "SendConfirmation",
          "States": {
            "SendConfirmation": {
              "Type": "Task",
              "Resource": "${SendEmailFunction}",
              "End": true
            }
          }
        },
        {
          "StartAt": "UpdateInventory",
          "States": {
            "UpdateInventory": {
              "Type": "Task",
              "Resource": "${UpdateInventoryFunction}",
              "End": true
            }
          }
        },
        {
          "StartAt": "CreateShipment",
          "States": {
            "CreateShipment": {
              "Type": "Task",
              "Resource": "${CreateShipmentFunction}",
              "End": true
            }
          }
        }
      ],
      "Next": "OrderCompleted"
    },
    
    "OrderCompleted": {
      "Type": "Succeed"
    },
    
    "OrderFailed": {
      "Type": "Fail",
      "Error": "OrderProcessingFailed",
      "Cause": "Order validation or processing failed"
    }
  }
}
```

### SAM Template for Step Functions

```yaml
Resources:
  OrderStateMachine:
    Type: AWS::Serverless::StateMachine
    Properties:
      Name: order-processing
      DefinitionUri: statemachine/order.asl.json
      DefinitionSubstitutions:
        ValidateOrderFunction: !GetAtt ValidateOrderFunction.Arn
        CheckInventoryFunction: !GetAtt CheckInventoryFunction.Arn
        ProcessPaymentFunction: !GetAtt ProcessPaymentFunction.Arn
        SendEmailFunction: !GetAtt SendEmailFunction.Arn
        UpdateInventoryFunction: !GetAtt UpdateInventoryFunction.Arn
        CreateShipmentFunction: !GetAtt CreateShipmentFunction.Arn
      Policies:
        - LambdaInvokePolicy:
            FunctionName: !Ref ValidateOrderFunction
        - LambdaInvokePolicy:
            FunctionName: !Ref CheckInventoryFunction
        # ... other policies
      Events:
        ApiEvent:
          Type: Api
          Properties:
            Path: /orders
            Method: POST
```

### Python SDK Integration

```python
import boto3
import json

sfn = boto3.client('stepfunctions')

def start_order_workflow(order: dict) -> str:
    response = sfn.start_execution(
        stateMachineArn='arn:aws:states:us-east-1:123456789:stateMachine:order-processing',
        name=f"order-{order['order_id']}-{int(time.time())}",
        input=json.dumps(order)
    )
    return response['executionArn']

def check_execution_status(execution_arn: str) -> dict:
    response = sfn.describe_execution(executionArn=execution_arn)
    return {
        'status': response['status'],  # RUNNING, SUCCEEDED, FAILED, TIMED_OUT, ABORTED
        'output': json.loads(response.get('output', '{}')) if response['status'] == 'SUCCEEDED' else None,
        'error': response.get('error'),
        'cause': response.get('cause')
    }
```

---

## EventBridge

### Event-Driven Architecture

```
┌─────────────┐     ┌──────────────┐     ┌─────────────────┐
│   Source    │────▶│  EventBridge │────▶│    Targets      │
│ (Producer)  │     │    (Bus)     │     │ (Consumers)     │
└─────────────┘     └──────────────┘     └─────────────────┘
                           │
                    ┌──────┴──────┐
                    │    Rules    │
                    │  (Filters)  │
                    └─────────────┘
```

### Custom Event Bus

```hcl
# EventBridge Bus
resource "aws_cloudwatch_event_bus" "orders" {
  name = "orders-bus"
}

# Rule: Route high-value orders
resource "aws_cloudwatch_event_rule" "high_value_orders" {
  name           = "high-value-orders"
  event_bus_name = aws_cloudwatch_event_bus.orders.name
  
  event_pattern = jsonencode({
    source      = ["orders.service"]
    detail-type = ["Order Created"]
    detail = {
      total = [{
        numeric = [">=", 1000]
      }]
    }
  })
}

# Target: Lambda for high-value order processing
resource "aws_cloudwatch_event_target" "high_value_processor" {
  rule           = aws_cloudwatch_event_rule.high_value_orders.name
  event_bus_name = aws_cloudwatch_event_bus.orders.name
  target_id      = "process-high-value"
  arn            = aws_lambda_function.high_value_processor.arn
}

# Rule: All orders to analytics
resource "aws_cloudwatch_event_rule" "all_orders" {
  name           = "all-orders-analytics"
  event_bus_name = aws_cloudwatch_event_bus.orders.name
  
  event_pattern = jsonencode({
    source      = ["orders.service"]
    detail-type = ["Order Created", "Order Updated", "Order Cancelled"]
  })
}

resource "aws_cloudwatch_event_target" "analytics" {
  rule           = aws_cloudwatch_event_rule.all_orders.name
  event_bus_name = aws_cloudwatch_event_bus.orders.name
  target_id      = "analytics-firehose"
  arn            = aws_kinesis_firehose_delivery_stream.analytics.arn
  role_arn       = aws_iam_role.eventbridge_firehose.arn
}
```

### Publishing Events

```python
import boto3
import json
from datetime import datetime

eventbridge = boto3.client('events')

def publish_order_event(order: dict, event_type: str):
    """Publish order event to EventBridge"""
    eventbridge.put_events(
        Entries=[
            {
                'Source': 'orders.service',
                'DetailType': event_type,
                'Detail': json.dumps({
                    **order,
                    'timestamp': datetime.utcnow().isoformat()
                }),
                'EventBusName': 'orders-bus'
            }
        ]
    )

# Usage
publish_order_event(
    order={'order_id': '12345', 'total': 1500, 'items': [...]},
    event_type='Order Created'
)
```

### Scheduled Events

```hcl
# Cron job: Daily report at 9 AM UTC
resource "aws_cloudwatch_event_rule" "daily_report" {
  name                = "daily-report"
  schedule_expression = "cron(0 9 * * ? *)"
}

resource "aws_cloudwatch_event_target" "daily_report" {
  rule      = aws_cloudwatch_event_rule.daily_report.name
  target_id = "generate-report"
  arn       = aws_lambda_function.report_generator.arn
}

# Rate-based: Every 5 minutes
resource "aws_cloudwatch_event_rule" "health_check" {
  name                = "health-check"
  schedule_expression = "rate(5 minutes)"
}
```

---

## SAM (Serverless Application Model)

### Project Structure

```
my-sam-app/
├── template.yaml           # SAM template
├── samconfig.toml          # Deployment config
├── src/
│   ├── handlers/
│   │   ├── __init__.py
│   │   ├── api.py
│   │   └── worker.py
│   ├── services/
│   │   └── dynamodb.py
│   └── requirements.txt
├── tests/
│   ├── unit/
│   └── integration/
└── events/                 # Test events
    └── api-event.json
```

### Complete SAM Template

```yaml
AWSTemplateFormatVersion: '2010-09-09'
Transform: AWS::Serverless-2016-10-31

Description: Production serverless application

Parameters:
  Environment:
    Type: String
    Default: dev
    AllowedValues: [dev, staging, prod]

Globals:
  Function:
    Timeout: 30
    MemorySize: 256
    Runtime: python3.11
    Architectures:
      - arm64
    Tracing: Active
    Environment:
      Variables:
        ENVIRONMENT: !Ref Environment
        LOG_LEVEL: !If [IsProd, INFO, DEBUG]
        POWERTOOLS_SERVICE_NAME: my-app

Conditions:
  IsProd: !Equals [!Ref Environment, prod]

Resources:
  # API
  HttpApi:
    Type: AWS::Serverless::HttpApi
    Properties:
      StageName: !Ref Environment
      AccessLogSettings:
        DestinationArn: !GetAtt ApiLogGroup.Arn
        Format: '{"requestId":"$context.requestId","ip":"$context.identity.sourceIp","method":"$context.httpMethod","path":"$context.path","status":"$context.status","latency":"$context.responseLatency"}'

  # API Handler
  ApiFunction:
    Type: AWS::Serverless::Function
    Properties:
      Handler: src.handlers.api.handler
      Description: API endpoints
      Events:
        GetItems:
          Type: HttpApi
          Properties:
            ApiId: !Ref HttpApi
            Path: /items
            Method: GET
        PostItem:
          Type: HttpApi
          Properties:
            ApiId: !Ref HttpApi
            Path: /items
            Method: POST
      Policies:
        - DynamoDBCrudPolicy:
            TableName: !Ref ItemsTable
      Layers:
        - !Ref DependenciesLayer
    Metadata:
      BuildMethod: python3.11

  # Async Worker
  WorkerFunction:
    Type: AWS::Serverless::Function
    Properties:
      Handler: src.handlers.worker.handler
      Description: SQS message processor
      Timeout: 60
      Events:
        SqsTrigger:
          Type: SQS
          Properties:
            Queue: !GetAtt ProcessingQueue.Arn
            BatchSize: 10
            FunctionResponseTypes:
              - ReportBatchItemFailures
      Policies:
        - SQSPollerPolicy:
            QueueName: !GetAtt ProcessingQueue.QueueName
        - DynamoDBCrudPolicy:
            TableName: !Ref ItemsTable

  # Lambda Layer
  DependenciesLayer:
    Type: AWS::Serverless::LayerVersion
    Properties:
      LayerName: dependencies
      ContentUri: src/
      CompatibleRuntimes:
        - python3.11
    Metadata:
      BuildMethod: python3.11

  # DynamoDB Table
  ItemsTable:
    Type: AWS::DynamoDB::Table
    Properties:
      TableName: !Sub ${AWS::StackName}-items
      BillingMode: PAY_PER_REQUEST
      AttributeDefinitions:
        - AttributeName: pk
          AttributeType: S
        - AttributeName: sk
          AttributeType: S
        - AttributeName: gsi1pk
          AttributeType: S
        - AttributeName: gsi1sk
          AttributeType: S
      KeySchema:
        - AttributeName: pk
          KeyType: HASH
        - AttributeName: sk
          KeyType: RANGE
      GlobalSecondaryIndexes:
        - IndexName: gsi1
          KeySchema:
            - AttributeName: gsi1pk
              KeyType: HASH
            - AttributeName: gsi1sk
              KeyType: RANGE
          Projection:
            ProjectionType: ALL
      PointInTimeRecoverySpecification:
        PointInTimeRecoveryEnabled: !If [IsProd, true, false]

  # SQS Queue
  ProcessingQueue:
    Type: AWS::SQS::Queue
    Properties:
      QueueName: !Sub ${AWS::StackName}-processing
      VisibilityTimeout: 120
      RedrivePolicy:
        deadLetterTargetArn: !GetAtt DLQ.Arn
        maxReceiveCount: 3

  DLQ:
    Type: AWS::SQS::Queue
    Properties:
      QueueName: !Sub ${AWS::StackName}-dlq
      MessageRetentionPeriod: 1209600  # 14 days

  # CloudWatch Log Groups
  ApiLogGroup:
    Type: AWS::Logs::LogGroup
    Properties:
      LogGroupName: !Sub /aws/apigateway/${AWS::StackName}
      RetentionInDays: !If [IsProd, 90, 14]

Outputs:
  ApiUrl:
    Description: API Gateway endpoint URL
    Value: !Sub "https://${HttpApi}.execute-api.${AWS::Region}.amazonaws.com/${Environment}"
  
  TableName:
    Description: DynamoDB table name
    Value: !Ref ItemsTable
```

### SAM CLI Commands

```bash
# Build
sam build --use-container

# Local testing
sam local start-api
sam local invoke ApiFunction --event events/api-event.json

# Deploy
sam deploy --guided  # First time
sam deploy            # Subsequent

# Logs
sam logs -n ApiFunction --stack-name my-app --tail

# Sync (hot reload in dev)
sam sync --watch --stack-name my-app-dev
```

---

## S3 & DynamoDB

### S3 Patterns

```python
import boto3
from botocore.config import Config

# Optimized S3 client
s3 = boto3.client('s3', config=Config(
    max_pool_connections=50,
    retries={'max_attempts': 3, 'mode': 'adaptive'}
))

# Multipart upload for large files
def upload_large_file(file_path: str, bucket: str, key: str):
    from boto3.s3.transfer import TransferConfig
    
    config = TransferConfig(
        multipart_threshold=8 * 1024 * 1024,  # 8MB
        max_concurrency=10,
        multipart_chunksize=8 * 1024 * 1024
    )
    
    s3.upload_file(file_path, bucket, key, Config=config)

# Presigned URLs
def get_upload_url(bucket: str, key: str, expires: int = 3600) -> str:
    return s3.generate_presigned_url(
        'put_object',
        Params={
            'Bucket': bucket,
            'Key': key,
            'ContentType': 'application/octet-stream'
        },
        ExpiresIn=expires
    )

# S3 Select (query inside S3)
def query_s3_json(bucket: str, key: str, sql: str) -> list:
    response = s3.select_object_content(
        Bucket=bucket,
        Key=key,
        ExpressionType='SQL',
        Expression=sql,
        InputSerialization={'JSON': {'Type': 'Lines'}},
        OutputSerialization={'JSON': {}}
    )
    
    records = []
    for event in response['Payload']:
        if 'Records' in event:
            records.append(event['Records']['Payload'].decode())
    return records
```

### DynamoDB Single-Table Design

```python
from boto3.dynamodb.conditions import Key, Attr
import boto3

dynamodb = boto3.resource('dynamodb')
table = dynamodb.Table('my-app')

# Entity types in single table
# pk=USER#123, sk=PROFILE -> User profile
# pk=USER#123, sk=ORDER#456 -> User's order
# pk=ORDER#456, sk=ORDER#456 -> Order details
# gsi1pk=ORDER#456, gsi1sk=USER#123 -> Query orders

def create_user(user_id: str, name: str, email: str):
    table.put_item(Item={
        'pk': f'USER#{user_id}',
        'sk': 'PROFILE',
        'name': name,
        'email': email,
        'entity_type': 'user',
        'gsi1pk': f'EMAIL#{email}',
        'gsi1sk': f'USER#{user_id}'
    })

def create_order(order_id: str, user_id: str, items: list, total: float):
    # Transaction: create order + update user
    table.meta.client.transact_write_items(
        TransactItems=[
            {
                'Put': {
                    'TableName': 'my-app',
                    'Item': {
                        'pk': {'S': f'ORDER#{order_id}'},
                        'sk': {'S': f'ORDER#{order_id}'},
                        'user_id': {'S': user_id},
                        'items': {'L': [{'S': str(i)} for i in items]},
                        'total': {'N': str(total)},
                        'status': {'S': 'pending'},
                        'entity_type': {'S': 'order'},
                        'gsi1pk': {'S': f'USER#{user_id}'},
                        'gsi1sk': {'S': f'ORDER#{order_id}'}
                    }
                }
            },
            {
                'Update': {
                    'TableName': 'my-app',
                    'Key': {
                        'pk': {'S': f'USER#{user_id}'},
                        'sk': {'S': 'PROFILE'}
                    },
                    'UpdateExpression': 'SET order_count = order_count + :inc',
                    'ExpressionAttributeValues': {':inc': {'N': '1'}}
                }
            }
        ]
    )

def get_user_orders(user_id: str, limit: int = 20):
    response = table.query(
        IndexName='gsi1',
        KeyConditionExpression=Key('gsi1pk').eq(f'USER#{user_id}') & Key('gsi1sk').begins_with('ORDER#'),
        Limit=limit,
        ScanIndexForward=False  # Newest first
    )
    return response['Items']
```

---

## Cost Optimization

### Lambda Memory/Duration Sweet Spot

```python
# aws_lambda_power_tuning results typically show:
# - 128MB: Cheapest per invocation, slowest
# - 1024MB: Often optimal price/performance
# - 1769MB: Gets 1 full vCPU

# Example: API handler
# 256MB @ 200ms = $0.0000008
# 1024MB @ 50ms = $0.0000005  <- Winner

# Use power tuning tool:
# github.com/alexcasalboni/aws-lambda-power-tuning
```

### Reserved Concurrency vs Provisioned

```hcl
# Reserved: Limits max concurrent (free)
resource "aws_lambda_function" "api" {
  reserved_concurrent_executions = 100  # Max 100 concurrent
}

# Provisioned: Pre-warmed instances ($$$)
resource "aws_lambda_provisioned_concurrency_config" "api" {
  provisioned_concurrent_executions = 10  # Always 10 warm
}
```

### Cost Monitoring

```python
# Tag all resources for cost allocation
tags = {
    'Project': 'my-app',
    'Environment': 'prod',
    'Team': 'backend'
}

# Use AWS Cost Explorer API
import boto3

ce = boto3.client('ce')

def get_lambda_costs(days: int = 30):
    response = ce.get_cost_and_usage(
        TimePeriod={
            'Start': (datetime.now() - timedelta(days=days)).strftime('%Y-%m-%d'),
            'End': datetime.now().strftime('%Y-%m-%d')
        },
        Granularity='DAILY',
        Metrics=['UnblendedCost'],
        Filter={
            'Dimensions': {
                'Key': 'SERVICE',
                'Values': ['AWS Lambda']
            }
        },
        GroupBy=[
            {'Type': 'DIMENSION', 'Key': 'USAGE_TYPE'}
        ]
    )
    return response['ResultsByTime']
```

---

## Observability

### AWS Lambda Powertools

```python
from aws_lambda_powertools import Logger, Tracer, Metrics
from aws_lambda_powertools.metrics import MetricUnit
from aws_lambda_powertools.utilities.typing import LambdaContext

logger = Logger()
tracer = Tracer()
metrics = Metrics()

@logger.inject_lambda_context(log_event=True)
@tracer.capture_lambda_handler
@metrics.log_metrics(capture_cold_start_metric=True)
def handler(event: dict, context: LambdaContext):
    logger.info("Processing request", extra={"order_id": event.get("order_id")})
    
    with tracer.capture_method():
        result = process_order(event)
    
    metrics.add_metric(name="OrdersProcessed", unit=MetricUnit.Count, value=1)
    metrics.add_metric(name="OrderValue", unit=MetricUnit.Count, value=result["total"])
    
    return result

@tracer.capture_method
def process_order(order: dict) -> dict:
    # Your logic here
    return {"status": "success", "total": order.get("total", 0)}
```

### CloudWatch Dashboard

```hcl
resource "aws_cloudwatch_dashboard" "main" {
  dashboard_name = "my-app-dashboard"
  
  dashboard_body = jsonencode({
    widgets = [
      {
        type   = "metric"
        x      = 0
        y      = 0
        width  = 12
        height = 6
        properties = {
          title  = "Lambda Invocations"
          region = "us-east-1"
          metrics = [
            ["AWS/Lambda", "Invocations", "FunctionName", "api-handler"],
            [".", "Errors", ".", "."],
            [".", "Throttles", ".", "."]
          ]
          period = 60
          stat   = "Sum"
        }
      },
      {
        type   = "metric"
        x      = 12
        y      = 0
        width  = 12
        height = 6
        properties = {
          title  = "API Gateway Latency"
          region = "us-east-1"
          metrics = [
            ["AWS/ApiGateway", "Latency", "ApiId", "abc123", { stat = "p50" }],
            ["...", { stat = "p90" }],
            ["...", { stat = "p99" }]
          ]
        }
      }
    ]
  })
}
```

---

## Related Resources

- [Docker](../docker/) - Container deployments
- [Data Pipelines](../../2-data-engineering/etl-pipelines/) - Glue + Lambda patterns
- [Event-Driven Systems](../../4-automation/event-driven/) - EventBridge patterns
