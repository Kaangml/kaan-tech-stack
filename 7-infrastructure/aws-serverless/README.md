# AWS Serverless

Lambda, API Gateway, S3, and AWS Glue for serverless architectures.

## AWS Lambda

### Basic Handler

```python
import json
import logging

logger = logging.getLogger()
logger.setLevel(logging.INFO)

def handler(event, context):
    logger.info(f"Event: {json.dumps(event)}")
    
    try:
        # Your logic here
        result = process(event)
        
        return {
            "statusCode": 200,
            "headers": {
                "Content-Type": "application/json",
                "Access-Control-Allow-Origin": "*"
            },
            "body": json.dumps({"data": result})
        }
    except ValueError as e:
        return {
            "statusCode": 400,
            "body": json.dumps({"error": str(e)})
        }
    except Exception as e:
        logger.error(f"Error: {str(e)}")
        return {
            "statusCode": 500,
            "body": json.dumps({"error": "Internal server error"})
        }
```

### API Gateway Integration

```python
def api_handler(event, context):
    # Path parameters
    path_params = event.get("pathParameters", {})
    user_id = path_params.get("userId")
    
    # Query parameters
    query_params = event.get("queryStringParameters", {}) or {}
    limit = int(query_params.get("limit", 10))
    
    # Body (POST/PUT)
    body = json.loads(event.get("body", "{}"))
    
    # HTTP method
    method = event.get("httpMethod")
    
    # Headers
    headers = event.get("headers", {})
    auth_token = headers.get("Authorization")
    
    return {
        "statusCode": 200,
        "body": json.dumps({"userId": user_id, "limit": limit})
    }
```

### Async Lambda (SQS Trigger)

```python
def sqs_handler(event, context):
    failed_messages = []
    
    for record in event["Records"]:
        try:
            message_body = json.loads(record["body"])
            process_message(message_body)
        except Exception as e:
            logger.error(f"Failed to process message: {e}")
            failed_messages.append({
                "itemIdentifier": record["messageId"]
            })
    
    # Partial batch response
    return {
        "batchItemFailures": failed_messages
    }
```

### Layers and Dependencies

```bash
# Create layer
mkdir -p python/lib/python3.11/site-packages
pip install requests -t python/lib/python3.11/site-packages/
zip -r layer.zip python

# Deploy layer
aws lambda publish-layer-version \
    --layer-name my-dependencies \
    --zip-file fileb://layer.zip \
    --compatible-runtimes python3.11
```

## Boto3

### S3 Operations

```python
import boto3
from botocore.exceptions import ClientError

s3 = boto3.client('s3')

# Upload
def upload_file(file_path: str, bucket: str, key: str):
    s3.upload_file(file_path, bucket, key)

# Upload with metadata
def upload_with_metadata(data: bytes, bucket: str, key: str):
    s3.put_object(
        Bucket=bucket,
        Key=key,
        Body=data,
        ContentType='application/json',
        Metadata={'processed': 'true'}
    )

# Download
def download_file(bucket: str, key: str, file_path: str):
    s3.download_file(bucket, key, file_path)

# Read directly
def read_s3_file(bucket: str, key: str) -> str:
    response = s3.get_object(Bucket=bucket, Key=key)
    return response['Body'].read().decode('utf-8')

# List objects
def list_files(bucket: str, prefix: str = ""):
    paginator = s3.get_paginator('list_objects_v2')
    for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
        for obj in page.get('Contents', []):
            yield obj['Key']

# Generate presigned URL
def get_presigned_url(bucket: str, key: str, expiration: int = 3600):
    return s3.generate_presigned_url(
        'get_object',
        Params={'Bucket': bucket, 'Key': key},
        ExpiresIn=expiration
    )
```

### DynamoDB

```python
import boto3
from boto3.dynamodb.conditions import Key, Attr

dynamodb = boto3.resource('dynamodb')
table = dynamodb.Table('users')

# Put item
def create_user(user_id: str, name: str, email: str):
    table.put_item(
        Item={
            'userId': user_id,
            'name': name,
            'email': email,
            'createdAt': datetime.now().isoformat()
        }
    )

# Get item
def get_user(user_id: str):
    response = table.get_item(Key={'userId': user_id})
    return response.get('Item')

# Query (requires partition key)
def get_user_orders(user_id: str):
    orders_table = dynamodb.Table('orders')
    response = orders_table.query(
        KeyConditionExpression=Key('userId').eq(user_id),
        FilterExpression=Attr('status').eq('completed')
    )
    return response['Items']

# Scan (expensive, avoid in production)
def find_users_by_email(email: str):
    response = table.scan(
        FilterExpression=Attr('email').eq(email)
    )
    return response['Items']

# Update
def update_user(user_id: str, updates: dict):
    update_expr = "SET " + ", ".join(f"#{k} = :{k}" for k in updates)
    expr_names = {f"#{k}": k for k in updates}
    expr_values = {f":{k}": v for k, v in updates.items()}
    
    table.update_item(
        Key={'userId': user_id},
        UpdateExpression=update_expr,
        ExpressionAttributeNames=expr_names,
        ExpressionAttributeValues=expr_values
    )

# Batch write
def batch_create_users(users: list):
    with table.batch_writer() as batch:
        for user in users:
            batch.put_item(Item=user)
```

### SQS

```python
sqs = boto3.client('sqs')
queue_url = "https://sqs.region.amazonaws.com/account/queue-name"

# Send message
def send_message(message: dict):
    sqs.send_message(
        QueueUrl=queue_url,
        MessageBody=json.dumps(message),
        MessageAttributes={
            'Type': {
                'DataType': 'String',
                'StringValue': 'order'
            }
        }
    )

# Send batch
def send_messages(messages: list):
    entries = [
        {
            'Id': str(i),
            'MessageBody': json.dumps(msg)
        }
        for i, msg in enumerate(messages)
    ]
    sqs.send_message_batch(QueueUrl=queue_url, Entries=entries)

# Receive and process
def poll_messages():
    response = sqs.receive_message(
        QueueUrl=queue_url,
        MaxNumberOfMessages=10,
        WaitTimeSeconds=20  # Long polling
    )
    
    for message in response.get('Messages', []):
        try:
            process(json.loads(message['Body']))
            sqs.delete_message(
                QueueUrl=queue_url,
                ReceiptHandle=message['ReceiptHandle']
            )
        except Exception as e:
            logger.error(f"Failed: {e}")
```

## AWS Glue

### Glue Job (PySpark)

```python
import sys
from awsglue.transforms import *
from awsglue.utils import getResolvedOptions
from pyspark.context import SparkContext
from awsglue.context import GlueContext
from awsglue.job import Job

args = getResolvedOptions(sys.argv, ['JOB_NAME', 'input_path', 'output_path'])

sc = SparkContext()
glueContext = GlueContext(sc)
spark = glueContext.spark_session
job = Job(glueContext)
job.init(args['JOB_NAME'], args)

# Read from S3
datasource = glueContext.create_dynamic_frame.from_options(
    connection_type="s3",
    connection_options={"paths": [args['input_path']]},
    format="json"
)

# Transform
transformed = datasource.toDF()
transformed = transformed.filter(transformed['status'] == 'active')
transformed = transformed.withColumn('processed_at', current_timestamp())

# Write to S3
glueContext.write_dynamic_frame.from_options(
    frame=DynamicFrame.fromDF(transformed, glueContext, "output"),
    connection_type="s3",
    connection_options={"path": args['output_path']},
    format="parquet"
)

job.commit()
```

### Glue Crawler (Terraform)

```hcl
resource "aws_glue_crawler" "data_crawler" {
  database_name = aws_glue_catalog_database.database.name
  name          = "data-crawler"
  role          = aws_iam_role.glue_role.arn

  s3_target {
    path = "s3://my-bucket/data/"
  }

  schedule = "cron(0 * * * ? *)"  # Hourly

  schema_change_policy {
    delete_behavior = "LOG"
    update_behavior = "UPDATE_IN_DATABASE"
  }
}
```

## Serverless Framework

### serverless.yml

```yaml
service: my-api

provider:
  name: aws
  runtime: python3.11
  region: us-east-1
  memorySize: 256
  timeout: 30
  environment:
    TABLE_NAME: ${self:service}-${self:provider.stage}
  iam:
    role:
      statements:
        - Effect: Allow
          Action:
            - dynamodb:Query
            - dynamodb:GetItem
            - dynamodb:PutItem
          Resource: !GetAtt UsersTable.Arn

functions:
  api:
    handler: src/handler.main
    events:
      - httpApi:
          path: /users
          method: GET
      - httpApi:
          path: /users/{id}
          method: GET
      - httpApi:
          path: /users
          method: POST

  worker:
    handler: src/worker.process
    events:
      - sqs:
          arn: !GetAtt ProcessingQueue.Arn
          batchSize: 10

resources:
  Resources:
    UsersTable:
      Type: AWS::DynamoDB::Table
      Properties:
        TableName: ${self:provider.environment.TABLE_NAME}
        BillingMode: PAY_PER_REQUEST
        AttributeDefinitions:
          - AttributeName: userId
            AttributeType: S
        KeySchema:
          - AttributeName: userId
            KeyType: HASH

    ProcessingQueue:
      Type: AWS::SQS::Queue
      Properties:
        QueueName: ${self:service}-processing

plugins:
  - serverless-python-requirements

custom:
  pythonRequirements:
    dockerizePip: true
    layer: true
```

## Cost Optimization

### Lambda

```python
# Reuse connections outside handler
import boto3

# Initialized once, reused across invocations
s3 = boto3.client('s3')
dynamodb = boto3.resource('dynamodb')

def handler(event, context):
    # Use the pre-initialized clients
    pass
```

### Provisioned Concurrency

```yaml
# serverless.yml
functions:
  api:
    handler: src/handler.main
    provisionedConcurrency: 5  # Keep 5 instances warm
```

## Related Resources

- [Docker](../docker/README.md) - Container-based deployment
- [ETL Pipelines](../../2-data-engineering/etl-pipelines/README.md) - Data pipeline patterns
- [Scalable Architectures](../../99-blueprints/README.md) - Production patterns
