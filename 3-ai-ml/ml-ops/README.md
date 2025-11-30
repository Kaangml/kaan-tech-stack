# ML-Ops

Machine learning frameworks, model training, and experiment management.

## Framework Comparison

| Framework | Best For | Learning Curve | Production Ready |
|-----------|----------|----------------|------------------|
| scikit-learn | Classical ML, prototyping | Easy | Yes |
| PyCaret | AutoML, rapid experimentation | Very Easy | Moderate |
| PyTorch | Research, custom architectures | Medium | Yes |
| TensorFlow/Keras | Production, deployment | Medium | Excellent |
| XGBoost/LightGBM | Tabular data, competitions | Easy | Yes |

## scikit-learn

### Standard ML Pipeline

```python
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score

# Preprocessing
numeric_features = ['age', 'salary', 'experience']
categorical_features = ['department', 'role']

preprocessor = ColumnTransformer(
    transformers=[
        ('num', Pipeline([
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler())
        ]), numeric_features),
        ('cat', Pipeline([
            ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
            ('onehot', OneHotEncoder(handle_unknown='ignore'))
        ]), categorical_features)
    ]
)

# Full pipeline
pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('classifier', RandomForestClassifier(n_estimators=100, random_state=42))
])

# Cross-validation
scores = cross_val_score(pipeline, X, y, cv=5, scoring='accuracy')
print(f"Accuracy: {scores.mean():.3f} (+/- {scores.std():.3f})")
```

### Hyperparameter Tuning

```python
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import randint, uniform

param_distributions = {
    'classifier__n_estimators': randint(50, 500),
    'classifier__max_depth': randint(3, 20),
    'classifier__min_samples_split': randint(2, 20),
    'classifier__min_samples_leaf': randint(1, 10)
}

search = RandomizedSearchCV(
    pipeline,
    param_distributions,
    n_iter=100,
    cv=5,
    scoring='f1_weighted',
    random_state=42,
    n_jobs=-1
)

search.fit(X_train, y_train)
print(f"Best params: {search.best_params_}")
print(f"Best score: {search.best_score_:.3f}")
```

## PyCaret

### AutoML Workflow

```python
from pycaret.classification import *

# Initialize
clf = setup(
    data=df,
    target='target',
    session_id=42,
    normalize=True,
    transformation=True,
    remove_outliers=True,
    feature_selection=True,
    n_jobs=-1
)

# Compare all models
best_models = compare_models(n_select=3, sort='AUC')

# Tune best model
tuned_model = tune_model(best_models[0], optimize='AUC')

# Ensemble
blended = blend_models(best_models)
stacked = stack_models(best_models)

# Final model
final_model = finalize_model(tuned_model)

# Save
save_model(final_model, 'production_model')
```

### Regression with PyCaret

```python
from pycaret.regression import *

reg = setup(data=df, target='price', session_id=42)

# Quick comparison
compare_models(sort='RMSE')

# Create specific model
model = create_model('lightgbm')

# Interpret
interpret_model(model)  # SHAP values
plot_model(model, plot='feature')  # Feature importance
plot_model(model, plot='residuals')  # Residual analysis
```

## PyTorch

### Custom Model Architecture

```python
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset

class CustomModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, output_dim)
        )
    
    def forward(self, x):
        return self.layers(x)

# Training loop
def train_model(model, train_loader, val_loader, epochs=100):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-2)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs)
    
    best_val_loss = float('inf')
    
    for epoch in range(epochs):
        # Training
        model.train()
        train_loss = 0
        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            
            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
        
        # Validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                outputs = model(batch_x)
                val_loss += criterion(outputs, batch_y).item()
        
        scheduler.step()
        
        # Early stopping / checkpointing
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), 'best_model.pt')
        
        print(f"Epoch {epoch}: Train Loss={train_loss:.4f}, Val Loss={val_loss:.4f}")
    
    return model
```

### PyTorch Lightning (Cleaner Training)

```python
import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint

class LitModel(pl.LightningModule):
    def __init__(self, input_dim, hidden_dim, output_dim, lr=1e-3):
        super().__init__()
        self.save_hyperparameters()
        self.model = CustomModel(input_dim, hidden_dim, output_dim)
        self.criterion = nn.CrossEntropyLoss()
    
    def forward(self, x):
        return self.model(x)
    
    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.criterion(logits, y)
        self.log('train_loss', loss)
        return loss
    
    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.criterion(logits, y)
        acc = (logits.argmax(dim=1) == y).float().mean()
        self.log('val_loss', loss)
        self.log('val_acc', acc)
    
    def configure_optimizers(self):
        return optim.AdamW(self.parameters(), lr=self.hparams.lr)

# Training
trainer = pl.Trainer(
    max_epochs=100,
    callbacks=[
        EarlyStopping(monitor='val_loss', patience=10),
        ModelCheckpoint(monitor='val_loss', save_top_k=1)
    ],
    accelerator='auto'
)

trainer.fit(model, train_loader, val_loader)
```

## TensorFlow / Keras

### Keras Sequential API

```python
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

model = keras.Sequential([
    layers.Dense(256, activation='relu', input_shape=(input_dim,)),
    layers.BatchNormalization(),
    layers.Dropout(0.3),
    layers.Dense(128, activation='relu'),
    layers.BatchNormalization(),
    layers.Dropout(0.3),
    layers.Dense(num_classes, activation='softmax')
])

model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=1e-3),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# Callbacks
callbacks = [
    keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True),
    keras.callbacks.ReduceLROnPlateau(factor=0.5, patience=5),
    keras.callbacks.ModelCheckpoint('best_model.keras', save_best_only=True)
]

history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=100,
    batch_size=32,
    callbacks=callbacks
)
```

### Keras Functional API (Complex Architectures)

```python
# Multi-input model
text_input = keras.Input(shape=(max_length,), name='text')
numeric_input = keras.Input(shape=(num_features,), name='numeric')

# Text branch
x1 = layers.Embedding(vocab_size, 128)(text_input)
x1 = layers.LSTM(64)(x1)

# Numeric branch
x2 = layers.Dense(32, activation='relu')(numeric_input)

# Merge
merged = layers.concatenate([x1, x2])
output = layers.Dense(1, activation='sigmoid')(merged)

model = keras.Model(
    inputs=[text_input, numeric_input],
    outputs=output
)
```

## Experiment Tracking

### MLflow

```python
import mlflow
import mlflow.sklearn

mlflow.set_tracking_uri("http://localhost:5000")
mlflow.set_experiment("classification-experiment")

with mlflow.start_run():
    # Log parameters
    mlflow.log_params({
        "n_estimators": 100,
        "max_depth": 10,
        "learning_rate": 0.1
    })
    
    # Train model
    model.fit(X_train, y_train)
    
    # Log metrics
    predictions = model.predict(X_test)
    mlflow.log_metrics({
        "accuracy": accuracy_score(y_test, predictions),
        "f1_score": f1_score(y_test, predictions, average='weighted')
    })
    
    # Log model
    mlflow.sklearn.log_model(model, "model")
    
    # Log artifacts
    mlflow.log_artifact("confusion_matrix.png")
```

### Weights & Biases

```python
import wandb

wandb.init(project="my-project", config={
    "learning_rate": 1e-3,
    "epochs": 100,
    "batch_size": 32
})

for epoch in range(epochs):
    train_loss = train_epoch(model, train_loader)
    val_loss, val_acc = validate(model, val_loader)
    
    wandb.log({
        "train_loss": train_loss,
        "val_loss": val_loss,
        "val_accuracy": val_acc
    })

wandb.finish()
```

## Model Serialization

### ONNX Export

```python
import torch.onnx

# PyTorch to ONNX
dummy_input = torch.randn(1, input_dim)
torch.onnx.export(
    model,
    dummy_input,
    "model.onnx",
    input_names=['input'],
    output_names=['output'],
    dynamic_axes={'input': {0: 'batch_size'}}
)

# Inference with ONNX Runtime
import onnxruntime as ort

session = ort.InferenceSession("model.onnx")
outputs = session.run(None, {"input": input_data.numpy()})
```

### TorchScript

```python
# Scripting (preserves control flow)
scripted_model = torch.jit.script(model)
scripted_model.save("model_scripted.pt")

# Tracing (captures execution)
traced_model = torch.jit.trace(model, example_input)
traced_model.save("model_traced.pt")

# Load and inference
loaded_model = torch.jit.load("model_scripted.pt")
output = loaded_model(input_tensor)
```

## Related Resources

- [Computer Vision](../computer-vision/README.md) - Vision model architectures
- [LLM Agents](../llm-agents/README.md) - LLM fine-tuning and deployment
- [Docker](../../7-infrastructure/docker/README.md) - Containerizing ML models
