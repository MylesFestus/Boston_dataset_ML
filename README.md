# Boston Housing Price Prediction

An end-to-end Machine Learning project that predicts Boston housing prices based on neighborhood and property characteristics. The project demonstrates the complete ML lifecycle, including data preprocessing, model training, evaluation, deployment on Heroku, data validation, monitoring, and MLOps best practices.

---

---
### Live Demo [https://bostonhousingapp-3326ffdc7187.herokuapp.com]
---

# Project Overview

This project uses a supervised machine learning regression model to predict median housing prices in Boston. The application is deployed using Heroku and provides a user-friendly web interface for generating predictions from housing-related input features.

The project covers:

- Exploratory Data Analysis (EDA)
- Data preprocessing and feature engineering
- Model training and selection
- Model evaluation
- Flask web application development
- Heroku deployment
- Docker containerization
- Data validation and quality checks
- Model monitoring and drift detection
- Continuous evaluation and retraining strategies

---

# Problem Statement

Real estate prices are influenced by multiple factors such as crime rate, number of rooms, accessibility, tax rates, and socioeconomic conditions. The objective of this project is to build a machine learning model capable of predicting housing prices based on these attributes.

---

# Dataset Description

The Boston Housing Dataset contains information collected by the U.S. Census Service concerning housing in the Boston area.

### Features

| Feature | Description |
|----------|-------------|
| CRIM | Per capita crime rate by town |
| ZN | Proportion of residential land zoned |
| INDUS | Proportion of non-retail business acres |
| CHAS | Charles River dummy variable |
| NOX | Nitric oxide concentration |
| RM | Average number of rooms per dwelling |
| AGE | Proportion of owner-occupied units built before 1940 |
| DIS | Weighted distances to employment centers |
| RAD | Accessibility to radial highways |
| TAX | Property tax rate |
| PTRATIO | Pupil-teacher ratio |
| B | Proportion of Black population |
| LSTAT | Percentage of lower-status population |

### Target Variable

**MEDV** – Median value of owner-occupied homes (in $1000s)

---

# Project Structure

```text
Boston-Housing-Project/
│
├── app.py
├── model.pkl
├── scaler.pkl
├── requirements.txt
├── Procfile
├── runtime.txt
├── Dockerfile
├── docker-compose.yml
│
├── data/
│   └── housing.csv
│
├── notebooks/
│   └── EDA.ipynb
│
├── templates/
│   └── index.html
│
├── static/
│
├── validation/
│   ├── schema.py
│   └── validate.py
│
├── monitoring/
│   ├── drift_detection.py
│   └── metrics.py
│
└── README.md
```

---

# Exploratory Data Analysis

EDA was performed to understand:

- Feature distributions
- Missing values
- Correlation between variables
- Outliers
- Feature importance
- Target variable distribution

Visualizations included:

- Histograms
- Heatmaps
- Pair plots
- Box plots
- Correlation matrices

---

# Data Preprocessing

The following preprocessing steps were applied before training:

### Missing Value Handling

- Identified missing values
- Imputed numerical values where required

### Outlier Detection

Techniques used:

- IQR Method
- Z-score Analysis

### Feature Scaling

Applied:

- StandardScaler

### Train-Test Split

```python
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)
```

---

# Model Training

Several regression algorithms were evaluated:

### Models Tested

- Linear Regression
- Ridge Regression
- Lasso Regression
- Decision Tree Regressor
- Random Forest Regressor
- Gradient Boosting Regressor
- XGBoost Regressor

### Model Selection Criteria

The final model was selected based on:

- RMSE
- MAE
- R² Score
- Cross-validation performance

---

# Model Evaluation

### Evaluation Metrics

#### Mean Absolute Error (MAE)

Measures average prediction error.

#### Mean Squared Error (MSE)

Measures squared prediction error.

#### Root Mean Squared Error (RMSE)

Provides error magnitude in target units.

#### R² Score

Measures variance explained by the model.

### Example Results

| Metric | Score |
|----------|---------|
| MAE | 2.15 |
| RMSE | 3.12 |
| R² Score | 0.89 |

---

# Flask Web Application

The trained model is integrated into a Flask application.

### User Workflow

1. Enter housing parameters.
2. Submit the form.
3. Features are validated.
4. Model generates prediction.
5. Predicted house price is displayed.

---

# Deployment on Heroku

The application is deployed using Heroku for public accessibility.

## Prerequisites

- Python 3.10+
- Git
- Heroku CLI
- Heroku Account

### Create Heroku Application

```bash
heroku create boston-housing-app
```

### Login

```bash
heroku login
```

### Initialize Git Repository

```bash
git init
git add .
git commit -m "Initial deployment"
```

### Deploy

```bash
git push heroku main
```

### Scale Dyno

```bash
heroku ps:scale web=1
```

### Open Application

```bash
heroku open
```

---

# Procfile

```text
web: gunicorn app:app
```

---

# Runtime File

```text
python-3.10.13
```

---

# Local Setup

## Clone Repository

```bash
git clone https://github.com/yourusername/boston-housing-project.git

cd boston-housing-project
```

## Create Virtual Environment

### Linux/Mac

```bash
python -m venv venv

source venv/bin/activate
```

### Windows

```bash
venv\Scripts\activate
```

## Install Dependencies

```bash
pip install -r requirements.txt
```

## Run Application

```bash
python app.py
```

Application URL:

```text
http://localhost:5000
```

---

# Docker Containerization

Containerization ensures reproducibility and consistent deployments across environments.

## Dockerfile

```dockerfile
FROM python:3.10-slim

WORKDIR /app

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 5000

CMD ["gunicorn", "app:app"]
```

---

## Docker Compose

```yaml
version: "3.9"

services:
  housing-app:
    build: .
    container_name: housing-app
    ports:
      - "5000:5000"
    restart: always
```

---

## Build Docker Image

```bash
docker build -t boston-housing .
```

## Run Container

```bash
docker run -p 5000:5000 boston-housing
```

## Run with Docker Compose

```bash
docker-compose up --build
```

---

# Data Validation and Inspection

Data validation is critical to ensure incoming requests meet model expectations.

## Why Data Validation?

Without validation, incorrect data can lead to:

- Invalid predictions
- Application crashes
- Data drift
- Reduced model performance

---

## Validation Rules

### Schema Validation

Verify required columns and data types.

```python
schema = {
    "CRIM": float,
    "ZN": float,
    "INDUS": float,
    "RM": float,
    "LSTAT": float
}
```

---

### Missing Value Checks

```python
if dataframe.isnull().sum().sum() > 0:
    raise ValueError("Missing values detected")
```

---

### Range Validation

```python
if RM <= 0:
    raise ValueError("Room count must be positive")
```

---

### Duplicate Detection

```python
if dataframe.duplicated().sum() > 0:
    raise ValueError("Duplicate records found")
```

---

### Statistical Validation

Compare incoming production data against training data.

Checks include:

- Mean comparison
- Standard deviation comparison
- Distribution comparison

---

# Data Quality Tools

Recommended tools:

- Great Expectations
- Pandera
- Evidently AI
- Pydantic
- Deequ

---

# Model Monitoring After Deployment

A model's performance can degrade over time due to changes in real-world data.

Monitoring helps identify issues before they affect users.

---

## What to Monitor?

### System Metrics

Monitor:

- API response time
- Error rate
- CPU usage
- Memory utilization
- Request throughput

Tools:

- Prometheus
- Grafana
- New Relic

---

### Prediction Monitoring

Track:

- Prediction frequency
- Prediction distribution
- Prediction ranges
- Outlier predictions

Example:

```text
Prediction Logs

Timestamp
Input Features
Prediction
Response Time
Model Version
```

---

# Data Drift Detection

Data drift occurs when incoming data differs from training data.

### Common Drift Indicators

- Population Stability Index (PSI)
- Kolmogorov-Smirnov Test
- Jensen-Shannon Distance

### Drift Monitoring Tools

- Evidently AI
- WhyLabs
- Arize AI
- Fiddler AI

---

# Concept Drift Monitoring

Concept drift occurs when the relationship between features and target changes.

Example:

```text
Training Data Relationship:

RM → House Price

Production Data Relationship:

RM no longer strongly influences price
```

Monitor:

- Prediction errors
- Residual distributions
- Accuracy degradation

---

# Logging Strategy

Store prediction requests for analysis.

Recommended log fields:

```text
Timestamp
Model Version
Input Features
Prediction
Request ID
Response Time
```

Store logs in:

- PostgreSQL
- MongoDB
- Cloud Storage
- ELK Stack

---

# Model Evaluation After Deployment

Offline evaluation is not sufficient.

Production evaluation should be continuous.

---

## Evaluation Workflow

```text
User Request
      │
      ▼
Validation
      │
      ▼
Prediction
      │
      ▼
Prediction Logging
      │
      ▼
Ground Truth Collection
      │
      ▼
Performance Evaluation
      │
      ▼
Retraining Decision
```

---

## Metrics to Track

### Regression Metrics

- MAE
- RMSE
- MAPE
- R²

### Business Metrics

- User engagement
- Prediction usage
- Application reliability

---

# Model Retraining Strategy

Retraining should be triggered when:

- Significant data drift detected
- RMSE increases by more than 20%
- New data becomes available
- Business requirements change

---

# Model Versioning

Maintain version control for:

- Model artifacts
- Training datasets
- Hyperparameters
- Feature engineering logic

Recommended tools:

- MLflow
- DVC
- Weights & Biases

---

# MLOps Recommendations

To make the project production-ready:

### CI/CD Pipeline

```text
Code Commit
     │
     ▼
Unit Tests
     │
     ▼
Data Validation
     │
     ▼
Model Evaluation
     │
     ▼
Docker Build
     │
     ▼
Deployment
     │
     ▼
Monitoring
```

Tools:

- GitHub Actions
- Jenkins
- GitLab CI/CD

---

# Security Best Practices

- Validate all user inputs
- Use environment variables for secrets
- Enable HTTPS
- Restrict access to model artifacts
- Implement request rate limiting

---

# Future Enhancements

- Automated retraining pipeline
- Feature store integration
- Kubernetes deployment
- Explainable AI using SHAP
- Batch prediction API
- Real-time monitoring dashboard
- A/B testing framework
- Cloud-native deployment

---

# Technologies Used

### Programming

- Python

### Machine Learning

- Scikit-Learn
- NumPy
- Pandas

### Visualization

- Matplotlib
- Seaborn

### Web Framework

- Flask

### Deployment

- Heroku
- Gunicorn

### Containerization

- Docker
- Docker Compose

### Monitoring

- Prometheus
- Grafana

### MLOps

- MLflow
- DVC
- Evidently AI

---

# Results

The deployed model successfully predicts Boston housing prices using user-provided features and demonstrates a complete machine learning deployment pipeline from data preprocessing to production monitoring.

---

# License

This project is licensed under the MIT License.

---

# Author

Your Name

Machine Learning Engineer | Data Scientist

GitHub: https://github.com/MylesFestus

LinkedIn: www.linkedin.com/in/festus-attor
