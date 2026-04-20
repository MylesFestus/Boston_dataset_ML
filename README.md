# 🏡 Boston Housing Dataset

## Machine Learning Analysis Project

### Predicting Median Home Values Using Regression Models

---

## 📊 Project Overview

This project applies a comprehensive machine learning pipeline to the Boston Housing dataset to predict the median value of owner-occupied homes (MEDV, in $1,000s). Five regression models were trained, evaluated, and compared — ranging from linear models to advanced ensemble methods.

---

## 📁 Dataset Summary

- **Dataset:** Boston Housing CSV  
- **Samples:** 506 rows  
- **Features:** 12 input variables  
- **Task:** Supervised Regression  
- **Target:** MEDV (Median home value in $1,000s)

---

## 🎯 Objectives

- Perform Exploratory Data Analysis (EDA)
- Data Preprocessing (scaling + train-test split)
- Train 5 regression models:
  - Linear Regression
  - Ridge Regression
  - Lasso Regression
  - Random Forest
  - Gradient Boosting
- Evaluate models using:
  - R²
  - RMSE
  - MAE
  - Cross-validation
- Identify key predictive features
- Generate insights and visualizations

---

## 📌 Target Variable (MEDV)

- Range: $5,000 – $50,000  
- Mean: ~$22,530  
- Distribution: Right-skewed  
- Interpretation: Median value of owner-occupied homes

---

## 📊 Exploratory Data Analysis

### Key Statistics
- **CRIM:** Wide range with extreme outliers (up to 88.98)
- **LSTAT:** 1.73% – 37.97% (socioeconomic variation)
- **MEDV:** Std dev ≈ $9,200

### Correlation with MEDV

- **LSTAT:** -0.738 (strong negative)
- **RM:** +0.695 (strong positive)
- **PTRATIO:** -0.508
- **INDUS:** -0.484
- **TAX:** -0.469
- **NOX:** -0.427

---

## ⚙️ Data Preprocessing

- No missing values
- Train-test split: 80/20 (404 / 102 samples)
- Feature scaling: StandardScaler
- Prevented data leakage (fit only on training data)

---

## 🤖 Models Used

### Linear Models
- Linear Regression (baseline)
- Ridge Regression (L2 regularization)
- Lasso Regression (L1 feature selection)

### Ensemble Models
- Random Forest (100 trees, bagging)
- Gradient Boosting (100 trees, boosting)

---

## 📏 Evaluation Metrics

- **R²:** Explained variance
- **RMSE:** Penalizes large errors
- **MAE:** Average absolute error
- **Cross-validation:** 5-fold CV for stability

---

## 📈 Results

### Model Performance Comparison

| Model              | R²     | RMSE  | MAE   |
|-------------------|--------|-------|-------|
| Gradient Boosting | 0.9150 | 2.496 | 1.905 |
| Random Forest     | 0.8908 | 2.830 | 2.035 |
| Linear Regression | 0.6894 | 4.773 | 3.111 |
| Ridge Regression  | 0.6889 | 4.776 | 3.107 |
| Lasso Regression  | 0.6671 | 4.941 | 3.168 |

---

## 🔁 Cross-Validation Results

- Gradient Boosting: **0.845 ± 0.056**
- Random Forest: **0.827 ± 0.050**
- Linear Models: **~0.71 ± 0.063**

✔ No overfitting detected  
✔ Stable model performance across folds  

---

## 🔍 Feature Importance

Top predictors:

- **RM (Rooms):** 50.5%
- **LSTAT (Lower status %):** 31.3%
- **DIS:** 6.1%
- **CRIM:** 4.1%
- Others (PTRATIO, TAX, NOX, AGE): <2% each

👉 RM + LSTAT account for **81%+ of predictive power**

---

## 🧠 Key Insights

- Ensemble models significantly outperform linear models
- Relationship between features and price is highly non-linear
- Socioeconomic status (LSTAT) strongly suppresses home value
- Number of rooms (RM) is the strongest positive driver
- Ridge/Lasso offer minimal benefit due to low dimensionality

---

## 📌 Conclusion

- 🏆 Best model: **Gradient Boosting**
- 📊 Performance: **R² = 0.915**
- 💡 Key predictors: RM & LSTAT
- ⚖️ Linear models are limited by non-linearity
- ✔ Models generalize well (validated via CV)

---

## 🚀 Recommendation

Use **Gradient Boosting** for deployment due to:
- Highest accuracy
- Stable cross-validation performance
- Strong interpretability via feature importance

---

## 📦 Project Deliverables

- ✔ Trained ML models  
- ✔ Evaluation metrics  
- ✔ Feature importance analysis  
- ✔ 8 publication-quality plots  
- ✔ Full Python implementation  
- ✔ This report  

---

## 📌 Executive Summary

- **Dataset:** 506 samples, 12 features, no missing values  
- **Best Model:** Gradient Boosting (R² = 0.915, RMSE = $2,496)  
- **Top Feature:** RM (50.5%)  
- **Second Feature:** LSTAT (31.3%)  
- **Best Approach:** Ensemble learning  
- **Outcome:** Strong predictive performance with stable generalization  

---
