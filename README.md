BOSTON HOUSING DATASET

Machine Learning Analysis -  Project Overview

Predicting Median Home Values Using Regression Models

- Dataset    :	Boston Housing CSV
- Samples    :	506 rows, 12 features
- Task       :	Supervised Regression
- Models     :	5 regression algorithms


Project Description

This project applies a comprehensive machine learning pipeline to the Boston Housing dataset, with the objective of predicting the median value of owner-occupied homes (MEDV, expressed in thousands of U.S. dollars). Five regression algorithms of varying complexity were trained, evaluated, and compared — ranging from classical linear models to modern ensemble methods.


1.1 Objectives
•	Perform exploratory data analysis (EDA) to understand feature distributions and relationships.
•	Preprocess the dataset through feature scaling and train-test splitting.
•	Train and evaluate five regression models: Linear Regression, Ridge, Lasso, Random Forest, and Gradient Boosting.
•	Compare model performance using R², RMSE, MAE, and cross-validation scores.
•	Identify the most influential features driving home price predictions.
•	Produce actionable insights and visualisations to communicate findings.

1.2 Target Variable
The target variable is MEDV — the median value of owner-occupied homes in $1,000s. It ranges from $5,000 to $50,000 with a mean of approximately $22,530, reflecting a right-skewed distribution with a concentration of mid-range home values.

2.1 Descriptive Statistics
Key statistical summaries of the dataset reveal substantial variation across features. CRIM (crime rate) ranges from near zero to 88.98, indicating the presence of extreme outliers in high-crime areas. MEDV spans from $5,000 to $50,000 with a standard deviation of $9,200. LSTAT has a mean of 12.65% with a range of 1.73 to 37.97, reflecting the diversity of socioeconomic conditions across the 506 tracts.

2.2 Correlations with MEDV
Pearson correlation analysis revealed the following key relationships with the target variable:

•	LSTAT (r = -0.738): The strongest negative correlate — higher proportions of lower-status population strongly suppress home values.
•	RM (r = +0.695): The strongest positive correlate — more rooms per dwelling strongly increases home values.
•	PTRATIO (r = -0.508): Higher pupil-teacher ratios, a proxy for lower school quality, reduce home values.
•	INDUS (r = -0.484) and TAX (r = -0.469): Industrial land use and high tax rates both suppress home values.
•	NOX (r = -0.427): Air pollution has a meaningful negative association with property prices.

3.1 Data Preprocessing
The dataset was first checked for missing values — none were found. The data was then split into training (80%, 404 samples) and test (20%, 102 samples) sets using a fixed random seed (42) to ensure reproducibility. Feature scaling was applied using StandardScaler, which standardises each feature to zero mean and unit variance. Critically, the scaler was fitted exclusively on the training set and applied to both sets, preventing data leakage from the test set into model training.

3.2 Model Selection
Five regression algorithms were selected to represent a spectrum of model complexity and interpretability:

•	Linear Regression: The baseline model. Assumes a linear relationship between features and the target. Highly interpretable but limited in capturing non-linear patterns.
•	Ridge Regression (L2): Extends linear regression with L2 regularisation (alpha = 1.0), penalising large coefficients to reduce overfitting.
•	Lasso Regression (L1): Uses L1 regularisation (alpha = 0.1), which can shrink some coefficients to zero, effectively performing feature selection.
•	Random Forest: An ensemble of 100 decision trees using bootstrap aggregation (bagging). Captures non-linear relationships and feature interactions robustly.
•	Gradient Boosting: An ensemble of 100 sequentially built decision trees, each correcting errors of the previous. Generally the most powerful model for structured tabular data.

3.3 Evaluation Metrics
Each model was evaluated on the held-out test set using three complementary metrics:

•	R² (Coefficient of Determination): Proportion of variance in MEDV explained by the model. Ranges from 0 to 1; higher is better.
•	RMSE (Root Mean Squared Error): Square root of mean squared prediction error, in the same units as MEDV ($000s). Penalises large errors more heavily.
•	MAE (Mean Absolute Error): Average absolute prediction error ($000s). Robust to outliers.

Additionally, 5-fold cross-validation on the training set was used to assess generalisation stability and detect overfitting.

4. Results
4.1 Model Performance Comparison

Model	                    R²	          RMSE	      MAE
Gradient Boosting      	0.9150	     2.496	       1.905
Random Forest          	0.8908	     2.830	       2.035
Linear Regression	      0.6894	     4.773         3.111
Ridge Regression	      0.6889	     4.776	       3.107
Lasso Regression	      0.6671	     4.941	       3.168

4.2 Cross-Validation Results
Five-fold cross-validation on the training set confirmed the generalisation of each model. Gradient Boosting achieved a mean CV R² of 0.845 ± 0.056, and Random Forest achieved 0.827 ± 0.050 — both with low standard deviations indicating stability across folds. Linear models achieved CV R² of approximately 0.71 ± 0.063, consistent with their test set performance.
4.3 Feature Importance
Feature importance was extracted from both tree-based models. The results were consistent across Random Forest and Gradient Boosting:

•	RM (Avg. rooms): 50.5% importance — the dominant predictor. Each additional room is strongly associated with higher home values.
•	LSTAT (Lower status %): 31.3% importance — the second most important feature. Economic deprivation in a neighbourhood has a strong suppressive effect on home prices.
•	DIS (Distance to employment): 6.1% importance — modest positive effect on home value.
•	CRIM (Crime rate): 4.1% importance — crime rates have a measurable negative influence.
•	PTRATIO, TAX, NOX, AGE: Each contribute less than 2% individually but collectively capture neighbourhood quality signals.

Together, RM and LSTAT alone account for over 81% of the predictive power of the Random Forest model — a remarkable concentration of explanatory power in just two features

5. Discussion
The results confirm that ensemble tree methods substantially outperform linear models on the Boston Housing dataset. The gap — approximately 0.22 in R² between Gradient Boosting and linear regression — is attributable to non-linear interactions between features. For example, the relationship between LSTAT and MEDV is non-linear: the suppressive effect of low-status populations accelerates at higher concentrations. Linear models assume a constant slope across the entire range of LSTAT, which underestimates this curvature.

The near-identical performance of Linear Regression and Ridge Regression suggests that L2 regularisation provides negligible benefit here. With only 12 features and 404 training samples, the problem is not high-dimensional enough for Ridge to offer meaningful regularisation gains. Lasso performs slightly worse because its feature elimination (setting some coefficients to zero) discards information that, while small, still contributes to predictive accuracy.

The dominance of RM and LSTAT as predictors is consistent with economic theory: property values are driven primarily by the physical characteristics of the dwelling (size, as proxied by rooms) and the socioeconomic composition of the neighbourhood. Environmental factors (NOX, CRIM) and accessibility (DIS, RAD) play secondary roles.

Cross-validation results are closely aligned with test set results for all models, indicating no overfitting and reliable generalisation. The Gradient Boosting model is recommended for deployment due to its superior accuracy, stable cross-validation performance, and ability to produce feature importances for interpretability.

6. Conclusion
This project successfully applied a full machine learning pipeline — from exploratory data analysis through model evaluation — to the Boston Housing dataset. Five regression models were trained and compared. The key conclusions are:

•	Gradient Boosting is the best-performing model (R² = 0.915, RMSE = $2,496), explaining over 91% of the variance in median home values on unseen data.
•	Non-linear ensemble models (Random Forest, Gradient Boosting) substantially outperform linear models, demonstrating the non-linear nature of the underlying feature-price relationships.
•	RM and LSTAT are the two dominant predictors, collectively accounting for over 81% of feature importance in tree-based models.
•	The dataset is clean, well-structured, and requires no imputation — making it an ideal benchmark for regression algorithm comparison.
•	All models generalise well, as evidenced by close alignment between test set metrics and 5-fold cross-validation scores.

For practitioners, the findings suggest that when predicting residential property values, the number of rooms and neighbourhood socioeconomic status should be prioritised as primary inputs. Environmental and accessibility variables, while statistically significant, contribute marginally to predictive accuracy and may be omitted in resource-constrained settings without major loss of performance.


7. Executive Summary
• Dataset	Boston Housing                                — 506 samples, 12 features, no missing values
• Target	                                              - MEDV: Median home value in $000s (range: $5k – $50k, mean: $22.5k)
• Best Model	                                          - Gradient Boosting — R² = 0.915, RMSE = $2,496, MAE = $1,905
• Runner-up                                             -	Random Forest — R² = 0.891, RMSE = $2,830
• Linear Models	                                        - Linear / Ridge / Lasso — R² ≈ 0.67–0.69 (limited by non-linearity)
• Top Feature	                                          - RM (avg. rooms per dwelling) — 50.5% of feature importance
• 2nd Feature                                           - LSTAT (lower status population %) — 31.3% of feature importance
• CV Stability                                          -	All models show low variance across 5-fold CV — no overfitting detected
• Recommendation                                        -	Deploy Gradient Boosting for maximum accuracy; use Random Forest for interpretability
• Outputs	                                              - 8 publication-quality plots + Python script + this project report


