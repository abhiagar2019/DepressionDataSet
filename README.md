# DepressionDataSet
Predicting 'History_of_Mental_Illness' from the available data.

## Overview
This project focuses on building and evaluating binary classification models using XGBoost, LightGBM, and Logistic Regression. The primary goal is to predict the target based on a set of features. The project includes comprehensive exploratory data analysis (EDA), model training, hyperparameter tuning, and evaluation.

## Table of Contents
- [1. Exploratory Data Analysis (EDA)](#1-exploratory-data-analysis-eda)
- [2. Models](#2-models)
- [3. Hyperparameter Tuning](#3-hyperparameter-tuning)
- [4. Model Evaluation](#4-model-evaluation)
- [5. Conclusion](#5-conclusion)

## 1. Exploratory Data Analysis (EDA)

### 1.1 Data Cleaning
- **Missing Values**: Identified and handled missing values 
- **Null & Outlier Detection**: Detected and addressed outliers 

### 1.2 Data Visualization
- **Univariate Analysis**: Visualized the distribution of individual features using histograms and box plots to understand their characteristics.
- **Bivariate Analysis**: Used scatter plots and correlation matrices to examine relationships between features and the target variable.
- **Categorical Analysis**: Analyzed categorical features using bar plots to observe class distributions and potential relationships with the target variable (Chi-square and Cramer's V test).

### 1.3 Feature Selection
- Performed feature selection using [techniques, e.g., correlation analysis, WoE/IV and XGBoost Feature Importance] to identify the most significant predictors for the target variable.

### 1.4 Addressing Class Imbalance
- Used oversampling techniques such as SMOTE to address class imbalance in the dataset. This ensures the model learns effectively from minority classes without overfitting.

## 2. Models

### 2.1 Models Implemented
- **XGBoost**: A powerful gradient boosting algorithm known for its speed and performance.
- **LightGBM**: An efficient gradient boosting framework that uses tree-based learning algorithms.
- **Logistic Regression**: A simple yet effective linear model for binary classification.

## 3. Hyperparameter Tuning

- **Grid SearchCV**: Conducted a grid search with cross-validation to optimize hyperparameters for each model, including:
  - **XGBoost**: Tuned parameters such as learning rate, max depth, and number of estimators.
  - **LightGBM**: Adjusted parameters like num_leaves, learning rate, and boosting type.
  - **Logistic Regression**: Optimized regularization strength (C) and solver method.
  
- The hyperparameter tuning process helped improve the model's performance by finding the best combination of parameters that minimize the loss function.

## 4. Model Evaluation

### 4.1 Evaluation Metrics
- **AUC-ROC Curve**: Plotted the AUC-ROC curve for each model to assess its ability to distinguish between classes. The Area Under the Curve (AUC) provides a single measure of overall model performance.
- **Precision-Recall Curve**: Analyzed precision and recall to evaluate the model's performance, especially under class imbalance. This curve helps understand the trade-off between precision (positive predictive value) and recall (sensitivity).

### 4.2 Results
- **Model Performance**: Summarized the performance of each model based on AUC-ROC and precision-recall scores.
- **Best Model Selection**: Chose the best-performing model based on evaluation metrics and cross-validation results.

