# Car Price Prediction with MLflow

## Overview

This project is a demonstration of how to use MLflow for machine learning experiment tracking and model management in both local and remote environments. It focuses on predicting car prices using various features such as mileage, year, fuel type, and more, leveraging machine learning models within a structured MLOps pipeline.

MLflow is used in two key configurations:

- Local Tracking (car_price_prediction_script.ipynb):
  Experiments are logged locally during development. MLflow captures model parameters, evaluation metrics, artifacts, and versioned models for reproducibility and comparison.

- Remote Tracking Server (app.py):
  MLflow is configured to connect to a remote tracking server and model registry. This enables centralized experiment logging, collaborative tracking, and production-ready model versioning.

This setup showcases how to:

- Seamlessly transition from local experimentation to remote production tracking.

- Maintain consistent and organized ML experimentation.

- Enable collaborative model development using MLflow’s web UI and model registry.

---

## Key Features

- **Data Preprocessing**: Cleans and transforms raw data for training.
- **Feature Engineering**: Encodes categorical variables and scales numerical features.
- **Model Training & Experiment Tracking**: Uses MLflow to track different model experiments.
- **Model Evaluation**: Compares models using RMSE, MAE, and R² scores.
- **Model Deployment**: Saves the best-performing model for future predictions.
- **Automated Logging**: MLflow logs parameters, metrics, and model artifacts.
- **Reproducibility**: Ensures consistent results through versioning.

---

## Tech Stack
- Scikit-learn
- Matplotlib
- Seaborn
- Pandas
- Mlflow
- logging

---

## Key Steps in the Project

### Data Collection & Preprocessing

- Load dataset and perform exploratory data analysis (EDA).
- Handle missing values and outliers.
- Encode categorical variables and normalize numerical features.

### Model Training & Experiment Tracking

- Train multiple machine learning models (e.g., **Linear Regression, Random Forest, Decision Tree Regression, Support Vector Regression**).
- Tune hyperparameters using RandomizedSearchCV.
- Log model parameters, metrics, and artifacts using **MLflow**.

### Model Evaluation & Selection

- Compare different models based on:
  - **Mean Absolute Error (MAE)**
  - **R² Score**
- Select the best model for deployment.

### Model Deployment & Inference

- Save the best model using **MLflow Model Registry**.
- Load the model for predictions.
- Deploy the model as a REST API (optional).

---

## Project Limitations

- The model's accuracy depends on the quality and diversity of training data.
- Limited feature set might not capture all real-world factors affecting car prices.
- The model does not consider external factors like **market demand, brand reputation, and economic conditions**.
- Hyperparameter tuning can be further optimized for better performance.
- Model deployment as a full-scale API is not included in this version.

---

## Conclusion

This project provides a complete **ML pipeline** for predicting car prices, utilizing **MLflow** for tracking and model management. Future improvements could include **more advanced feature engineering, deep learning models, and cloud-based deployment**.
