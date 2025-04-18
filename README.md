# Car Price Prediction with MLflow

## Overview
This project aims to predict car prices based on various features such as mileage, year, fuel type, and more. It leverages **machine learning models** and **MLflow** for experiment tracking, model versioning, and deployment. The project follows a structured **MLOps pipeline**, ensuring model reproducibility and efficient tracking of different experiment runs.

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

## Key Steps in the Project
### Data Collection & Preprocessing
- Load dataset and perform exploratory data analysis (EDA).
- Handle missing values and outliers.
- Encode categorical variables and normalize numerical features.

### Model Training & Experiment Tracking
- Train multiple machine learning models (e.g., **Linear Regression, Random Forest, XGBoost**).
- Tune hyperparameters using GridSearchCV or RandomizedSearchCV.
- Log model parameters, metrics, and artifacts using **MLflow**.

### Model Evaluation & Selection
- Compare different models based on:
  - **Root Mean Squared Error (RMSE)**
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
