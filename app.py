'''
This python code is used to perform ML model tracking using MLflow remote tracking server
'''

from sklearn.metrics import r2_score, mean_absolute_error, make_scorer
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR

import pandas as pd
import numpy as np

import mlflow.sklearn
from mlflow.models import infer_signature
import mlflow

from urllib.parse import urlparse
from dotenv import load_dotenv
import logging
import os


logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

load_dotenv()
os.environ["MLFLOW_TRACKING_URI"]=os.getenv('MLFLOW_TRACKING_URI')

if __name__ == '__main__':
    df = pd.read_csv('car_price_dataset.csv')

    categorical_cols = df.select_dtypes(include='object').columns
    cols_to_scale = ['Engine_Size', "Mileage", 'Doors', 'Owner_Count']

    df_v1 = df.copy()
    le = LabelEncoder()
    for i in categorical_cols:
        df_v1[i] = le.fit_transform(df_v1[i])

    df_v2 = df.copy()
    def one_hot_encoding(df, categorical_cols):
        for col in categorical_cols:
            dummies = pd.get_dummies(df[col], prefix=col).astype(int)
            col_index = df.columns.get_loc(col) + 1
            for dummy_col in reversed(dummies.columns):
                df.insert(col_index, dummy_col, dummies[dummy_col])
            df.drop(col, axis=1, inplace=True)
        return df
    df_v2 = one_hot_encoding(df_v2, categorical_cols)

    df_v3 = df_v1.copy()
    for i in cols_to_scale:
        scaler = MinMaxScaler()
        df_v3[i] = scaler.fit_transform(df_v3[[i]])

    df_v4 = df_v1.copy()
    for i in cols_to_scale:
        scaler = StandardScaler()
        df_v4[i] = scaler.fit_transform(df_v4[[i]])

    df_v5 = df_v2.copy()
    for i in cols_to_scale:
        scaler = MinMaxScaler()
        df_v5[i] = scaler.fit_transform(df_v5[[i]])

    df_v6 = df_v2.copy()
    for i in cols_to_scale:
        scaler = StandardScaler()
        df_v6[i] = scaler.fit_transform(df_v6[[i]])

    # List of datasets
    datasets = {'df1': df_v1, 'df2': df_v2, 'df3': df_v3, 'df4': df_v4, 'df5': df_v5, 'df6': df_v6}

    # Dictionary to store train and test splits
    train_sets = {}
    test_sets = {}

    # Perform train-test split separately for each dataset
    for name, df in datasets.items():
        train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)
        train_sets[name] = train_df  # Store train set
        test_sets[name] = test_df    # Store test set

    # Access train and test splits
    df_v1_train = train_sets['df1']
    df_v1_test = test_sets['df1']

    df_v2_train = train_sets['df2']
    df_v2_test = test_sets['df2']

    df_v3_train = train_sets['df3']
    df_v3_test = test_sets['df3']

    df_v4_train = train_sets['df4']
    df_v4_test = test_sets['df4']

    df_v5_train = train_sets['df5']
    df_v5_test = test_sets['df5']

    df_v6_train = train_sets['df6']
    df_v6_test = test_sets['df6']

    datasets = [
        ['Label encoding', [df_v1_train[df_v1_train.columns[:-1]], df_v1_train[df_v1_train.columns[-1]], df_v1_test[df_v1_test.columns[:-1]], df_v1_test[df_v1_test.columns[-1]]]],
        ['One-hot encoding', [df_v2_train[df_v2_train.columns[:-1]], df_v2_train[df_v2_train.columns[-1]], df_v2_test[df_v2_test.columns[:-1]], df_v2_test[df_v2_test.columns[-1]]]],
        ['Label encoding + MinMax scaler', [df_v3_train[df_v3_train.columns[:-1]], df_v3_train[df_v3_train.columns[-1]], df_v3_test[df_v3_test.columns[:-1]], df_v3_test[df_v3_test.columns[-1]]]],
        ['Label encoding + Standard scaler', [df_v4_train[df_v4_train.columns[:-1]], df_v4_train[df_v4_train.columns[-1]], df_v4_test[df_v4_test.columns[:-1]], df_v4_test[df_v4_test.columns[-1]]]],
        ['One-hot encoding + MinMax scaler', [df_v5_train[df_v5_train.columns[:-1]], df_v5_train[df_v5_train.columns[-1]], df_v5_test[df_v5_test.columns[:-1]], df_v5_test[df_v5_test.columns[-1]]]],
        ['One-hot encoding + Standard scaler', [df_v6_train[df_v6_train.columns[:-1]], df_v6_train[df_v6_train.columns[-1]], df_v6_test[df_v6_test.columns[:-1]], df_v6_test[df_v6_test.columns[-1]]]]
    ]

    models = [
        {
            'model': RandomForestRegressor(),
            'name': 'Random Forest Regressor',
            'params': {
                'n_estimators': np.arange(50, 201, 50),
                'max_depth': [None] + list(np.arange(10, 51, 10)),
                'min_samples_split': np.arange(2, 11, 2),
                'min_samples_leaf': np.arange(1, 5),
                'max_features': ['sqrt', 'log2'],
                'bootstrap': [True, False]
            }
        },
        {
            'model': LinearRegression(),
            'name': 'Linear Regression',
            'params': {
                'fit_intercept': [True, False],
                'positive': [True, False]
            }
        },
        {
            'model': DecisionTreeRegressor(random_state=42),
            'name': 'Decision Tree Regressor',
            'params': {
                'max_depth': [None] + list(np.arange(3, 21, 3)),
                'min_samples_split': np.arange(2, 11, 2),
                'min_samples_leaf': np.arange(1, 10),
                'max_features': ['sqrt', 'log2', None],
            }
        },
        {
            'model': SVR(),
            'name': 'Support Vector Regressor',
            'params': {}
        }
    ]

    # Initialize MLflow
    load_dotenv()
    mlflow.set_experiment("Car Price Prediction 1")
    mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI"))
    tracking_url_type_store=urlparse(mlflow.get_tracking_uri()).scheme

    scoring = {
        'r2': make_scorer(r2_score),
        'mae': make_scorer(mean_absolute_error)
    }

    total_runs = len(datasets) * len(models)
    run_counter = 1
    for i in datasets:
        dataset_name = i[0]
        logging.info(f"Starting training on dataset: {dataset_name}")
        X_train, y_train, X_test, y_test = i[1]
        for j in models:
            run_name = f"{dataset_name} + {j['name']}"
            logging.info(f"[{run_counter}/{total_runs}] Training model: {run_name}")
            run_counter += 1

            run_name=f"{i[0]} + {j['name']}"
            logging.info(f"Training {run_name}...")
            model = RandomizedSearchCV(j['model'], j['params'], cv=3, random_state=42, scoring=scoring, refit='r2')
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            r2, mae = r2_score(y_test, y_pred), mean_absolute_error(y_test, y_pred)

            signature=infer_signature(X_train,y_train)
            with mlflow.start_run(run_name=run_name):
                mlflow.log_param("model", run_name)
                mlflow.log_params(model.best_params_)
                mlflow.log_metric('r2_score', r2)
                mlflow.log_metric('mean_absolute_error', mae)
                # mlflow.sklearn.log_model(model, "model")

                if tracking_url_type_store !='file':
                    mlflow.sklearn.log_model(model,"model",registered_model_name=f"Best {run_name}")
                else:
                    mlflow.sklearn.log_model(model,"model",signature=signature)
                logging.info(f"✅ Finished training: {run_name}")