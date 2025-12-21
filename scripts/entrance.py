"""
Script: entrance.py
Purpose: run on different df and models
Run: python scripts/entrance.py
"""

import pandas as pd
from xgboost import XGBClassifier
from sales_forecasting.experiment import Experiment
from sales_forecasting.spatial_decomposition import df_after_missing_value_handling

test_path = "data/raw/test_nfcJ3J5.csv"
target_col = "unit_solds"

my_dataframe = df_after_missing_value_handling.copy()

model = XGBClassifier(
    n_estimators=300,
    max_depth=6,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    eval_metric="logloss"
)

exp = exp = Experiment(
    df=my_dataframe,
    target_col="units_sold",
    model=model,
    task_type="regression",
    Title = "XGB_RAW_DATA"
)

def load_and_split_data(
    df_train: pd.DataFrame,
    df_test: pd.DataFrame,
    target_col: str
):

    for name, df in [("train", df_train), ("test", df_test)]:
        if target_col not in df.columns:
            raise ValueError(
                f"{target_col} not found in {name} dataset columns"
            )

    X_train = df_train.drop(columns=[target_col])
    y_train = df_train[target_col]

    X_test = df_test.drop(columns=[target_col])
    y_test = df_test[target_col]

    return X_train, X_test, y_train, y_test