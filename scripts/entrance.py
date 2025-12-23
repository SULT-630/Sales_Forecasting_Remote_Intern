"""
Script: entrance.py
Purpose: run on different df and models
Run: python scripts/entrance.py
"""

import pandas as pd
from xgboost import XGBRegressor
from sales_forecasting.experiment import Experiment
from sales_forecasting.spatial_decomposition import df_after_missing_value_handling
from sales_forecasting.data_preprocess import preprocess_data

test_path = "data/raw/test_nfcJ3J5.csv"
target_col = "unit_solds"

my_dataframe = df_after_missing_value_handling.copy()
my_dataframe = preprocess_data(my_dataframe, target_col='units_sold', id_col='record_ID')
print("--- Dataframe loaded for experiment: ---")
print(my_dataframe.head(5))

model = XGBRegressor(
    objective="reg:squarederror",
    n_estimators=500,
    learning_rate=0.05,
    max_depth=8,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    n_jobs=-1,
    enable_categorical=True
)

exp = Experiment(
    df=my_dataframe,
    target_col="log_sales",
    model=model,
    task_type="regression",
    Title = "XGB_log_sales"
)

exp.run(Title = "XGB_log_sales",transform_type='log1p')