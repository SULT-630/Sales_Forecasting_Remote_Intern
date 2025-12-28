"""
Script: entrance.py
Purpose: run on different df and models
Run: python scripts/entrance.py
"""

import pandas as pd
import numpy as np
from xgboost import XGBRegressor
from sales_forecasting.experiment import Experiment
from sales_forecasting.spatial_decomposition import df_after_missing_value_handling
from sales_forecasting.data_preprocess import preprocess_data
from sales_forecasting.Train_Eval import DataProcessor
test_path = "data/raw/test_nfcJ3J5.csv"
target_col = "unit_solds"

my_dataframe = df_after_missing_value_handling.copy()
df_after_log = df_after_missing_value_handling.copy()
df_after_log['log_sales'] = np.log1p(df_after_log['units_sold'])
df_after_log = df_after_log.drop(columns=['units_sold'], errors='ignore')
my_dataframe = preprocess_data(my_dataframe, target_col='units_sold', id_col='record_ID') 
print("--- Dataframe loaded for experiment: ---")
print(my_dataframe.head(5))

df_sorted = my_dataframe.sort_values("week")
print("--- First weeks: ---")
print(df_sorted.head())


processor = DataProcessor(
    target_col='log_sales', 
    id_col='record_ID', 
    time_col='week', 
    test_size=0.2)

X_train_false, X_test_false, y_train_false, y_test_false = processor.split(my_dataframe) # 训练的时候用这个
X_train, X_test, y_train, y_test = processor.split(df_after_log) # 测试的时候用这个

# 验证是否y train/test 和 false 一致
print("--- Validating train/test splits ---")
assert y_train.equals(y_train_false), "y_train 不一致"
assert y_test.equals(y_test_false), "y_test 不一致"
# 理论上应该是一样的 

# 超参数调优暂时不考虑
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
    Title = None
)

# exp.run(X_train, X_test, y_train, y_test, X_train_false, y_train_false, Title = "XGB_log_sales",transform_type='log1p') #真实的rolling predict
exp.run(X_train_false, X_test_false, y_train_false, y_test_false, X_train_false, y_train_false, Title = "XGB_log_sales_champion-target_changing_rate",transform_type='log1p')