"""
Module: data_preprocess.py
Description: Data preprocessing utilities for sales forecasting
"""

import pandas as pd
import numpy as np
from sales_forecasting.load_data import clear_raw_csvs 
from sales_forecasting.spatial_decomposition import df_after_missing_value_handling

def preprocess_data(df: pd.DataFrame, target_col: str, id_col: str = "record_ID") -> pd.DataFrame:
    """
    Preprocess the input dataframe by various methods
    New features: log_sales
    Drop features: week, units_sold
    """

    df = df.copy()
    df['log_sales'] = np.log1p(df[target_col])
    df = df.drop(columns=['week','is_discount_sku','discount_ratio', target_col], errors='ignore')

    return df