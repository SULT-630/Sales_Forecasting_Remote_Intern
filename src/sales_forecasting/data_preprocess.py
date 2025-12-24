"""
Module: data_preprocess.py
Description: Data preprocessing utilities for sales forecasting
"""

import pandas as pd
import numpy as np
from sales_forecasting.load_data import clear_raw_csvs 
from sales_forecasting.spatial_decomposition import df_after_missing_value_handling
from sales_forecasting.spatial_decomposition import encoding_year_month_quarter
from sales_forecasting.spatial_decomposition import encoding_week_of_year
from sales_forecasting.spatial_decomposition import encoding_is_month_start_end
from sales_forecasting.spatial_decomposition import encoding_is_quarter_start_end
from sales_forecasting.spatial_decomposition import encoding_sin_cos_week_of_year
from sales_forecasting.spatial_decomposition import encoding_sin_cos_month
from sales_forecasting.spatial_decomposition import encoding_week_from_start
from sales_forecasting.spatial_decomposition import encoding_gap_since_lag_record
from sales_forecasting.spatial_decomposition import encoding_lag_features
from sales_forecasting.spatial_decomposition import rolling_mean_std_features
from sales_forecasting.spatial_decomposition import encoding_EWMA_features
from sales_forecasting.spatial_decomposition import encoding_target_changing_rate
from sales_forecasting.spatial_decomposition import encoding_target_changing_rate_per_gap
# from sales_forecasting.spatial_decomposition import week_total_units_sold_feature
# from sales_forecasting.spatial_decomposition import store_total_units_sold_feature
# from sales_forecasting.spatial_decomposition import sku_total_units_sold_feature


def preprocess_data(df: pd.DataFrame, target_col: str, id_col: str = "record_ID") -> pd.DataFrame:
    """
    Preprocess the input dataframe by various methods
    New features: log_sales
    Drop features: week, units_sold
    """
    df = df.copy()
    if 'log_sales' not in df.columns:
        df['log_sales'] = np.log1p(df[target_col])
        df = df.drop(columns=[target_col], errors='ignore')
    df = encoding_year_month_quarter(df)
    df = encoding_week_of_year(df)
    df = encoding_is_month_start_end(df)
    df = encoding_is_quarter_start_end(df)
    df = encoding_sin_cos_week_of_year(df)
    df = encoding_sin_cos_month(df)
    # 看一下是在哪里进行的排序的
    # df = encoding_week_from_start(df, start_date=pd.Timestamp("2020-01-01"))
    df = encoding_gap_since_lag_record(df, lag_weeks=[1,2,4])
    df = encoding_lag_features(df, lag_weeks=[1,2,4]) 
    df = rolling_mean_std_features(df, window_sizes=[2,3,4]) 
    df = encoding_EWMA_features(df, spans=[2,3,4]) 
    df = encoding_target_changing_rate(df, periods=[1,2,4]) 
    df = encoding_target_changing_rate_per_gap(df, periods=[1,2,4]) 
    # df = week_total_units_sold_feature(df) 
    # df = store_total_units_sold_feature(df) 
    # df = sku_total_units_sold_feature(df) 
    return df