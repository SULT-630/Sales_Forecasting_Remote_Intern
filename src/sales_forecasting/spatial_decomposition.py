"""
Script: spatial_decomposition.py
Purpose: Performs spatial decomposition on geospatial datasets and the feature cols to dataframe.
Run: python scripts/data_cleaning.py
"""

from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sales_forecasting.load_data import load_kaggle_dataset
from sales_forecasting.load_data import clear_raw_csvs
from sales_forecasting.data_cleaning import Dataset_type_transform
from sales_forecasting.data_cleaning import Dataset_missing_values
from sales_forecasting.data_cleaning import Dataset_univariate_analysis

# 各种路径
PROJECT_ROOT = Path(__file__).resolve().parents[2]
FIG_DIR = PROJECT_ROOT / "artifacts" / "figures" / "spatial_decomposition"
METRIC_DIR = PROJECT_ROOT / "artifacts" / "metrics" / "spatial_decomposition"
FIG_DIR.mkdir(parents=True, exist_ok=True)
METRIC_DIR.mkdir(parents=True, exist_ok=True)

# 加载数据
raw_df = load_kaggle_dataset()
df_after_type_transform = Dataset_type_transform(raw_df)
df_after_missing_value_handling = Dataset_missing_values(df_after_type_transform)
cleaned_df = Dataset_univariate_analysis(df_after_missing_value_handling)

def depict_mixed_and_seperate_sku_sales_curve(df: pd.DataFrame) -> None:
    df = df.copy()

    # 混合sku曲线
    print("\n--- Depicting Mixed SKU Sales Curve ---")
    mixed_sales = (
        df.groupby('week')['units_sold']
          .sum()
          .sort_index()
    )

    plt.figure(figsize=(12, 5))
    plt.plot(mixed_sales.index, mixed_sales.values)
    plt.xlabel('Week')
    plt.ylabel('Units Sold')
    plt.title('Weekly Total Units Sold (All SKUs)')
    plt.grid(True)
    plt.tight_layout()
    sku_sales_fig_path = FIG_DIR / "mixed_sku_sales_curve.png"
    clear_raw_csvs(FIG_DIR, patterns=["mixed_sku_sales_curve.png"])
    plt.savefig(sku_sales_fig_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved SKU sales curve to: {sku_sales_fig_path}")

    # 每个sku单独曲线
    print("\n--- Depicting Seperate SKU Sales Curve ---")
    for sku_id in df['sku_id'].unique():
        sku_df = df[df['sku_id'] == sku_id]

        sku_sales = (
            sku_df.groupby('week')['units_sold']
                  .sum()
                  .sort_index()
        )

        plt.figure(figsize=(12, 5))
        plt.plot(sku_sales.index, sku_sales.values)
        plt.xlabel('Week')
        plt.ylabel('Units Sold')
        plt.title(f'Weekly Units Sold - SKU {sku_id}')
        plt.grid(True)
        plt.tight_layout()
        sku_sales_fig_path = FIG_DIR / f"sku_id_equals_{sku_id}_sales_curve.png"
        clear_raw_csvs(FIG_DIR, patterns=[f"sku_id_equals_{sku_id}sales_curve.png"])
        plt.savefig(sku_sales_fig_path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"Saved SKU sales curve to: {sku_sales_fig_path}")

# 特征工程和时间编码等
def encoding_year_month_quarter(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    # df['year'] = df['week'].dt.year
    # df['month'] = df['week'].dt.month
    df['quarter'] = df['week'].dt.quarter
    return df

def encoding_week_of_year(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df['week_of_year'] = df['week'].dt.isocalendar().week
    return df

def encoding_is_month_start_end(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df['is_month_start'] = df['week'].dt.is_month_start.astype(int)
    df['is_month_end'] = df['week'].dt.is_month_end.astype(int)
    return df

def encoding_is_quarter_start_end(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df['is_quarter_start'] = df['week'].dt.is_quarter_start.astype(int)
    df['is_quarter_end'] = df['week'].dt.is_quarter_end.astype(int)
    return df

def encoding_sin_cos_week_of_year(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df['week_of_year'] = df['week'].dt.isocalendar().week
    df['sin_week_of_year'] = np.sin(2 * np.pi * df['week_of_year'] / 52)
    df['cos_week_of_year'] = np.cos(2 * np.pi * df['week_of_year'] / 52)
    df.drop(columns=['week_of_year'], inplace=True)
    return df

def encoding_sin_cos_month(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df['month'] = df['week'].dt.month
    df['sin_month'] = np.sin(2 * np.pi * df['month'] / 12)
    df['cos_month'] = np.cos(2 * np.pi * df['month'] / 12)    
    df.drop(columns=['month'], inplace=True)
    return df

# 调用的时候注意
def encoding_sin_cos_week_from_start(df: pd.DataFrame, start_date: pd.Timestamp) -> pd.DataFrame:
    df = df.copy()
    df['week_from_start'] = (df['week'] - start_date).dt.days // 7
    df['sin_week_from_start'] = np.sin(2 * np.pi * df['week_from_start'] / 52)
    df['cos_week_from_start'] = np.cos(2 * np.pi * df['week_from_start'] / 52)
    df.drop(columns=['week_from_start'], inplace=True)
    return df

# 检查是否正确
def encoding_gap_since_lag_record(df: pd.DataFrame, lag_weeks: list) -> pd.DataFrame:
    df = df.copy()
    df = df.sort_values(by=['sku_id', 'week'])
    for lag in lag_weeks:
        df[f'gap_since_lag_{lag}_records'] = df.groupby('sku_id', observed=False)['week'].diff(lag).dt.days.fillna(0) / 7
    return df

def encoding_lag_features(df: pd.DataFrame, lag_weeks: list) -> pd.DataFrame:
    df = df.copy()
    df = df.sort_values(by=['sku_id', 'week'])
    for lag in lag_weeks:
        df[f'lag_{lag}_weeks'] = df.groupby('sku_id', observed=False)['log_sales'].shift(lag)
    return df


def rolling_mean_std_features(df: pd.DataFrame, window_sizes: list) -> pd.DataFrame:
    df = df.copy()
    df = df.sort_values(by=['sku_id', 'week'])

    for window in window_sizes:
        hist = df.groupby('sku_id', observed=False)['log_sales'].shift(1)

        df[f'rolling_mean_{window}_records'] = (
            hist.groupby(df['sku_id'], observed=False)
                .transform(lambda x: x.rolling(window, min_periods=1).mean())
        )

        df[f'rolling_std_{window}_records'] = (
            hist.groupby(df['sku_id'], observed=False)
                .transform(lambda x: x.rolling(window, min_periods=1).std())
                .fillna(0)
        )

    return df


def encoding_EWMA_features(df: pd.DataFrame, spans: list) -> pd.DataFrame:
    df = df.copy()
    df = df.sort_values(by=['sku_id', 'week'])

    hist = df.groupby('sku_id', observed=False)['log_sales'].shift(1)

    for span in spans:
        df[f'ewma_{span}_records'] = (
            hist.groupby(df['sku_id'], observed=False)
                .transform(lambda x: x.ewm(span=span, adjust=False).mean())
        )

    return df


def encoding_target_changing_rate(df: pd.DataFrame, periods: list, target_col='log_sales') -> pd.DataFrame:
    df = df.copy()
    df = df.sort_values(by=['sku_id', 'week'])

    hist = df.groupby('sku_id', observed=False)[target_col].shift(1)  # t 时刻只能看到 t-1
    for period in periods:
        # 这里的 pct_change 发生在 hist 上：相当于 (y_{t-1} - y_{t-1-period}) / y_{t-1-period}
        df[f'target_changing_rate_{period}_records'] = (
            hist.groupby(df['sku_id'], observed=False)
                .pct_change(periods=period, fill_method=None)
                .fillna(0)
        )
    return df


def encoding_target_changing_rate_per_gap(
    df: pd.DataFrame,
    periods: list,
    gap_col_template: str = 'gap_since_lag_{k}_records',
    eps: float = 1e-6
) -> pd.DataFrame:
    df = df.copy()
    df = encoding_target_changing_rate(df, periods=periods)
    for k in periods:
        gap_col = gap_col_template.format(k=k)
        if gap_col not in df.columns:
            raise ValueError(f'Missing gap column: {gap_col}')

        df[f'target_changing_rate_per_week_{k}'] = (
            df[f'target_changing_rate_{k}_records'] /
            (df[gap_col] + eps)
        )
    df.drop(columns=[f'target_changing_rate_{k}_records' for k in periods], inplace=True)
    return df


# 聚合特征
# def week_total_units_sold_feature(df: pd.DataFrame) -> pd.DataFrame:
#     df = df.copy()
#     week_sum = df.groupby('week')['log_sales'].sum()
#     week_sum_lag1 = week_sum.sort_index().shift(1)
#     df['week_total_units_sold_lag1'] = df['week'].map(week_sum_lag1)

#     return df

# def store_total_units_sold_feature(df: pd.DataFrame) -> pd.DataFrame:
#     df = df.copy()
#     store_sum = df.groupby('store_id', observed=False)['log_sales'].sum()
#     store_sum_lag1 = store_sum.sort_index().shift(1)
#     df['store_total_units_sold_lag1'] = df['store_id'].map(store_sum_lag1)
#     return df

# def sku_total_units_sold_feature(df: pd.DataFrame) -> pd.DataFrame:
#     df = df.copy()
#     sku_sum = df.groupby('sku_id', observed=False)['log_sales'].sum()
#     sku_sum_lag1 = sku_sum.sort_index().shift(1)
#     df['sku_total_units_sold_lag1'] = df['sku_id'].map(sku_sum_lag1)
#     return df



def main():
    print(cleaned_df.head())
    print("Spatial decomposition module executed successfully.")
    depict_mixed_and_seperate_sku_sales_curve(cleaned_df)

if __name__ == "__main__":
    main()