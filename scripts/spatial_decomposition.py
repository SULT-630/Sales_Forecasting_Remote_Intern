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
PROJECT_ROOT = Path(__file__).resolve().parents[1]
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
def encoding(df: pd.DataFrame) -> pd.DataFrame:


def main():
    print(cleaned_df.head())
    print("Spatial decomposition module executed successfully.")
    depict_mixed_and_seperate_sku_sales_curve(cleaned_df)

if __name__ == "__main__":
    main()