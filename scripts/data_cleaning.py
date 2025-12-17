"""
Script: data_cleaning.py
Purpose: Data Cleaning Module for Weekly SKU-Store Sales Panel Data including EDA, solving missing values, and outlier detection with summary and visualizations.
Run: python scripts/data_cleaning.py
"""

# Dataset Description:
"""
Dataset Granularity:
- Each row represents the weekly aggregated sales of a specific SKU
  at a specific store.

Features:
- record_ID: Unique identifier for each record.
- week: Start date of the sales week.
- store_id: Identifier for the store.
- sku_id: Identifier for the SKU (product).
- total_price: Actual selling price after discounts/promotions.
- base_price: Regular price of the SKU.
- is_featured_sku: Binary flag indicating if the SKU was featured that week.
- is_display_sku: Binary flag indicating if the SKU was on display that week.

Target Variable:
- units_sold: Number of units sold in that week.

Notes:
- record_ID is an artificial identifier and has no business meaning.
- total_price < base_price can occur even when promotion flags are 0.
- Time stamps may be shifted and should not be aligned with real-world dates.
- The dataset is a panel (time series) across (store_id, sku_id).
"""

from pathlib import Path
from sales_forecasting.load_data import load_kaggle_dataset
from sales_forecasting.schema import COLUMN_METADATA
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# 各种路径
PROJECT_ROOT = Path(__file__).resolve().parents[1]
FIG_DIR = PROJECT_ROOT / "artifacts" / "figures"
METRIC_DIR = PROJECT_ROOT / "artifacts" / "metrics"
FIG_DIR.mkdir(parents=True, exist_ok=True)
METRIC_DIR.mkdir(parents=True, exist_ok=True)

# Shape and structure of the dataset
def Dataset_shape(df: pd.DataFrame, target_col: str = "units_sold") -> None:
    """
    Run EDA 
    Saves: missing_summary.csv, numeric_summary.csv, correlation_heatmap.png, target_hist.png
    """

    print("\n" + "=" * 80)
    print("1) Shape and Strucutre")
    print("=" * 80)
    n_rows, n_cols = df.shape
    print(f"Rows: {n_rows:,}")
    print(f"Cols: {n_cols:,}")
    print("\n--- head(5) ---")
    print(df.head(5))
    print("\n--- tail(5) ---")
    print(df.tail(5))
    print("\n--- dtypes ---")
    for col, dtype in df.dtypes.items():
        print(f"{col:<25} {dtype}")

# Missing values detection and ratio
# def Dataset_missing_values(df: pd.DataFrame, target_col: str = "units_sold") -> None:
#     """
#     Missing values detection and ratio
#     """
#     print("\n" + "=" * 80)
#     print("2) Missing Values detection and ratio")
#     print("=" * 80)



    # print("\n" + "=" * 80)
    # print("2) HEAD / TAIL")
    # print("=" * 80)
    # print("\n--- head(5) ---")
    # print(df.head(5))
    # print("\n--- tail(5) ---")
    # print(df.tail(5))

    # print("\n" + "=" * 80)
    # print("3) DTYPES")
    # print("=" * 80)
    # print(df.dtypes)

    # # 1. 列对照
    # print("\n" + "=" * 80)
    # print("Schema quick check (columns in schema vs df)")
    # print("=" * 80)
    # schema_cols = set(COLUMN_METADATA.keys())
    # df_cols = set(df.columns)
    # missing_in_df = sorted(list(schema_cols - df_cols))
    # extra_in_df = sorted(list(df_cols - schema_cols))
    # if missing_in_df:
    #     print("Missing columns in df (present in schema):", missing_in_df)
    # else:
    #     print("No missing columns relative to schema.")
    # if extra_in_df:
    #     print("Extra columns in df (not in schema):", extra_in_df)
    # else:
    #     print("No extra columns relative to schema.")

    # # 2. 缺失值概览
    # print("\n" + "=" * 80)
    # print("4) MISSING VALUES OVERVIEW")
    # print("=" * 80)
    # miss_cnt = df.isna().sum()
    # miss_pct = (miss_cnt / len(df) * 100).round(2)
    # missing_summary = (
    #     pd.DataFrame({"missing_count": miss_cnt, "missing_pct": miss_pct})
    #     .query("missing_count > 0")
    #     .sort_values("missing_pct", ascending=False)
    # )
    # if missing_summary.empty:
    #     print("No missing values found.")
    # else:
    #     print(missing_summary)

    # # 3. 存储缺失值
    # missing_summary_path = METRIC_DIR / "missing_summary.csv"
    # missing_summary.to_csv(missing_summary_path, index=True)
    # print(f"\nSaved missing summary to: {missing_summary_path}")

    # # 4. 数值型统计汇总
    # print("\n" + "=" * 80)
    # print("5) NUMERIC SUMMARY (describe)")
    # print("=" * 80)
    # numeric_cols = df.select_dtypes(include="number").columns
    # if len(numeric_cols) == 0:
    #     print("No numeric columns found.")
    #     numeric_summary = pd.DataFrame()
    # else:
    #     numeric_summary = df[numeric_cols].describe().T
    #     print(numeric_summary)

    # # 5. 存储数值型统计汇总
    # numeric_summary_path = METRIC_DIR / "numeric_summary.csv"
    # numeric_summary.to_csv(numeric_summary_path, index=True)
    # print(f"\nSaved numeric summary to: {numeric_summary_path}")

    # # 6. 目标变量分布
    # print("\n" + "=" * 80)
    # print("6) TARGET DISTRIBUTION")
    # print("=" * 80)
    # if target_col not in df.columns:
    #     print(f"Target column '{target_col}' not found in df. Skipping target distribution.")
    # else:
    #     y = df[target_col]
    #     print("Target basic stats:")
    #     print(y.describe())

    #     zero_ratio = (y == 0).mean() * 100
    #     print(f"Zero ratio: {zero_ratio:.2f}%")

    #     # 直方图
    #     plt.figure(figsize=(8, 5))
    #     plt.hist(y.dropna(), bins=50)
    #     plt.title(f"Target Distribution: {target_col}")
    #     plt.xlabel(target_col)
    #     plt.ylabel("Count")
    #     target_fig_path = FIG_DIR / "target_hist.png"
    #     plt.savefig(target_fig_path, dpi=150, bbox_inches="tight")
    #     plt.close()
    #     print(f"Saved target histogram to: {target_fig_path}")

    #     # log1p 直方图
    #     plt.figure(figsize=(8, 5))
    #     plt.hist((y.dropna()).map(lambda v: 0 if v < 0 else v).pipe(lambda s: s.apply(lambda v: __import__("math").log1p(v))),
    #              bins=50)
    #     plt.title(f"Target Distribution (log1p): {target_col}")
    #     plt.xlabel(f"log1p({target_col})")
    #     plt.ylabel("Count")
    #     target_log_fig_path = FIG_DIR / "target_hist_log1p.png"
    #     plt.savefig(target_log_fig_path, dpi=150, bbox_inches="tight")
    #     plt.close()
    #     print(f"Saved log1p target histogram to: {target_log_fig_path}")

    # # 7. 数值型相关性热力图
    # print("\n" + "=" * 80)
    # print("7) CORRELATION / HEATMAP (numeric only)")
    # print("=" * 80)
    # if len(numeric_cols) < 2:
    #     print("Not enough numeric columns to compute correlation.")
    # else:
    #     corr = df[numeric_cols].corr()

    #     plt.figure(figsize=(10, 8))
    #     sns.heatmap(corr, annot=False, center=0)
    #     plt.title("Correlation Heatmap (Numeric Features)")
    #     heatmap_path = FIG_DIR / "correlation_heatmap.png"
    #     plt.savefig(heatmap_path, dpi=150, bbox_inches="tight")
    #     plt.close()
    #     print(f"Saved correlation heatmap to: {heatmap_path}")

def main():

    raw_df = load_kaggle_dataset()

    Dataset_shape(raw_df, target_col="units_sold")


if __name__ == "__main__":
    main()