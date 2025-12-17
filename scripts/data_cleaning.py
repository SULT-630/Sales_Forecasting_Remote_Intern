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
from sales_forecasting.load_data import clear_raw_csvs
import pandas as pd
import numpy as np
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

# Type transformation and semantic checks
def Dataset_type_transform(df: pd.DataFrame) -> pd.DataFrame:
    print("\n" + "=" * 80)
    print("2) Type transformation and semantic checks")
    print("=" * 80)
    cols_to_drop = []

    for col, meta in COLUMN_METADATA.items():
        if col not in df.columns:
            print(f"Column '{col}' not found in df. Skipping type transformation.")
            continue
        
        use_in_model = meta.get("use_in_model")

        # 只在明确标注 False 时才 drop
        if use_in_model is False:
            cols_to_drop.append(col)
            print(f"Dropping column '{col}' (use_in_model=False)")

        expected_type = meta.get("dtype")
        if expected_type is None:
            print(f"Column '{col}' has no dtype in metadata. Skipping type transformation.")
            continue

        if expected_type == "category":
            df[col] = df[col].astype("category")
        elif expected_type == "datetime":
            df[col] = pd.to_datetime(df[col], format="%y/%m/%d", errors="coerce")
        elif expected_type == "int":
            df[col] = pd.to_numeric(df[col], errors="coerce").astype("Int64")
        elif expected_type == "float":
            df[col] = pd.to_numeric(df[col], errors="coerce").astype("Float64")

        # 语义检查
        if "allowed_values" in meta:
            allowed_values = meta["allowed_values"]
            invalid_values = df[~df[col].isin(allowed_values)][col].unique()
            if len(invalid_values) > 0:
                print(f"Column '{col}' has invalid values: {invalid_values}")
            else: print(f"Column '{col}' passed allowed values check.")

    # 成交价格比基础价格高
    mask1 = df["total_price"] > df["base_price"]
    count = mask1.sum()
    ratio = count / len(df)
    print(f"Count if total price > base price: {count}")
    print(f"Ratio if total price > base price: {ratio:.2%}")

    # 折扣条件
    discount_mask = df["total_price"] < df["base_price"]

    # 新增是否折扣 SKU（0 / 1）
    df["is_discount_sku"] = discount_mask.astype(int)

    # 新增折扣力度（成交价 / 基础价）
    df["discount_ratio"] = df["total_price"] / df["base_price"]

    if cols_to_drop:
        df = df.drop(columns=cols_to_drop)

    print("\n--- head(5) ---")
    print(df.head(5))
    print("\n--- tail(5) ---")
    print(df.tail(5))
    print("\n--- dtypes ---")
    for col, dtype in df.dtypes.items():
        print(f"{col:<25} {dtype}")

    return df

# Missing values detection and ratio
def Dataset_missing_values(df: pd.DataFrame, target_col: str = "units_sold") -> pd.DataFrame:

    print("\n" + "=" * 80)
    print("3) Missing Values detection and ratio")
    print("=" * 80)

    n_rows = len(df)
    missing_cnt = df.isna().sum()
    missing_pct = (missing_cnt / n_rows * 100).round(2)
    missing_summary = (
        pd.DataFrame({"missing_count": missing_cnt, "missing_pct": missing_pct})
        .query("missing_count > 0")
        .sort_values("missing_pct", ascending=False)
    )
    if missing_summary.empty:
        print("No missing values found.")
    else:
        print(missing_summary)

    df_cleaned = df.dropna()
    print(f"\nAfter dropping missing values, new shape: {df_cleaned.shape}")

    # 存储缺失值
    clear_raw_csvs(METRIC_DIR, patterns=["missing_summary.csv"])
    missing_summary_path = METRIC_DIR / "missing_summary.csv"
    missing_summary.to_csv(missing_summary_path, index=True)
    print(f"\nSaved missing summary to: {missing_summary_path}")

    return df_cleaned

# Repeated value detection and ratio
def Dataset_repeated_values(df: pd.DataFrame, target_col: str = "units_sold") -> None:
    print("\n" + "=" * 80)
    print("4) Repeated Values detection and ratio")
    print("=" * 80)

    dup_row_count = df.duplicated().sum()
    dup_row_ratio = dup_row_count / len(df)
    print(f"Duplicated rows count: {dup_row_count}")
    print(f"Duplicated rows ratio: {dup_row_ratio:.2%}")

# Univaraite analysis and visualizations
def Dataset_univariate_analysis(df: pd.DataFrame, target_col: str = "units_sold") -> None:
    print("\n" + "=" * 80)
    print("5) Univaraite analysis and visualizations")
    print("=" * 80)

    print("\n--- 连续数值型分析 ---")
    continuous_cols = list("total_price base_price discount_ratio".split())
    cont = df[continuous_cols]
    desc = cont.describe().T
    print(desc)

    scale_dispersion = pd.DataFrame(index=continuous_cols)

    out = pd.DataFrame(index=continuous_cols)

    # range
    out["range"] = cont.max() - cont.min()

    # IQR
    q1 = cont.quantile(0.25)
    q3 = cont.quantile(0.75)
    out["IQR"] = q3 - q1

    # CV = std / mean（注意：mean接近0会爆炸；可能为负时解释也会变怪）
    mean = cont.mean()
    std = cont.std(ddof=1)
    eps = 1e-12
    out["CV"] = (std / (mean.abs() + eps)).replace([np.inf, -np.inf], np.nan)

    # CV经验分级（你可以按业务再调阈值）
    # low: <=0.10  (相对稳定)
    # medium: (0.10, 0.30]
    # high: >0.30 (相对波动大)
    out["CV_flag"] = pd.cut(
        out["CV"],
        bins=[-np.inf, 0.10, 0.30, np.inf],
        labels=["low", "medium", "high"]
    )

    # rCV（robust CV）：用IQR替代std、用median替代mean
    # 常见定义：rCV = (IQR / 1.349) / |median|
    # 其中 1.349 让“正态分布下 IQR/1.349 ≈ std”
    median = cont.median()
    robust_std = out["IQR"] / 1.349
    out["rCV"] = (robust_std / (median.abs() + eps)).replace([np.inf, -np.inf], np.nan)

    # rCV经验分级（同样可调）
    out["rCV_flag"] = pd.cut(
        out["rCV"],
        bins=[-np.inf, 0.10, 0.30, np.inf],
        labels=["low", "medium", "high"]
    )

    print("\n连续数值型离散度指标:")
    print(out) 

    # 连续性分布绘图
    price_cols = ["total_price", "base_price"]
    for col in price_cols:
        data = df[col].dropna()
        plt.figure(figsize=(8, 5))

        # 1) 直方图
        plt.hist(data, bins=50, density=True, alpha=0.6)

        # 2) KDE 曲线（pandas 自带）
        data.plot(kind="kde")

        plt.title(f"Distribution of {col}")
        plt.xlabel(col)
        plt.ylabel("Density")

        plt.tight_layout()
        cont_fig_path = FIG_DIR / f"{col}_distribution.png"
        clear_raw_csvs(FIG_DIR, patterns=[f"{col}_distribution.png"])
        plt.savefig(cont_fig_path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"Saved target histogram to: {cont_fig_path}")

    # 保存图片
   

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
    df_after_type_and_semantic = Dataset_type_transform(raw_df)
    df_after_missing = Dataset_missing_values(df_after_type_and_semantic, target_col="units_sold")
    Dataset_repeated_values(df_after_missing, target_col="units_sold")

    Dataset_univariate_analysis(df_after_missing, target_col="units_sold")


if __name__ == "__main__":
    main()