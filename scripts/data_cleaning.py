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
def Dataset_univariate_analysis(df: pd.DataFrame, target_col: str = "units_sold") -> pd.DataFrame:
    print("\n" + "=" * 80)
    print("5) Univaraite analysis and visualizations")
    print("=" * 80)

    # 连续数值型分析
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

    # CV经验分级（阈值可以重新调整）
    # low: <=0.10  (相对稳定)
    # medium: (0.10, 0.30]
    # high: >0.30 (相对波动大)
    out["CV_flag"] = pd.cut(
        out["CV"],
        bins=[-np.inf, 0.10, 0.30, np.inf],
        labels=["low", "medium", "high"]
    )

    # rCV（robust CV）：用IQR替代std、用median替代mean
    median = cont.median()
    robust_std = out["IQR"] / 1.349
    out["rCV"] = (robust_std / (median.abs() + eps)).replace([np.inf, -np.inf], np.nan)

    # rCV经验分级（同样可调）
    out["rCV_flag"] = pd.cut(
        out["rCV"],
        bins=[-np.inf, 0.10, 0.30, np.inf],
        labels=["low", "medium", "high"]
    )
    print("\n--- 连续数值型离散度指标 ---")
    print(out) 

    # 连续性分布绘图
    print("\n--- 连续数值型直方图, KDE与箱图 ---")
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
        print(f"Saved histogram to: {cont_fig_path}")

    # 箱图与outliers
    for col in price_cols:
        data = df[col].dropna()

        plt.figure(figsize=(6, 5))

        plt.boxplot(
            data,
            vert=True,
            showfliers=True   # 显示异常值
        )

        plt.title(f"Boxplot of {col}")
        plt.xlabel(col)
        plt.ylabel(col)

        plt.tight_layout()

        cont_fig_path = FIG_DIR / f"{col}_boxplot.png"
        clear_raw_csvs(FIG_DIR, patterns=[f"{col}_boxplot.png"])
        plt.savefig(cont_fig_path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"Saved boxplot to: {cont_fig_path}")

    for col in price_cols:
        data = df[col].dropna()
        # IQR 方法检测异常值数量
        print("\n---"  f"{col} Outliers ---")
        n = len(data)
        q1 = data.quantile(0.25)
        q3 = data.quantile(0.75)
        iqr = q3 - q1
        lb_iqr = q1 - 1.5 * iqr
        ub_iqr = q3 + 1.5 * iqr
        out_iqr = ((data < lb_iqr) | (data > ub_iqr)).sum()
        print(f"{col} - IQR outliers: {out_iqr} ({out_iqr / n:.2%})")

    # 类别变量分析
    print("\n--- 类别变量分析 ---")
    category_cols = list("store_id sku_id is_featured_sku is_display_sku is_discount_sku".split())

    binary_cols = ["is_featured_sku", "is_display_sku", "is_discount_sku"]

    binary_summary = {}

    for col in binary_cols:
        vc = df[col].value_counts(dropna=False)
        ratio = df[col].value_counts(normalize=True, dropna=False)

        summary = pd.DataFrame({
            "count": vc,
            "ratio": ratio
        })

        print(f"\n--- {col} ---")
        print(summary)

        binary_summary[col] = summary

    store_counts = df["store_id"].value_counts()

    store_summary = pd.DataFrame({
        "count": store_counts,
        "ratio": store_counts / len(df)
    })

    print("\n--- store_id summary (top 10) ---")
    print(store_summary.head(10))

    print("\n--- store_id summary (Least 10) ---")
    print(store_summary.tail(10))

    print("\nNumber of stores:", store_counts.shape[0])
    print("Median samples per store:", store_counts.median())
    print("Min samples per store:", store_counts.min())

    print("\n--- sku_id ---")
    sku_counts = df["sku_id"].value_counts()

    print("\nNumber of SKUs:", sku_counts.shape[0])
    print("Median samples per SKU:", sku_counts.median())
    print("Min samples per SKU:", sku_counts.min())

    # 看看 SKU 样本数分布（分位数）
    print("\nSKU sample count quantiles:")
    print(sku_counts.quantile([0.1, 0.25, 0.5, 0.75, 0.9]))

    # 目标变量
    print("\n--- units sold (target var) ---")

    if target_col not in df.columns:
        print(f"Target column '{target_col}' not found in df. Skipping target distribution.")
    else:
        y = df[target_col]
        print("\n--- Basic Statistics ---")
        print(y.describe())

        zero_cnt = (y == 0).sum()
        zero_ratio = zero_cnt / len(y) * 100
        print("\n--- Zero values ---")
        print("Zero count:", zero_cnt)
        print("Zero ratio:", f"{zero_ratio:.2%}")

        # 直方图
        print("\n--- 目标变量直方图 ---")
        plt.figure(figsize=(8, 5))
        plt.hist(y.dropna(), bins=50)
        plt.title(f"Target Distribution: {target_col}")
        plt.xlabel(target_col)
        plt.ylabel("Count")
        target_fig_path = FIG_DIR / "target_hist.png"
        clear_raw_csvs(FIG_DIR, patterns=["target_hist.png"])
        plt.savefig(target_fig_path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"Saved target histogram to: {target_fig_path}")

        # log1p 直方图
        plt.figure(figsize=(8, 5))
        plt.hist((y.dropna()).map(lambda v: 0 if v < 0 else v).pipe(lambda s: s.apply(lambda v: __import__("math").log1p(v))),
                 bins=50)
        plt.title(f"Target Distribution (log1p): {target_col}")
        plt.xlabel(f"log1p({target_col})")
        plt.ylabel("Count")
        target_log_fig_path = FIG_DIR / "target_hist_log1p.png"
        clear_raw_csvs(FIG_DIR, patterns=["target_hist_log1p.png"])
        plt.savefig(target_log_fig_path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"Saved log1p target histogram to: {target_log_fig_path}")

    df = df.copy()
    # 添加 log_sales 列
    df["log_sales"] = np.log1p(df["units_sold"])
    print("\nAdded 'log_sales' column to df.")
    print(df[["units_sold", "log_sales"]].head(5))
    print(df[["units_sold", "log_sales"]].tail(5))
    print("\n--- dtypes ---")
    for col in ["units_sold", "log_sales"]:
        print(f"{col:<25} {df[col].dtype}")

    return df

# 双变量分析 feature v.s target
# 观察关系形状
def plot_scatter_sample(df, x, y = "log_sales", sample_size=30000, title = None):
    d = df[[x, y]].dropna()
    if len(d) > sample_size:
        d = d.sample(sample_size, random_state=42)
    plt.figure(figsize=(8, 5))
    plt.scatter(d[x], d[y], alpha=0.5, s=10)
    plt.xlabel(x)
    plt.ylabel(y)
    plt.title(title or f"{y} vs {x} (sample scatter)")
    plt.tight_layout()
    scatter_fig_path = FIG_DIR / f"scatter_{x}_vs_{y}.png"
    clear_raw_csvs(FIG_DIR, patterns=[f"scatter_{x}_vs_{y}.png"])
    plt.savefig(scatter_fig_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved scatter plot to: {scatter_fig_path}")

def plot_hexbin(df, x, y="log_sales", gridsize=60, title=None):
    d = df[[x, y]].dropna()
    plt.figure(figsize=(8, 5))
    hb = plt.hexbin(d[x], d[y], gridsize=gridsize, mincnt=1)
    plt.colorbar(hb, label="count")
    plt.xlabel(x)
    plt.ylabel(y)
    plt.title(title or f"{y} vs {x} (hexbin)")
    plt.tight_layout()
    hexbin_fig_path = FIG_DIR / f"hexbin_{x}_vs_{y}.png"
    clear_raw_csvs(FIG_DIR, patterns=[f"hexbin_{x}_vs_{y}.png"])
    plt.savefig(hexbin_fig_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved hexbin plot to: {hexbin_fig_path}")

def plot_binned_mean(df, x, y="log_sales", bins=20, title=None, show_count=False):
    d = df[[x, y]].dropna().copy()

    # 分位数分箱：每箱数量更均衡（更稳）
    d["bin"] = pd.qcut(d[x], q=bins, duplicates="drop")

    g = d.groupby("bin", observed=True).agg(
        x_mid=(x, "median"),
        y_mean=(y, "mean"),
        y_median=(y, "median"),
        n=(y, "size"),
        y_std=(y, "std")
    ).reset_index()

    # 简单误差条：标准误
    g["y_se"] = g["y_std"] / np.sqrt(g["n"].clip(lower=1))

    plt.figure(figsize=(8, 5))
    plt.plot(g["x_mid"], g["y_mean"], marker="o", label="bin mean")
    plt.fill_between(g["x_mid"],
                     g["y_mean"] - 1.96 * g["y_se"],
                     g["y_mean"] + 1.96 * g["y_se"],
                     alpha=0.2, label="~95% CI")
    plt.xlabel(x)
    plt.ylabel(f"mean({y}) within bins")
    plt.title(title or f"Binned mean curve: {y} vs {x}")
    plt.tight_layout()
    plt.legend()
    binned_mean_fig_path = FIG_DIR / f"binned_mean_{x}_vs_{y}.png"
    clear_raw_csvs(FIG_DIR, patterns=[f"binned_mean_{x}_vs_{y}.png"])
    plt.savefig(binned_mean_fig_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved binned_mean plot to: {binned_mean_fig_path}")
    

    if show_count:
        for _, row in g.iterrows():
            plt.text(row["x_mid"], row["y_mean"], int(row["n"]), fontsize=8, alpha=0.7)

    plt.show()

    return g

# 类别变量与目标变量关系
def plot_category_mean_with_count(df, x, y="log_sales", sort=True, min_count=1):
    g = (
        df.groupby(x,observed = True)[y]
        .agg(mean="mean", median="median", count="size")
        .reset_index()
    )
    g = g[g["count"] >= min_count]
    if sort:
        g = g.sort_values("mean")

    fig, ax1 = plt.subplots(figsize=(10,4))
    ax1.plot(g[x].astype(str), g["mean"], marker="o", label="mean")
    ax1.plot(g[x].astype(str), g["median"], marker="x", label="median")
    ax1.set_ylabel(f"{y}")
    ax1.legend()
    ax1.tick_params(axis="x", rotation=45)

    ax2 = ax1.twinx()
    ax2.bar(g[x].astype(str), g["count"], alpha=0.2)
    ax2.set_ylabel("count")

    plt.title(f"{x}: mean/median({y}) with counts")
    plt.tight_layout()

    category_mean_count_fig_path = FIG_DIR / f"category_mean_count_{x}_vs_{y}.png"
    clear_raw_csvs(FIG_DIR, patterns=[f"category_mean_count_{x}_vs_{y}.png"])
    plt.savefig(category_mean_count_fig_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved category_mean_count plot to: {category_mean_count_fig_path}")
    return g

# def Dataset_bivariate_analysis(df: pd.DataFrame, target_col: str = "units_sold") -> None:



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
    df_after_log = Dataset_univariate_analysis(df_after_missing, target_col="units_sold")
    
    cont = {}
    continuous_cols = list("total_price base_price discount_ratio".split())
    for col in continuous_cols:
        plot_scatter_sample(df_after_log, x=col, y="log_sales", sample_size=30000)
        plot_hexbin(df_after_log, x=col, y="log_sales", gridsize=60)
        cont[col] = plot_binned_mean(df_after_log, x=col, y="log_sales", bins=15)

    cate = {}
    category_cols = list("store_id sku_id is_featured_sku is_display_sku is_discount_sku".split())
    for col in category_cols:
        cate[col] = plot_category_mean_with_count(df_after_log, x=col, y="log_sales", sort=True, min_count=50) 
if __name__ == "__main__":
    main()