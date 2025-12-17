"""
Script: load_data.py
Purpose: Connect to Kaggle, download dataset, and load raw data into DataFrame
Run: python scripts/load_data.py
"""

from pathlib import Path
from dotenv import load_dotenv
import kagglehub
import pandas as pd

# 1. 项目根目录
PROJECT_ROOT = Path(__file__).resolve().parents[2]

# 2. 加载 .env 中的环境变量（KAGGLE_API_TOKEN）
load_dotenv()

# 3. data/raw 目录
RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"
RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)

def clear_raw_csvs(raw_dir: Path, patterns: list[str], keep_files: list[str] | None = None) -> None:
    """
    clear out old CSV files in raw data directory
    """
    if keep_files is None:
        keep_files = [".gitkeep"]

    for pat in patterns:
        for f in raw_dir.glob(pat):
            if f.name in keep_files:
                continue
            try:
                f.unlink()  # 删除文件
                print(f"Removed old raw file: {f}")
            except Exception as e:
                print(f"Failed to remove {f}: {e}")

def load_kaggle_dataset() -> pd.DataFrame:
    """
    Download Kaggle dataset via kagglehub and load CSV into DataFrame.
    """

    # 清理旧的 CSV 文件
    clear_raw_csvs(RAW_DATA_DIR, patterns=["*.csv"])
    
    # 4. 下载数据集
    dataset_path = kagglehub.dataset_download(
        "aswathrao/demand-forecasting"
    )

    print("Dataset downloaded to:", dataset_path)

    dataset_path = Path(dataset_path)

    # 5. 找到所有CSV文件
    csv_files = list(dataset_path.glob("*.csv"))
    if not csv_files:
        raise FileNotFoundError("No CSV file found in dataset.")
    
    # 6. 列出所有CSV文件并选择训练文件
    print("Found CSV files:")
    for f in csv_files:
        print(" -", f.name)

    csv_path = csv_files[2]
    print("Using file:", csv_path.name)

    # 7. 读成 DataFrame
    df = pd.read_csv(csv_path)

    # 8. 复制一份到 data/raw（项目规范）
    output_path = RAW_DATA_DIR / csv_path.name
    df.to_csv(output_path, index=False)

    print("Raw data saved to:", output_path)

    return df


def main():
    df = load_kaggle_dataset()
    print("Data shape:", df.shape)
    print(df.head())


if __name__ == "__main__":
    main()
