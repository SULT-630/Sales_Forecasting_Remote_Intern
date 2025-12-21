"""
Script: Train_Eval.py
Purpose: Entrance to run and evaluate different dataframes and models
Run: python src/sales_forecasting/Train_Eval.py
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.model_selection import train_test_split
from sales_forecasting.load_data import clear_raw_csvs
from sklearn.metrics import roc_curve
from sklearn.metrics import (
    accuracy_score, roc_auc_score, f1_score,
    precision_score, recall_score,
    mean_squared_error, mean_absolute_error, r2_score
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
FIG_DIR = PROJECT_ROOT / "artifacts" / "figures" / "Model_evaluation"
METRIC_DIR = PROJECT_ROOT / "artifacts" / "metrics" / "Model_evaluation"
FIG_DIR.mkdir(parents=True, exist_ok=True)
METRIC_DIR.mkdir(parents=True, exist_ok=True)

class DataProcessor:
    def __init__(self, target_col, id_col: str = "record_ID", test_size=0.2, random_state=42):
        self.target_col = target_col
        self.id_col = id_col
        self.test_size = test_size
        self.random_state = random_state

    def get_Xy(self, df: pd.DataFrame):
        drop_cols = [self.target_col]
        if self.id_col in df.columns:
            drop_cols.append(self.id_col)
        X = df.drop(columns=drop_cols)
        y = df[self.target_col]
        if self.id_col in df.columns:
            X.index = df.loc[X.index, self.id_col]
            y.index = X.index
            
        return X, y
    
    def split(self, df: pd.DataFrame):
        X, y = self.get_Xy(df)

        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=self.test_size,
            random_state=self.random_state
        )

        return X_train, X_test, y_train, y_test
    
class ModelRunner:
    def __init__(self, model):
        self.model = model

    def train(self, X_train, y_train):
        self.model.fit(X_train, y_train)

    def predict(self, X_test):
        if hasattr(self.model, "predict_proba"):
            return self.model.predict(X_test), self.model.predict_proba(X_test)
        else:
            return self.model.predict(X_test), None
        
    def build_dataframe(self, X_test, y_test, y_pred, Title, y_prob=None):
        df = X_test.copy()
        
        if hasattr(y_test, "reindex"):  # pandas Series/DataFrame
            y_true_aligned = y_test.reindex(df.index)
            # 如果 y_test 是 DataFrame，取第一列
            if isinstance(y_true_aligned, pd.DataFrame):
                y_true_aligned = y_true_aligned.iloc[:, 0]
            df["Y_true"] = y_true_aligned.values
        else:
            df["Y_true"] = y_test

        # 3) y_pred
        df["Y_pred"] = y_pred

        # 4) y_prob（仅在存在时写入）
        if y_prob is not None:
            # 二分类
            if getattr(y_prob, "ndim", 1) == 2 and y_prob.shape[1] == 2:
                df["Y_prob_pos"] = y_prob[:, 1]
            # 多分类：每列一个类别概率
            elif getattr(y_prob, "ndim", 1) == 2 and y_prob.shape[1] > 2:
                for k in range(y_prob.shape[1]):
                    df[f"Y_prob_class_{k}"] = y_prob[:, k]
            else:
                # 其他情况：直接存原值（不一定常见）
                df["Y_prob"] = y_prob

        clear_raw_csvs(METRIC_DIR, patterns=[f"{Title}_Compare.csv"])
        compare_path = METRIC_DIR / f"{Title}_Compare.csv"
        df.to_csv(compare_path, index=True)
        print(f"\nSaved {Title} Compare dataframe to: {compare_path}")

class Evaluator:
    def __init__(self, task_type):
        self.task_type = task_type
    
    def inverse_transform(self, y, transform_type: str):
        if transform_type == "log1p":
            return np.expm1(y)
        return y

    def evaluate(self, y_true, y_pred, Title, transform_type=None, y_prob=None):
        metrics = {}

        if self.task_type == "regression":
            if transform_type is not None:
                y_true = self.inverse_transform(y_true, transform_type=transform_type)
                y_pred = self.inverse_transform(y_pred, transform_type=transform_type)
            metrics["RMSE"] = np.sqrt(mean_squared_error(y_true, y_pred))
            metrics["MAE"] = mean_absolute_error(y_true, y_pred)
            metrics["R2"] = r2_score(y_true, y_pred)

        elif self.task_type == "binary":
            metrics["Accuracy"] = accuracy_score(y_true, y_pred)
            metrics["F1"] = f1_score(y_true, y_pred)
            metrics["Precision"] = precision_score(y_true, y_pred)
            metrics["Recall"] = recall_score(y_true, y_pred)

            if y_prob is not None:
                metrics["AUC"] = roc_auc_score(y_true, y_prob[:, 1])
        print(f"\nEvaluation Metrics for {Title}:")
        print(metrics)

        return metrics
    
class Visualizer:
    @staticmethod
    def plot_regression(y_true, y_pred, Title):
        plt.figure()
        plt.scatter(y_true, y_pred, alpha=0.5)
        plt.plot([y_true.min(), y_true.max()],
                 [y_true.min(), y_true.max()], 'r--')
        plt.xlabel("True")
        plt.ylabel("Predicted")
        plt.title("Prediction vs True")
        plt.grid(True)
        plt.tight_layout()
        Regression_fig_path = FIG_DIR / f"{Title}_Regression_curve.png"
        clear_raw_csvs(FIG_DIR, patterns=[f"{Title}_Regression_curve.png"])
        plt.savefig(Regression_fig_path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"Saved {Title} Regression curve to: {Regression_fig_path}")

    @staticmethod
    def plot_roc(y_true, y_prob, Title):
        fpr, tpr, _ = roc_curve(y_true, y_prob[:, 1])
        plt.figure()
        plt.plot(fpr, tpr)
        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlabel("FPR")
        plt.ylabel("TPR")
        plt.title("ROC Curve")
        plt.grid(True)
        plt.tight_layout()
        ROC_fig_path = FIG_DIR / f"{Title}_ROC_curve.png"
        clear_raw_csvs(FIG_DIR, patterns=[f"{Title}_ROC_curve.png"])
        plt.savefig(ROC_fig_path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"Saved {Title} ROC curve to: {ROC_fig_path}")
