"""
Script: Train_Eval.py
Purpose: Entrance to run and evaluate different dataframes and models
Run: python src/sales_forecasting/Train_Eval.py
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
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
        
    def build_dataframe(X_test, y_test, y_pred, Title, y_prob=None):
        df = X_test.copy()
        df["Y_true"] = y_test.values
        df["Y_pred"] = y_pred
        if y_prob is not None:
            df["y_pred_prob"] = y_prob[:,1]

        clear_raw_csvs(METRIC_DIR, patterns=[f"{Title}_Compare.csv"])
        compare_path = METRIC_DIR / f"{Title}_Compare.csv"
        df.to_csv(compare_path, index=True)
        print(f"\nSaved {Title} Compare dataframe to: {compare_path}")

class Evaluator:
    def __init__(self, task_type):
        self.task_type = task_type

    def evaluate(self, y_true, y_pred, Title, y_prob=None):
        metrics = {}

        if self.task_type == "regression":
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

        clear_raw_csvs(METRIC_DIR, patterns=[f"{Title}_Eval_index.csv"])
        metrics_path = METRIC_DIR / f"{Title}_Eval_index.csv"
        metrics.to_csv(metrics_path, index=True)
        print(f"\nSaved {Title} Evaluation index to: {metrics_path}")

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
 