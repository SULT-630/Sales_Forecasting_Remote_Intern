"""
Script: experiment.py
Purpose: Use src classes to train and evaluate for different dataframe and modelss
Run: python src/sales_forecasting/experiments.py
"""
from copy import deepcopy
import pandas as pd
import numpy as np
from sales_forecasting.Train_Eval import ModelRunner
from sales_forecasting.Train_Eval import Evaluator
from sales_forecasting.Train_Eval import MAPE
from sales_forecasting.Train_Eval import Visualizer
from sales_forecasting.Train_Eval import DataProcessor
from sklearn.model_selection import KFold

class Experiment:
    def __init__(self, df, target_col, model, task_type, Title):
        self.df = df
        self.target_col = target_col
        self.model = model
        self.task_type = task_type
        self.Title = Title

    def run(self, X_train, X_test, y_train, y_test, Title,transform_type=None):
        # processor = DataProcessor(self.target_col)
        # X_train, X_test, y_train, y_test = processor.split(self.df)

        runner = ModelRunner(self.model)
        runner.train(X_train, y_train)

        # y_pred, full_df = runner.rolling_predict(X_train,X_test, y_train)
        y_pred, y_prob = runner.predict(X_test)
        Compare = runner.build_dataframe(X_test, y_test, y_pred, Title)
        evaluator = Evaluator(self.task_type)
        feature_names = X_test.columns
        feature_names = feature_names.drop("week", errors="ignore")
        fi = evaluator.get_xgb_feature_importance(self.model, feature_names, Title)
        metrics = evaluator.evaluate(y_test, y_pred, Title, transform_type)
        
        mape_week_sku, mape_by_week, overall_mape = MAPE(
            df=Compare,
            week_col="week",
            Title=Title,
            sku_col="sku_id",
            y_true_col="Y_true",
            y_pred_col="Y_pred",
        )
        print(f"--- Overall MAPE (by week & sku) :  ---")
        print(f"\n{overall_mape:.4f}")
        print(f"--- MAPE by week : ---")
        print(f"\n{mape_by_week}")
        print(f"--- MAPE by (week, sku) top 10: ---")
        print(
            mape_week_sku
            .sort_values("mape_week_sku", ascending=False)
            .head(10)
        )
        print(f"--- Feature importance: ---")
        print(fi)

        if self.task_type == "regression":
            Visualizer.plot_regression(y_test, y_pred, Title)
        else:
            if y_prob is not None:
                Visualizer.plot_roc(y_test, y_prob, Title)

        return {
            "metrics": metrics,
            "y_pred": y_pred,
            "y_prob": y_prob
        }

    # def run_kfold(self, Title, transform_type=None, n_splits=5, shuffle=True, random_state=42):

    #     processor = DataProcessor(self.target_col)
    #     X, y = processor.get_Xy(self.df)  

    #     kf = KFold(n_splits=n_splits, shuffle=shuffle, random_state=random_state)

    #     fold_metrics = []
    #     fold_train_metrics = []
    #     all_X_val = []
    #     all_y_val = []
    #     all_pred_val = []
    #     all_prob_val = []

    #     for fold, (train_idx, val_idx) in enumerate(kf.split(X), start=1):
    #         X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
    #         y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

           
    #         model_fold = deepcopy(self.model)

    #         runner = ModelRunner(model_fold)
    #         runner.train(X_train, y_train)

    #         y_pred_val, y_prob_val = runner.predict(X_val)
    #         y_pred_train, y_prob_train = runner.predict(X_train)

    #         evaluator = Evaluator(self.task_type)
    #         metrics_val = evaluator.evaluate(y_val, y_pred_val, f"{Title}_fold{fold}", transform_type, y_prob_val)
    #         metrics_train = evaluator.evaluate(y_train, y_pred_train, f"{Title}_fold{fold}_train", transform_type, y_prob_train)

    #         fold_metrics.append(metrics_val)
    #         fold_train_metrics.append(metrics_train)
            
    #         all_X_val.append(X_val)
    #         all_y_val.append(y_val)
    #         all_pred_val.append(pd.Series(y_pred_val, index=X_val.index, name="Y_pred"))
    #         if y_prob_val is not None:
    #             all_prob_val.append((X_val.index, y_prob_val))

    #     X_test_all = pd.concat(all_X_val, axis=0)
    #     y_test_all = pd.concat(all_y_val, axis=0)

    #     # y_pred 拼成一个 Series，再按 X_test_all 的 index 对齐成 ndarray
    #     y_pred_all_series = pd.concat(all_pred_val, axis=0)
    #     y_pred_all = y_pred_all_series.reindex(X_test_all.index).values
    #     # 按照原来index排序
    #     X_test_all = X_test_all.sort_index()
    #     y_test_all = y_test_all.loc[X_test_all.index]
    #     y_pred_all = pd.Series(y_pred_all,index=y_test_all.index).loc[X_test_all.index].values
    #     y_prob_all = None
    #     runner.build_dataframe(X_test_all, y_test_all, y_pred_all, Title, y_prob=y_prob_all)

    #     def summarize(metrics_list):
    #         keys = metrics_list[0].keys()
    #         mean = {k: float(np.mean([m[k] for m in metrics_list])) for k in keys}
    #         std  = {k: float(np.std([m[k] for m in metrics_list], ddof=1)) for k in keys}  # ddof=1 是样本标准差
    #         return mean, std

    #     cv_mean, cv_std = summarize(fold_metrics)
    #     train_mean, train_std = summarize(fold_train_metrics)

    #     print("\n===== CV Summary (Validation) =====")
    #     print("mean:", cv_mean)
    #     print("std :", cv_std)

    #     print("\n===== CV Summary (Train) =====")
    #     print("mean:", train_mean)
    #     print("std :", train_std)

    #     return {
    #         "cv_mean": cv_mean,
    #         "cv_std": cv_std,
    #         "train_mean": train_mean,
    #         "train_std": train_std,
    #         "fold_metrics": fold_metrics
    #     }