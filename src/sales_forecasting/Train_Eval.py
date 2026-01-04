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
from sales_forecasting.data_preprocess import preprocess_data
from sales_forecasting.load_data import clear_raw_csvs
from sklearn.metrics import roc_curve
from sklearn.metrics import (
    accuracy_score, roc_auc_score, f1_score,
    precision_score, recall_score,
    mean_squared_error, mean_absolute_error, r2_score
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
FIG_DIR = PROJECT_ROOT / "artifacts" / "figures" / "Model_evaluation_lastweek"
METRIC_DIR = PROJECT_ROOT / "artifacts" / "metrics" / "Model_evaluation_lastweek"
FIG_DIR.mkdir(parents=True, exist_ok=True)
METRIC_DIR.mkdir(parents=True, exist_ok=True)

class DataProcessor:
    def __init__(self, target_col, id_col: str = "record_ID", time_col: str = "week", test_size=0.2):
        self.target_col = target_col
        self.id_col = id_col
        self.time_col = time_col
        self.test_size = test_size

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
        if self.time_col not in df.columns:
            raise ValueError(f"df 中找不到时间列 time_col='{self.time_col}'")

        df_sorted = df.sort_values(by=[self.time_col, 'sku_id']).reset_index(drop=True)

        n = len(df_sorted)
        n_test = int(round(n * self.test_size))
        n_test = max(1, n_test)          # 至少 1 条 test
        n_train = n - n_test
        if n_train <= 0:
            raise ValueError("数据量太小或 test_size 太大，导致训练集为空。")

        train_df = df_sorted.iloc[:n_train].copy()
        test_df  = df_sorted.iloc[n_train:].copy()

        X_train, y_train = self.get_Xy(train_df)
        X_test,  y_test  = self.get_Xy(test_df)
        # print(X_test.tail())
        return X_train, X_test, y_train, y_test
    
    def split_lastweek(self, df: pd.DataFrame):
        if self.time_col not in df.columns:
            raise ValueError(f"df 中找不到时间列 time_col='{self.time_col}'")

        df_sorted = df.sort_values(by=[self.time_col, 'sku_id']).reset_index(drop=True)

        # 关键：确保时间列可比较（如果已经是 datetime，这句不会破坏）
        time_series = pd.to_datetime(df_sorted[self.time_col], errors="coerce")
        if time_series.isna().any():
            bad = df_sorted.loc[time_series.isna(), self.time_col].head(5).tolist()
            raise ValueError(f"时间列 {self.time_col} 存在无法解析的值(示例前5个):{bad}")

        max_time = time_series.max()

        test_mask = time_series.eq(max_time)
        test_df = df_sorted.loc[test_mask].copy()
        train_df = df_sorted.loc[~test_mask].copy()

        if len(test_df) == 0:
            raise ValueError("未能切出测试集(max_time 对应的数据为空)。")
        if len(train_df) == 0:
            raise ValueError("训练集为空：所有数据都在最后一个时间点上。")

        X_train, y_train = self.get_Xy(train_df)
        X_test,  y_test  = self.get_Xy(test_df)
        print(X_test.head())
        print(X_test[self.time_col].unique())
        return X_train, X_test, y_train, y_test

    
class ModelRunner:
    def __init__(self, model, best_n=None):
        self.model = model
        self.best_n = best_n

    def train(self, X_train, y_train):
        X=X_train.copy()
        X["log_sales"] = y_train.values
        df = X.sort_values(by=['week'])#.reset_index(drop=True)
        unique_weeks = df['week'].drop_duplicates().sort_values()
        valid_weeks = unique_weeks.iloc[-3:]

        is_valid = df['week'].isin(valid_weeks)
        df_valid = df[is_valid]
        df_train = df[~is_valid]
        X_train_new = df_train.drop(columns="log_sales")
        y_train_new = df_train["log_sales"]
        X_valid = df_valid.drop(columns="log_sales")
        y_valid = df_valid["log_sales"]

        if "week" in X_train_new.columns:
            X_train_new_no_week = X_train_new.drop(columns=["week"])
        if "week" in X_valid.columns:
            X_valid_no_week = X_valid.drop(columns=["week"])

        print("--- Training on data: ---")
        print(X_train_new.head())
        print("--- Training target: ---")
        print(y_train_new.head())
        self.model.fit(X_train_new_no_week, y_train_new,eval_set=[(X_valid_no_week, y_valid)], verbose=100)
        print("--- Finding best iteration: ---")
        print("Best iteration:", self.model.best_iteration)
        self.best_n = self.model.best_iteration+1
        
        return X_train_new, y_train_new, X_valid, y_valid

    def predict(self, X_test):
        X = X_test.copy()
        if "week" in X.columns:
            X = X.drop(columns=["week"])
        print("--- Predicting on data: ---")
        print(X.head())
        if hasattr(self.model, "predict_proba"):
            return self.model.predict(X, iteration_range=(0, self.best_n)), self.model.predict_proba(X)
        else:
            return self.model.predict(X, iteration_range=(0, self.best_n)), None
    
    # def rolling_predict(self, X_train, X_test, y_train, target_col = 'log_sales', time_col='week', group_col='sku_id'): # 检查group col还应该包含哪些
    #     train_X = X_train.copy()
    #     test_X = X_test.copy()
    #     train_y = y_train.copy()
    #     train_df = train_X.copy()
    #     train_df[target_col] = train_y

    #     test_df = test_X.copy()
    #     test_df[target_col] = np.nan
    #     full_df = pd.concat([train_df, test_df], axis=0)

    #     n_train = len(train_df)
    #     n_test  = len(test_df)

    #     train_feat_df = preprocess_data(train_df.copy())  
    #     feature_cols = [c for c in train_feat_df.columns if c != target_col]
    #     if time_col in feature_cols:
    #         feature_cols.remove(time_col)
        
    #     preds = np.zeros(n_test, dtype=float)

    #     for i in range(n_test):
    #         pos = n_train + i  # full_df 里当前要预测的行位置（用 iloc，不受 record_ID index 影响）

    #         # 取到 “截至当前行” 的子表
    #         hist_df = full_df.iloc[:pos + 1].copy()

    #         # 构造动态特征（lag/rolling/ewma等）
    #         feat_hist_df = preprocess_data(hist_df)

    #         # 取当前行的特征 X_curr，并对齐训练列 应该去除week
    #         x_curr = feat_hist_df.iloc[-1][feature_cols]

    #         x_curr = x_curr.reindex(feature_cols)

    #         x_curr = x_curr.fillna(0.0)

    #         # 预测
    #         y_pred = float(self.model.predict(x_curr.to_frame().T)[0])
    #         preds[i] = y_pred

    #         # 把预测写回 full_df（这样下一步 rolling/lag 会用到它）
    #         full_df.iat[pos, full_df.columns.get_loc(target_col)] = y_pred

    #     # 输出：用 test 的 index（record_ID）对齐
    #     pred_series = pd.Series(preds, index=test_df.index, name=f"pred_{target_col}")

    #     return np.array(pred_series), full_df
    def rolling_predict(
        self, X_train, X_test, y_train,
        target_col='log_sales', time_col='week', group_col='sku_id',
        debug=True, check_every=50, sample_steps=5
    ):

        train_X = X_train.copy()
        test_X = X_test.copy()
        train_y = y_train.copy()

        train_df = train_X.copy()
        train_df[target_col] = train_y

        test_df = test_X.copy()
        test_df[target_col] = np.nan

        full_df = pd.concat([train_df, test_df], axis=0)

        n_train = len(train_df)
        n_test  = len(test_df)

        # ====== 0) 基础检查：必须包含 group/time ======
        assert group_col in full_df.columns, f"full_df 缺少 group_col={group_col}"
        assert time_col in full_df.columns, f"full_df 缺少 time_col={time_col}"

        # ====== 1) 顺序检查：你现在 rolling 是按行号滚动，所以必须先排序 ======
        # 如果你“不想排序”，那就至少检查它已排序；这里我默认强制排序更安全
        # 注意：排序会改变 train/test 的“行位置”，所以要记录 test 原 index 的位置映射
        full_df["_is_train"] = [1]*n_train + [0]*n_test
        full_df["_orig_index"] = full_df.index

        full_df = full_df.sort_values([group_col, time_col, "_is_train"]).reset_index(drop=True)

        # 重新定位 train/test 行（按 _is_train）
        train_mask = full_df["_is_train"].values == 1
        test_mask  = full_df["_is_train"].values == 0

        # 训练特征列来自 preprocess(train_df) 的列集合（但现在 train_df 已被排序重置了）
        train_df_sorted = full_df.loc[train_mask].copy()
        train_feat_df = preprocess_data(train_df_sorted.copy(), target_col=target_col)

        feature_cols = [c for c in train_feat_df.columns if c != target_col]
        if time_col in feature_cols:
            feature_cols.remove(time_col)

        print(feature_cols)

        preds = np.zeros(n_test, dtype=float)

        # ====== debug 采样记录 ======
        debug_rows = []
        sample_points = set(np.linspace(0, n_test-1, num=min(sample_steps, n_test), dtype=int)) if debug else set()

        # ====== 2) 逐步滚动 ======
        test_positions = np.where(test_mask)[0]  # full_df 中 test 行的位置（排序后）

        for i, pos in enumerate(test_positions):
            print(f"Rolling predict step {i+1}/{n_test} (full_df pos={pos})")
            # 截止到当前行（包含当前行）
            hist_df = full_df.iloc[:pos + 1].copy()

            # —— 核心：动态特征
            feat_hist_df = preprocess_data(hist_df, target_col=target_col)
            feat_hist_df = feat_hist_df.drop(columns=[time_col], errors='ignore')

            # 训练时的列必须都在，否则列漂移
            missing = set(feature_cols) - set(feat_hist_df.columns)
            extra   = set(feat_hist_df.columns) - set(feature_cols) - {target_col}
            # 当前行特征对齐
            x_curr = feat_hist_df.iloc[[-1]].reindex(columns=feature_cols)
            assert x_curr.shape[0] == 1, f"x_curr should be 1 row, got {x_curr.shape}"
            x_curr.drop(labels=['week','_is_train','_orig_index'], errors='ignore', axis=1, inplace=True)
            if debug and (missing or extra):
                print(f"[WARN] step={i} feature drift: missing={list(missing)[:5]} extra={list(extra)[:5]}")


            # 打印datetime行信息
            # dt_cols = x_curr.select_dtypes(include=["datetime64[ns]", "datetimetz"]).columns
            # print("datetime cols:", dt_cols.tolist())


            # ====== 3) 核心检查：当前行特征不能含 target_col 的任何“未来信息” ======
            # 3.1 当前行 target 是 NaN（未写回前应该是 NaN）
            # 注意：你是预测后才写回，所以这里应该仍为 NaN
            if debug:
                # 如果这里不是 NaN，说明这个 pos 在历史里已经被填过（或 preprocess 改写了）
                if not pd.isna(full_df.iloc[pos][target_col]):
                    print(f"[WARN] step={i} 当前行在预测前 target_col 已非 NaN, 可能顺序/重复写回有问题")

            # 3.2 x_curr 不应有 inf
            if debug:
                # 1) 强制数值化
                x_num = x_curr.apply(pd.to_numeric, errors="coerce")

                # 2) 关键：把 pandas NA / 可空类型，统一变成 float numpy
                arr = x_num.to_numpy(dtype="float64", na_value=np.nan)

                # print("x_num dtypes:\n", x_num.dtypes)
                # print("arr dtype:", arr.dtype)

                has_inf = np.isinf(arr).any()
                has_nan = np.isnan(arr).any()

                if has_inf or has_nan:
                    # 找到具体哪几列出问题
                    bad = pd.DataFrame({
                        "nan_cnt": np.isnan(arr).sum(axis=0),
                        "inf_cnt": np.isinf(arr).sum(axis=0),
                    }, index=x_num.columns)
                    bad = bad[(bad["nan_cnt"] > 0) | (bad["inf_cnt"] > 0)].sort_values(["inf_cnt","nan_cnt"], ascending=False)
                    print("bad cols:\n", bad)

                    raise ValueError(f"step={i} x_curr 含 inf/nan, 可能写回逻辑有问题")

            # —— 预测
            y_pred = float(self.model.predict(x_curr)[0])
            preds[i] = y_pred

            # 写回（供下一步 lag/rolling 使用）
            full_df.iat[pos, full_df.columns.get_loc(target_col)] = y_pred

            # ====== 4) 目的验证：写回是否真的“影响下一步特征” ======
            # 简单做法：在采样点，比较“写回前后”下一行（同组下一个时间点）某些特征是否变化
            if debug and i in sample_points:
                g = full_df.iloc[pos][group_col]
                t = full_df.iloc[pos][time_col]
                x_row = x_curr.iloc[0]

                debug_rows.append({
                    "step": i,
                    "pos": int(pos),
                    "group": g,
                    "time": t,
                    "y_pred": y_pred,
                    "x_curr_nonzero": int((x_row != 0).sum()),
                    "x_curr_nan": int(pd.isna(x_row).sum()),
                })

            # ====== 5) 每隔 check_every 步做一次结构性检查 ======
            if debug and (i % check_every == 0):
                # 检查 hist_df 内部：同一 group 的 time 是否单调
                sub = hist_df[[group_col, time_col]]
                # 只抽最近一小段避免太慢
                tail = sub.tail(1000)
                bad = tail.groupby(group_col, observed=False)[time_col].apply(lambda s: not s.is_monotonic_increasing)
                if bad.any():
                    bad_groups = bad[bad].index.tolist()[:5]
                    raise AssertionError(f"发现 group 内 time 非递增（示例 {bad_groups}），滚动顺序不可靠，请先排序或修正数据。")

        # ====== 6) 输出对齐回 test 原 index ======
        # 注意：我们排序 reset_index 了，所以要用 _orig_index 找回 test 原 index 顺序
        test_orig_index = full_df.loc[test_mask, "_orig_index"].values
        pred_series = pd.Series(preds, index=test_orig_index, name=f"pred_{target_col}")
        print("Rolling predict completed.")
        # 清理辅助列
        full_df = full_df.drop(columns=["_is_train", "_orig_index"])

        if debug:
            debug_df = pd.DataFrame(debug_rows)
            print("=== rolling debug sample ===")
            print(debug_df)

        return pred_series.values, full_df

        
    def build_dataframe(self, X_test, y_test, y_pred, Title, y_prob=None) -> pd.DataFrame:
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
        return df

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
    def get_xgb_feature_importance(self, model, feature_names, Title, importance_type="gain"):
        """
        importance_type:
            - 'gain'   : 分裂带来的平均增益（最常用、最有意义）
            - 'weight' : 特征被用来分裂的次数
            - 'cover'  : 覆盖的样本数
        """
        booster = model.get_booster()
        score = booster.get_score(importance_type=importance_type)

        fi = (
            pd.DataFrame(
                score.items(),
                columns=["feature", "importance"]
            )
            .assign(feature=lambda df: df["feature"].map(
                dict(zip(booster.feature_names, feature_names))
            ))
            .sort_values("importance", ascending=False)
            .reset_index(drop=True)
        )
        clear_raw_csvs(METRIC_DIR, patterns=[f"{Title}_Feature_Importance.csv"])
        path = METRIC_DIR / f"{Title}_Feature_Importance.csv"
        fi.to_csv(path, index=True)
        print(f"\nSaved {Title} Feature importance dataframe to: {path}")
        return fi
    

def GROUPED_MAPE(group: pd.DataFrame, true_col: str, pred_col: str) -> float:
    y_true = group[true_col].to_numpy()
    y_pred = group[pred_col].to_numpy()
    mask = (y_true != 0)
    if mask.sum() == 0:
        return np.nan
    mape = np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100
    return mape

def MAPE(
    df: pd.DataFrame,
    Title: str,
    week_col: str,
    sku_col: str,
    y_true_col: str,
    y_pred_col: str,
    sort_week: bool = True
):

    # --- Step 1: (week, sku) 级别 MAPE ---
    mape_week_sku = (
        df.groupby([week_col, sku_col], dropna=False,observed=False)
          .apply(lambda g: GROUPED_MAPE(g, y_true_col, y_pred_col),include_groups=False)
          .reset_index(name="mape_week_sku")   # <-- 保证结果一定是三列
    )

    # --- Step 2: week 级别 MAPE（对 sku 平均）---
    mape_by_week = (
        mape_week_sku.groupby(week_col, as_index=False, observed=False)["mape_week_sku"]
                    .mean()
                    .rename(columns={"mape_week_sku": "mape_by_week"})
    )

    if sort_week:
        mape_by_week = mape_by_week.sort_values(by=week_col).reset_index(drop=True)

    # # --- Step 3: 画时间序列图 ---
    # plt.figure()
    # plt.plot(mape_by_week[week_col], mape_by_week["mape_by_week"], marker="o")
    # plt.xlabel(week_col)
    # plt.ylabel("MAPE")
    # plt.title(f"{Title} MAPE over time (averaged across sku)")
    # plt.xticks(rotation=45)
    # plt.tight_layout()
    # plt.grid(True)
    # MAPE_fig_path = FIG_DIR / f"{Title}_MAPE_Time_curve.png"
    # clear_raw_csvs(FIG_DIR, patterns=[f"{Title}_MAPE_Time_curve.png"])
    # plt.savefig(MAPE_fig_path, dpi=150, bbox_inches="tight")
    # plt.close()
    # print(f"Saved {Title} MAPE time sequence curve to: {MAPE_fig_path}")

    # --- Step 4: 总 MAPE（对 week 平均）---
    overall_mape = mape_by_week["mape_by_week"].mean()

    return mape_week_sku, mape_by_week, overall_mape
                   
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