"""
Script: experiment.py
Purpose: Use src classes to train and evaluate for different dataframe and modelss
Run: python src/sales_forecasting/experiments.py
"""

from sales_forecasting.Train_Eval import ModelRunner
from sales_forecasting.Train_Eval import Evaluator
from sales_forecasting.Train_Eval import Visualizer

class Experiment:
    def __init__(self, df, target_col, model, task_type):
        self.df = df
        self.target_col = target_col
        self.model = model
        self.task_type = task_type

    def run(self, X_test, X_train, y_test, y_train, Title):

        runner = ModelRunner(self.model)
        runner.train(X_train, y_train)

        y_pred, y_prob = runner.predict(X_test)
        runner.build_dataframe(y_test, y_pred, Title, y_prob)
        evaluator = Evaluator(self.task_type)
        metrics = evaluator.evaluate(y_test, y_pred, Title, y_prob)

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