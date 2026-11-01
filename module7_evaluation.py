"""
Module 7: Model Evaluation & Benchmarking
==========================================
Focus: Evaluating models using MAE, MSE, RMSE, R-Squared (R2 Score), and MAPE.
Selecting the champion model based on validation metrics.
Includes:
- Mathematical metrics calculation
- Residual diagnostics
- Model comparison leaderboard
- Automated champion model selector
"""

import math
from typing import List, Dict, Any, Tuple
import logging

logger = logging.getLogger("CarPriceAI.Module7")


class ModelEvaluator:
    """
    Evaluates regression model predictions and benchmarks multiple algorithms.
    """

    @staticmethod
    def calculate_mae(y_true: List[float], y_pred: List[float]) -> float:
        """Mean Absolute Error: (1/n) * sum(|y - y_hat|)"""
        n = len(y_true)
        if n == 0:
            return 0.0
        return sum(abs(y_true[i] - y_pred[i]) for i in range(n)) / n

    @staticmethod
    def calculate_mse(y_true: List[float], y_pred: List[float]) -> float:
        """Mean Squared Error: (1/n) * sum((y - y_hat)^2)"""
        n = len(y_true)
        if n == 0:
            return 0.0
        return sum((y_true[i] - y_pred[i]) ** 2 for i in range(n)) / n

    @classmethod
    def calculate_rmse(cls, y_true: List[float], y_pred: List[float]) -> float:
        """Root Mean Squared Error: sqrt(MSE)"""
        return math.sqrt(cls.calculate_mse(y_true, y_pred))

    @staticmethod
    def calculate_r2(y_true: List[float], y_pred: List[float]) -> float:
        """
        Coefficient of Determination (R^2 Score):
        R^2 = 1 - (SS_res / SS_tot)
        """
        n = len(y_true)
        if n < 2:
            return 0.0
        mean_y = sum(y_true) / n
        ss_tot = sum((y - mean_y) ** 2 for y in y_true)
        ss_res = sum((y_true[i] - y_pred[i]) ** 2 for i in range(n))

        if ss_tot <= 1e-9:
            return 0.0
        return max(-1.0, 1.0 - (ss_res / ss_tot))

    @staticmethod
    def calculate_mape(y_true: List[float], y_pred: List[float]) -> float:
        """Mean Absolute Percentage Error (MAPE) in percentage."""
        n = len(y_true)
        valid = [abs(y_true[i] - y_pred[i]) / y_true[i] for i in range(n) if y_true[i] > 0]
        if not valid:
            return 0.0
        return (sum(valid) / len(valid)) * 100.0

    @classmethod
    def evaluate_model(cls, model_name: str, y_true: List[float], y_pred: List[float]) -> Dict[str, Any]:
        """Generates a complete metric scorecard for a model."""
        mae = cls.calculate_mae(y_true, y_pred)
        mse = cls.calculate_mse(y_true, y_pred)
        rmse = cls.calculate_rmse(y_true, y_pred)
        r2 = cls.calculate_r2(y_true, y_pred)
        mape = cls.calculate_mape(y_true, y_pred)

        scorecard = {
            "model_name": model_name,
            "mae": round(mae, 2),
            "mse": round(mse, 2),
            "rmse": round(rmse, 2),
            "r2_score": round(r2, 4),
            "mape_percent": round(mape, 2)
        }
        logger.info("Evaluation [%s] -> R2: %.4f | RMSE: %.2f | MAE: %.2f", model_name, r2, rmse, mae)
        return scorecard

    @classmethod
    def select_champion_model(cls, scorecards: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Selects champion model maximizing R2 score while minimizing RMSE."""
        if not scorecards:
            return {}
        # Sort primarily by descending R2, secondarily by ascending RMSE
        sorted_models = sorted(scorecards, key=lambda x: (-x["r2_score"], x["rmse"]))
        champion = sorted_models[0]
        logger.info("Champion Model Selected: %s (R2: %.4f)", champion["model_name"], champion["r2_score"])
        return champion
