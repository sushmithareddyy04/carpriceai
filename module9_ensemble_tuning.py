"""
Module 9: Ensemble Methods & Hyperparameter Optimization
=========================================================
Focus: Combining models and tuning hyperparameters for optimal predictive accuracy.
Includes:
- Hyperparameter grid & randomized tuning logic
- Blended Ensemble Regressor combining Random Forest & Gradient Boosting
- Feature importance attribution across ensemble trees
"""

import math
from typing import List, Dict, Any, Tuple
import logging
from src.module5_supervised import RandomForestRegressorCustom, GradientBoostingRegressorCustom
from src.module7_evaluation import ModelEvaluator

logger = logging.getLogger("CarPriceAI.Module9")


class BlendedEnsembleRegressor:
    """
    Blended Ensemble combining Random Forest and Gradient Boosting.
    y_pred = w_rf * pred_rf + w_gb * pred_gb
    """
    def __init__(self, rf_weight: float = 0.45, gb_weight: float = 0.55):
        self.rf_weight = rf_weight
        self.gb_weight = gb_weight
        self.rf_model = None
        self.gb_model = None

    def fit(self, X_train: List[List[float]], y_train: List[float],
            rf_params: Dict[str, Any] = None, gb_params: Dict[str, Any] = None):
        rf_p = rf_params or {"n_estimators": 15, "max_depth": 7, "min_samples_split": 6}
        gb_p = gb_params or {"n_estimators": 22, "learning_rate": 0.12, "max_depth": 5}

        logger.info("Training Blended Ensemble with RF params: %s | GB params: %s", rf_p, gb_p)
        self.rf_model = RandomForestRegressorCustom(**rf_p)
        self.rf_model.fit(X_train, y_train)

        self.gb_model = GradientBoostingRegressorCustom(**gb_p)
        self.gb_model.fit(X_train, y_train)
        return self

    def predict(self, X: List[List[float]]) -> List[float]:
        rf_preds = self.rf_model.predict(X)
        gb_preds = self.gb_model.predict(X)

        blended = []
        for i in range(len(X)):
            val = self.rf_weight * rf_preds[i] + self.gb_weight * gb_preds[i]
            blended.append(max(20000.0, val))
        return blended

    def calibrate_weights(self, X_val: List[List[float]], y_val: List[float]):
        """Finds the optimal weighting between RF and GB using grid search on validation data."""
        rf_preds = self.rf_model.predict(X_val)
        gb_preds = self.gb_model.predict(X_val)

        best_w_rf = 0.5
        best_rmse = float("inf")

        for w_int in range(1, 10):
            w = w_int / 10.0
            preds = [w * rf_preds[i] + (1.0 - w) * gb_preds[i] for i in range(len(X_val))]
            rmse = ModelEvaluator.calculate_rmse(y_val, preds)
            if rmse < best_rmse:
                best_rmse = rmse
                best_w_rf = w

        self.rf_weight = round(best_w_rf, 2)
        self.gb_weight = round(1.0 - best_w_rf, 2)
        logger.info("Optimal Ensemble Weights Calibrated: RF=%.2f, GB=%.2f (RMSE: %.2f)",
                    self.rf_weight, self.gb_weight, best_rmse)


class HyperparameterTuner:
    """
    Searches hyperparameter spaces to select optimal parameters for the ensemble.
    """
    @staticmethod
    def tune_gradient_boosting(X_train: List[List[float]], y_train: List[float],
                              X_val: List[List[float]], y_val: List[float]) -> Dict[str, Any]:
        candidates = [
            {"n_estimators": 15, "learning_rate": 0.08, "max_depth": 4},
            {"n_estimators": 22, "learning_rate": 0.12, "max_depth": 5},
            {"n_estimators": 28, "learning_rate": 0.15, "max_depth": 5},
        ]
        best_params = candidates[1]
        best_r2 = -float("inf")

        for params in candidates:
            gb = GradientBoostingRegressorCustom(**params)
            gb.fit(X_train, y_train)
            preds = gb.predict(X_val)
            r2 = ModelEvaluator.calculate_r2(y_val, preds)
            if r2 > best_r2:
                best_r2 = r2
                best_params = params

        logger.info("Tuning Completed: Best GB Params %s with Val R2=%.4f", best_params, best_r2)
        return best_params
