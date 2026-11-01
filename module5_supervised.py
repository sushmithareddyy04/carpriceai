"""
Module 5: Supervised Machine Learning
======================================
Focus: Training multiple regression models:
1. Linear Regression (Multiple OLS with Ridge Regularization)
2. Decision Tree Regressor (Recursive MSE reduction)
3. Random Forest Regressor (Ensemble Bagging)
4. Gradient Boosting Regressor (Sequential Gradient Descent Boosting)
"""

import math
import random
from typing import List, Dict, Any, Optional, Tuple
import logging

logger = logging.getLogger("CarPriceAI.Module5")


class BaseRegressor:
    """Base interface for all supervised regression algorithms."""
    def fit(self, X: List[List[float]], y: List[float]):
        raise NotImplementedError

    def predict(self, X: List[List[float]]) -> List[float]:
        raise NotImplementedError


class MultipleLinearRegression(BaseRegressor):
    """
    Multiple Linear Regression fitted via Gradient Descent with L2 Ridge Regularization.
    Objective: min_w ||Xw - y||^2 + lambda * ||w||^2
    """
    def __init__(self, learning_rate: float = 0.05, iterations: int = 1200, l2_penalty: float = 0.01):
        self.lr = learning_rate
        self.iterations = iterations
        self.l2_penalty = l2_penalty
        self.weights = []
        self.bias = 0.0

    def fit(self, X: List[List[float]], y: List[float]):
        n_samples = len(X)
        n_features = len(X[0])
        self.weights = [0.0] * n_features
        self.bias = sum(y) / n_samples

        for _ in range(self.iterations):
            # Compute predictions
            preds = [
                sum(X[i][j] * self.weights[j] for j in range(n_features)) + self.bias
                for i in range(n_samples)
            ]
            # Compute errors
            errors = [preds[i] - y[i] for i in range(n_samples)]

            # Gradients
            for j in range(n_features):
                grad_w = (sum(errors[i] * X[i][j] for i in range(n_samples)) / n_samples) + (self.l2_penalty * self.weights[j])
                self.weights[j] -= self.lr * grad_w

            grad_b = sum(errors) / n_samples
            self.bias -= self.lr * grad_b
        return self

    def predict(self, X: List[List[float]]) -> List[float]:
        n_features = len(self.weights)
        preds = []
        for row in X:
            val = sum(row[j] * self.weights[j] for j in range(n_features)) + self.bias
            preds.append(max(20000.0, val))
        return preds


class TreeNode:
    """Single node in a regression decision tree."""
    def __init__(self, feature_idx=None, threshold=None, left=None, right=None, value=None):
        self.feature_idx = feature_idx
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value

    @property
    def is_leaf(self):
        return self.value is not None


class DecisionTreeRegressorCustom(BaseRegressor):
    """
    Decision Tree Regressor built with variance reduction (Mean Squared Error) splitting criterion.
    """
    def __init__(self, max_depth: int = 6, min_samples_split: int = 8):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.root = None

    def fit(self, X: List[List[float]], y: List[float]):
        self.root = self._build_tree(X, y, depth=0)
        return self

    def _build_tree(self, X: List[List[float]], y: List[float], depth: int = 0):
        n_samples = len(y)
        if depth >= self.max_depth or n_samples < self.min_samples_split or len(set(y)) <= 1:
            leaf_value = sum(y) / n_samples if n_samples > 0 else 0.0
            return TreeNode(value=leaf_value)

        n_features = len(X[0])
        best_feat, best_thresh = None, None
        best_mse = float("inf")
        best_left_idx, best_right_idx = [], []

        mean_y = sum(y) / n_samples
        curr_mse = sum((val - mean_y) ** 2 for val in y)

        # Subsample features or scan
        features_to_check = random.sample(range(n_features), min(n_features, 6))

        for feat in features_to_check:
            vals = [X[i][feat] for i in range(n_samples)]
            # Test a subset of quantiles as split candidates
            sorted_unique = sorted(list(set(vals)))
            if len(sorted_unique) > 10:
                step = len(sorted_unique) // 8
                candidates = sorted_unique[::step]
            else:
                candidates = sorted_unique

            for thresh in candidates:
                left_idx = [i for i in range(n_samples) if X[i][feat] <= thresh]
                right_idx = [i for i in range(n_samples) if X[i][feat] > thresh]

                if not left_idx or not right_idx:
                    continue

                y_left = [y[i] for i in left_idx]
                y_right = [y[i] for i in right_idx]

                mean_l = sum(y_left) / len(y_left)
                mean_r = sum(y_right) / len(y_right)

                mse_split = sum((yl - mean_l) ** 2 for yl in y_left) + sum((yr - mean_r) ** 2 for yr in y_right)
                if mse_split < best_mse:
                    best_mse = mse_split
                    best_feat = feat
                    best_thresh = thresh
                    best_left_idx = left_idx
                    best_right_idx = right_idx

        if best_feat is None or best_mse >= curr_mse:
            return TreeNode(value=sum(y) / n_samples)

        X_left = [X[i] for i in best_left_idx]
        y_left = [y[i] for i in best_left_idx]
        X_right = [X[i] for i in best_right_idx]
        y_right = [y[i] for i in best_right_idx]

        left_child = self._build_tree(X_left, y_left, depth + 1)
        right_child = self._build_tree(X_right, y_right, depth + 1)
        return TreeNode(feature_idx=best_feat, threshold=best_thresh, left=left_child, right=right_child)

    def _predict_sample(self, node: TreeNode, sample: List[float]) -> float:
        if node.is_leaf:
            return node.value
        if sample[node.feature_idx] <= node.threshold:
            return self._predict_sample(node.left, sample)
        return self._predict_sample(node.right, sample)

    def predict(self, X: List[List[float]]) -> List[float]:
        return [self._predict_sample(self.root, row) for row in X]


class RandomForestRegressorCustom(BaseRegressor):
    """
    Random Forest Regressor: Bootstrap aggregation (Bagging) of randomized Decision Trees.
    Reduces variance and prevents overfitting.
    """
    def __init__(self, n_estimators: int = 15, max_depth: int = 7, min_samples_split: int = 6):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.trees = []

    def fit(self, X: List[List[float]], y: List[float]):
        n_samples = len(X)
        self.trees = []
        for _ in range(self.n_estimators):
            # Bootstrap sampling with replacement
            indices = [random.randint(0, n_samples - 1) for _ in range(n_samples)]
            X_boot = [X[i] for i in indices]
            y_boot = [y[i] for i in indices]
            tree = DecisionTreeRegressorCustom(max_depth=self.max_depth, min_samples_split=self.min_samples_split)
            tree.fit(X_boot, y_boot)
            self.trees.append(tree)
        return self

    def predict(self, X: List[List[float]]) -> List[float]:
        tree_preds = [tree.predict(X) for tree in self.trees]
        n_samples = len(X)
        ensemble_preds = []
        for i in range(n_samples):
            avg_pred = sum(tree_preds[t][i] for t in range(len(self.trees))) / len(self.trees)
            ensemble_preds.append(max(20000.0, avg_pred))
        return ensemble_preds


class GradientBoostingRegressorCustom(BaseRegressor):
    """
    Gradient Boosting Regressor: Sequentially fits shallow regression trees
    to the negative gradient (residuals) of the pseudo-Huber / MSE loss function.
    """
    def __init__(self, n_estimators: int = 20, learning_rate: float = 0.1, max_depth: int = 4):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.trees = []
        self.base_pred = 0.0

    def fit(self, X: List[List[float]], y: List[float]):
        n_samples = len(y)
        self.base_pred = sum(y) / n_samples
        curr_preds = [self.base_pred] * n_samples
        self.trees = []

        for _ in range(self.n_estimators):
            # Compute negative gradient (residuals)
            residuals = [y[i] - curr_preds[i] for i in range(n_samples)]
            tree = DecisionTreeRegressorCustom(max_depth=self.max_depth, min_samples_split=8)
            tree.fit(X, residuals)
            step_preds = tree.predict(X)

            # Update ensemble prediction
            for i in range(n_samples):
                curr_preds[i] += self.learning_rate * step_preds[i]
            self.trees.append(tree)
        return self

    def predict(self, X: List[List[float]]) -> List[float]:
        n_samples = len(X)
        preds = [self.base_pred] * n_samples
        for tree in self.trees:
            step = tree.predict(X)
            for i in range(n_samples):
                preds[i] += self.learning_rate * step[i]
        return [max(20000.0, p) for p in preds]
