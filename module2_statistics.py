"""
Module 2: Statistics & Probability
===================================
Focus: Statistical summaries, distribution checks, and outlier detection on car prices
and kilometers driven using mathematical and statistical formulations.
Includes:
- Central tendency (Mean, Median, Mode)
- Dispersion (Variance, Standard Deviation, Range, IQR)
- Shape metrics (Skewness, Kurtosis)
- Outlier detection (IQR Method, Z-Score Method)
"""

import math
from typing import List, Dict, Tuple, Any
import logging

logger = logging.getLogger("CarPriceAI.Module2")


class StatisticalAnalyzer:
    """
    Computes statistical profiles, probability distributions, and outlier metrics
    for continuous numerical features.
    """

    @staticmethod
    def calculate_summary(values: List[float]) -> Dict[str, float]:
        """Calculates comprehensive descriptive statistics for a numerical series."""
        clean_vals = [float(v) for v in values if v is not None and not math.isnan(float(v))]
        n = len(clean_vals)
        if n == 0:
            return {}

        sorted_vals = sorted(clean_vals)
        mean_val = sum(sorted_vals) / n

        # Median
        if n % 2 == 1:
            median_val = sorted_vals[n // 2]
        else:
            median_val = (sorted_vals[n // 2 - 1] + sorted_vals[n // 2]) / 2.0

        # Variance and Standard Deviation
        variance_val = sum((x - mean_val) ** 2 for x in sorted_vals) / (n - 1) if n > 1 else 0.0
        std_val = math.sqrt(variance_val)

        # Quartiles (Tukey Method)
        q1_idx = int(0.25 * n)
        q3_idx = int(0.75 * n)
        q1 = sorted_vals[q1_idx]
        q3 = sorted_vals[q3_idx]
        iqr = q3 - q1

        # Skewness (Fisher-Pearson coefficient of skewness)
        if std_val > 0 and n > 2:
            m3 = sum((x - mean_val) ** 3 for x in sorted_vals) / n
            skewness = m3 / (std_val ** 3)
            m4 = sum((x - mean_val) ** 4 for x in sorted_vals) / n
            kurtosis = (m4 / (std_val ** 4)) - 3.0  # Excess kurtosis
        else:
            skewness = 0.0
            kurtosis = 0.0

        return {
            "count": float(n),
            "mean": round(mean_val, 2),
            "median": round(median_val, 2),
            "std": round(std_val, 2),
            "min": round(sorted_vals[0], 2),
            "q1": round(q1, 2),
            "q3": round(q3, 2),
            "iqr": round(iqr, 2),
            "max": round(sorted_vals[-1], 2),
            "skewness": round(skewness, 4),
            "kurtosis": round(kurtosis, 4)
        }

    @staticmethod
    def detect_outliers_iqr(values: List[float], multiplier: float = 1.5) -> Tuple[List[int], float, float]:
        """
        Detects outlier indices using the Interquartile Range (IQR) rule:
        Lower Bound = Q1 - 1.5 * IQR
        Upper Bound = Q3 + 1.5 * IQR
        """
        summary = StatisticalAnalyzer.calculate_summary(values)
        q1 = summary.get("q1", 0.0)
        q3 = summary.get("q3", 0.0)
        iqr = summary.get("iqr", 0.0)

        lower_bound = max(0.0, q1 - multiplier * iqr)
        upper_bound = q3 + multiplier * iqr

        outlier_indices = []
        for idx, val in enumerate(values):
            if val is not None:
                v = float(val)
                if v < lower_bound or v > upper_bound:
                    outlier_indices.append(idx)

        logger.info(
            "IQR Outlier Bounds: [%.2f, %.2f] -> Found %d outliers out of %d records",
            lower_bound, upper_bound, len(outlier_indices), len(values)
        )
        return outlier_indices, lower_bound, upper_bound

    @staticmethod
    def detect_outliers_zscore(values: List[float], threshold: float = 3.0) -> List[int]:
        """Detects outlier indices based on standard normal Z-Score: |(x - μ) / σ| > threshold."""
        summary = StatisticalAnalyzer.calculate_summary(values)
        mean_val = summary.get("mean", 0.0)
        std_val = summary.get("std", 1.0)
        if std_val == 0:
            return []

        outlier_indices = []
        for idx, val in enumerate(values):
            if val is not None:
                z = abs((float(val) - mean_val) / std_val)
                if z > threshold:
                    outlier_indices.append(idx)
        return outlier_indices

    @staticmethod
    def compute_histogram_bins(values: List[float], num_bins: int = 20) -> Dict[str, Any]:
        """Computes bin edges and counts for distribution visualization."""
        clean = [float(v) for v in values if v is not None and not math.isnan(float(v))]
        if not clean:
            return {"bins": [], "counts": []}

        min_v = min(clean)
        max_v = max(clean)
        step = (max_v - min_v) / num_bins if max_v > min_v else 1.0

        bins = [min_v + i * step for i in range(num_bins + 1)]
        counts = [0] * num_bins

        for v in clean:
            bin_idx = min(int((v - min_v) / step), num_bins - 1)
            counts[bin_idx] += 1

        bin_labels = [f"{bins[i]:,.0f} - {bins[i+1]:,.0f}" for i in range(num_bins)]
        return {"bins": bins, "bin_labels": bin_labels, "counts": counts}
