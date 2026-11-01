"""
Module 4: Exploratory Data Analysis (EDA) & Visualization
==========================================================
Focus: Matplotlib & Seaborn visual charts analyzing price vs. year, price vs. km driven,
fuel type distribution, and correlation heatmaps.
Includes:
- Pearson correlation matrix calculation
- Price vs. Year depreciation trend aggregation
- Price vs. Mileage / Kilometers driven distributions
- Fuel type and Transmission market share analytics
- Static chart rendering utilities using Matplotlib & Seaborn
"""

import math
import os
from typing import List, Dict, Any, Tuple
import logging

logger = logging.getLogger("CarPriceAI.Module4")


class EDAAnalyzer:
    """
    Computes visual analytics data structures, correlation matrices,
    and trends for automotive market datasets.
    """

    @staticmethod
    def compute_pearson_correlation(x: List[float], y: List[float]) -> float:
        """Calculates the Pearson correlation coefficient between two numeric lists."""
        n = len(x)
        if n < 2 or len(y) != n:
            return 0.0

        mean_x = sum(x) / n
        mean_y = sum(y) / n

        numerator = sum((x[i] - mean_x) * (y[i] - mean_y) for i in range(n))
        denom_x = sum((x[i] - mean_x) ** 2 for i in range(n))
        denom_y = sum((y[i] - mean_y) ** 2 for i in range(n))

        if denom_x <= 0 or denom_y <= 0:
            return 0.0

        return round(numerator / math.sqrt(denom_x * denom_y), 4)

    @classmethod
    def generate_correlation_matrix(cls, records: List[Dict[str, Any]], features: List[str]) -> Dict[str, Dict[str, float]]:
        """Generates a complete correlation matrix across numerical attributes."""
        # Extract columns
        columns = {}
        for feat in features:
            columns[feat] = [float(r.get(feat, 0.0)) for r in records]

        matrix = {}
        for f1 in features:
            matrix[f1] = {}
            for f2 in features:
                matrix[f1][f2] = cls.compute_pearson_correlation(columns[f1], columns[f2])
        return matrix

    @staticmethod
    def aggregate_price_by_year(records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Aggregates average and median resale price by manufacturing year."""
        year_groups = {}
        for r in records:
            yr = int(r.get("year", 2018))
            p = float(r.get("selling_price", 0.0))
            if yr not in year_groups:
                year_groups[yr] = []
            year_groups[yr].append(p)

        trend = []
        for yr in sorted(year_groups.keys()):
            prices = year_groups[yr]
            avg_p = sum(prices) / len(prices)
            s_prices = sorted(prices)
            med_p = s_prices[len(prices) // 2]
            trend.append({
                "year": yr,
                "avg_price": round(avg_p, 2),
                "median_price": round(med_p, 2),
                "count": len(prices)
            })
        return trend

    @staticmethod
    def aggregate_by_category(records: List[Dict[str, Any]], category_key: str) -> List[Dict[str, Any]]:
        """Aggregates vehicle counts and average resale prices by categorical attribute."""
        groups = {}
        for r in records:
            cat = str(r.get(category_key, "Unknown"))
            p = float(r.get("selling_price", 0.0))
            if cat not in groups:
                groups[cat] = []
            groups[cat].append(p)

        output = []
        for cat, prices in sorted(groups.items(), key=lambda x: -sum(x[1]) / len(x[1])):
            avg_p = sum(prices) / len(prices)
            output.append({
                "category": cat,
                "count": len(prices),
                "avg_price": round(avg_p, 2),
                "min_price": round(min(prices), 2),
                "max_price": round(max(prices), 2)
            })
        return output

    @staticmethod
    def render_static_eda_plots(records: List[Dict[str, Any]], output_dir: str = "models/eda_charts"):
        """Generates static PNG plots using matplotlib if available in the runtime."""
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            import seaborn as sns

            os.makedirs(output_dir, exist_ok=True)
            years = [int(r["year"]) for r in records]
            prices = [float(r["selling_price"]) for r in records]
            km = [float(r["km_driven"]) for r in records]

            # 1. Price vs Year
            plt.figure(figsize=(9, 5))
            sns.scatterplot(x=years, y=prices, alpha=0.5, color="#38bdf8")
            plt.title("Automotive Selling Price vs. Manufacturing Year", fontsize=14, fontweight="bold")
            plt.xlabel("Year")
            plt.ylabel("Selling Price (INR)")
            plt.grid(True, linestyle="--", alpha=0.5)
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, "price_vs_year.png"), dpi=150)
            plt.close()

            # 2. Price vs KM Driven
            plt.figure(figsize=(9, 5))
            sns.scatterplot(x=km, y=prices, alpha=0.5, color="#f59e0b")
            plt.title("Selling Price vs. Kilometers Driven", fontsize=14, fontweight="bold")
            plt.xlabel("Kilometers Driven")
            plt.ylabel("Selling Price (INR)")
            plt.grid(True, linestyle="--", alpha=0.5)
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, "price_vs_km.png"), dpi=150)
            plt.close()

            logger.info("Static EDA plots generated in %s", output_dir)
        except Exception as e:
            logger.warning("Matplotlib plot rendering skipped or unavailable: %s", e)
