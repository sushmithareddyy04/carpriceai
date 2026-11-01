"""
Module 8: Feature Engineering
==============================
Focus: Categorical encoding (One-Hot, Ordinal, Brand Segment) and feature scaling
(StandardScaler) for numerical attributes.
Includes:
- Brand extraction from car title
- Vehicle Age computation (reference_year - manufacturing_year)
- One-Hot Encoding for fuel, transmission, seller_type
- Ordinal Encoding for owner hierarchy
- StandardScaler implementation (fit, transform, inverse_transform)
- Complete unified FeaturePipeline
"""

import math
from typing import List, Dict, Any, Tuple
import logging

logger = logging.getLogger("CarPriceAI.Module8")


class StandardScalerCustom:
    """
    StandardScaler: Standardizes features by removing the mean and scaling to unit variance.
    z = (x - u) / s
    """
    def __init__(self):
        self.means = []
        self.stds = []
        self.feature_names = []

    def fit(self, X: List[List[float]], feature_names: List[str] = None):
        """Computes the mean and std to be used for later scaling."""
        n_samples = len(X)
        if n_samples == 0:
            return self
        n_features = len(X[0])
        self.feature_names = feature_names or [f"f_{i}" for i in range(n_features)]

        self.means = []
        self.stds = []

        for j in range(n_features):
            col = [X[i][j] for i in range(n_samples)]
            mean_j = sum(col) / n_samples
            var_j = sum((x - mean_j) ** 2 for x in col) / n_samples
            std_j = math.sqrt(var_j) if var_j > 1e-9 else 1.0
            self.means.append(mean_j)
            self.stds.append(std_j)
        return self

    def transform(self, X: List[List[float]]) -> List[List[float]]:
        """Perform standardization by centering and scaling."""
        transformed = []
        for row in X:
            scaled_row = [(row[j] - self.means[j]) / self.stds[j] for j in range(len(row))]
            transformed.append(scaled_row)
        return transformed

    def fit_transform(self, X: List[List[float]], feature_names: List[str] = None) -> List[List[float]]:
        return self.fit(X, feature_names).transform(X)


class FeatureEngineer:
    """
    Production-grade Feature Engineering Pipeline for Automotive Valuation.
    Extracts brand tiers, computes depreciation age, encodes categoricals, and scales numerics.
    """

    OWNER_MAP = {
        "First Owner": 1.0,
        "Second Owner": 2.0,
        "Third Owner": 3.0,
        "Fourth & Above Owner": 4.0,
        "Test Drive Car": 0.5
    }

    LUXURY_BRANDS = {"BMW", "Mercedes-Benz", "Audi", "Jaguar", "Land Rover", "Volvo", "Porsche"}
    PREMIUM_BRANDS = {"Toyota", "Skoda", "Volkswagen", "Ford", "Honda", "Kia", "MG", "Jeep"}

    def __init__(self, reference_year: int = 2026):
        self.reference_year = reference_year
        self.scaler = StandardScalerCustom()
        self.is_fitted = False
        self.feature_names = []

    def extract_brand_tier(self, brand: str) -> float:
        """Categorizes brand into pricing tiers: 3.0 (Luxury), 2.0 (Premium/Executive), 1.0 (Economy)."""
        if brand in self.LUXURY_BRANDS:
            return 3.0
        elif brand in self.PREMIUM_BRANDS:
            return 2.0
        return 1.0

    def transform_single_to_vector(self, record: Dict[str, Any]) -> List[float]:
        """Transforms a single car record into an unscaled feature vector."""
        name = str(record.get("name", ""))
        brand = record.get("brand") or (name.split()[0] if name else "Unknown")
        year = int(record.get("year", 2018))
        age = max(0, self.reference_year - year)

        km = float(record.get("km_driven", 50000))
        mileage = float(record.get("mileage", 19.0))
        engine = float(record.get("engine", 1497.0))
        power = float(record.get("max_power", 90.0))
        seats = float(record.get("seats", 5.0))

        owner_str = str(record.get("owner", "First Owner"))
        owner_val = self.OWNER_MAP.get(owner_str, 2.0)

        fuel = str(record.get("fuel", "Petrol"))
        is_diesel = 1.0 if fuel == "Diesel" else 0.0
        is_petrol = 1.0 if fuel == "Petrol" else 0.0
        is_cng = 1.0 if fuel in ["CNG", "LPG"] else 0.0

        trans = str(record.get("transmission", "Manual"))
        is_automatic = 1.0 if trans == "Automatic" else 0.0

        seller = str(record.get("seller_type", "Individual"))
        is_dealer = 1.0 if "Dealer" in seller else 0.0

        brand_tier = self.extract_brand_tier(brand)

        # Unscaled continuous & categorical interaction vector
        vector = [
            float(age),          # 0: vehicle_age
            km,                  # 1: km_driven
            mileage,             # 2: mileage
            engine,              # 3: engine_cc
            power,               # 4: max_power
            seats,               # 5: seats
            owner_val,           # 6: owner_tier
            brand_tier,          # 7: brand_tier
            is_diesel,           # 8: fuel_diesel
            is_petrol,           # 9: fuel_petrol
            is_cng,              # 10: fuel_cng
            is_automatic,        # 11: transmission_automatic
            is_dealer            # 12: seller_dealer
        ]
        return vector

    def fit(self, records: List[Dict[str, Any]]):
        """Fits the StandardScaler on the transformed feature vectors."""
        self.feature_names = [
            "vehicle_age", "km_driven", "mileage", "engine_cc", "max_power", "seats",
            "owner_tier", "brand_tier", "fuel_diesel", "fuel_petrol", "fuel_cng",
            "transmission_automatic", "seller_dealer"
        ]
        unscaled_X = [self.transform_single_to_vector(r) for r in records]
        self.scaler.fit(unscaled_X, self.feature_names)
        self.is_fitted = True
        return self

    def transform(self, records: List[Dict[str, Any]]) -> List[List[float]]:
        """Transforms records to standardized scaled vectors."""
        if not self.is_fitted:
            raise ValueError("FeatureEngineer must be fitted before calling transform().")
        unscaled_X = [self.transform_single_to_vector(r) for r in records]
        return self.scaler.transform(unscaled_X)

    def fit_transform(self, records: List[Dict[str, Any]]) -> Tuple[List[List[float]], List[float]]:
        """Fits scaler and transforms records into X (features) and y (target selling prices)."""
        self.fit(records)
        X = self.transform(records)
        y = [float(r.get("selling_price", 0.0)) for r in records]
        return X, y
