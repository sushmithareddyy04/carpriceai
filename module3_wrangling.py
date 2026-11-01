"""
Module 3: Data Wrangling & Preprocessing
=========================================
Focus: Handling missing values, dropping duplicates, and correcting data types.
Includes:
- Unit extraction from strings ('1248 CC' -> 1248, '18.9 kmpl' -> 18.9, '82 bhp' -> 82)
- Duplicate detection and removal
- Null value imputation (grouped medians for engine, mileage, max_power, mode for seats)
- Type casting and schema consistency enforcement
"""

import re
import math
from typing import List, Dict, Any, Tuple
import logging

logger = logging.getLogger("CarPriceAI.Module3")


class DataWrangler:
    """
    Cleans raw car records, standardizes unit strings into numerical values,
    removes duplicates, and performs median/mode imputation.
    """

    @staticmethod
    def extract_float(value: Any) -> float:
        """Extracts the first floating point or integer number from a string with units."""
        if value is None:
            return float("nan")
        if isinstance(value, (int, float)):
            return float(value)
        match = re.search(r"[-+]?\d*\.?\d+", str(value))
        if match:
            try:
                return float(match.group(0))
            except ValueError:
                return float("nan")
        return float("nan")

    @classmethod
    def clean_record(cls, record: Dict[str, Any]) -> Dict[str, Any]:
        """Cleans and casts individual fields in a single record."""
        cleaned = dict(record)

        # Name and brand extraction
        name = str(record.get("name", "")).strip()
        cleaned["name"] = name
        cleaned["brand"] = name.split()[0] if name else "Unknown"

        # Numerical type conversions
        try:
            cleaned["year"] = int(record.get("year", 2018))
        except (ValueError, TypeError):
            cleaned["year"] = 2018

        try:
            cleaned["selling_price"] = float(record.get("selling_price", 0.0))
        except (ValueError, TypeError):
            cleaned["selling_price"] = 0.0

        try:
            cleaned["km_driven"] = float(record.get("km_driven", 0.0))
        except (ValueError, TypeError):
            cleaned["km_driven"] = 0.0

        # Unit stripping
        cleaned["mileage"] = cls.extract_float(record.get("mileage"))
        cleaned["engine"] = cls.extract_float(record.get("engine"))
        cleaned["max_power"] = cls.extract_float(record.get("max_power"))
        cleaned["seats"] = cls.extract_float(record.get("seats"))

        # Categoricals
        cleaned["fuel"] = str(record.get("fuel", "Petrol")).strip()
        cleaned["seller_type"] = str(record.get("seller_type", "Individual")).strip()
        cleaned["transmission"] = str(record.get("transmission", "Manual")).strip()
        cleaned["owner"] = str(record.get("owner", "First Owner")).strip()

        return cleaned

    @classmethod
    def drop_duplicates(cls, records: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], int]:
        """Identifies and eliminates identical duplicate records based on core attributes."""
        seen = set()
        unique_records = []
        duplicate_count = 0

        for r in records:
            key = (
                r.get("name"),
                r.get("year"),
                r.get("selling_price"),
                r.get("km_driven"),
                r.get("fuel"),
                r.get("transmission")
            )
            if key in seen:
                duplicate_count += 1
            else:
                seen.add(key)
                unique_records.append(r)

        logger.info("Deduplication: Removed %d duplicate rows. %d rows remain.", duplicate_count, len(unique_records))
        return unique_records, duplicate_count

    @classmethod
    def impute_missing_values(cls, records: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], Dict[str, int]]:
        """
        Imputes missing values in numerical columns (mileage, engine, max_power, seats)
        using brand-level medians or overall global medians.
        """
        # Calculate brand-specific medians and global medians
        features_to_impute = ["mileage", "engine", "max_power", "seats"]
        global_vals = {feat: [] for feat in features_to_impute}
        brand_vals = {}

        for r in records:
            brand = r.get("brand", "Unknown")
            if brand not in brand_vals:
                brand_vals[brand] = {feat: [] for feat in features_to_impute}

            for feat in features_to_impute:
                v = r.get(feat)
                if v is not None and not math.isnan(v) and v > 0:
                    global_vals[feat].append(v)
                    brand_vals[brand][feat].append(v)

        # Compute median helper
        def get_median(lst: List[float], fallback: float) -> float:
            if not lst:
                return fallback
            s = sorted(lst)
            mid = len(s) // 2
            return s[mid] if len(s) % 2 == 1 else (s[mid - 1] + s[mid]) / 2.0

        global_medians = {
            "mileage": get_median(global_vals["mileage"], 19.0),
            "engine": get_median(global_vals["engine"], 1497.0),
            "max_power": get_median(global_vals["max_power"], 88.5),
            "seats": get_median(global_vals["seats"], 5.0)
        }

        imputed_counts = {feat: 0 for feat in features_to_impute}
        imputed_records = []

        for r in records:
            r_copy = dict(r)
            brand = r_copy.get("brand", "Unknown")

            for feat in features_to_impute:
                v = r_copy.get(feat)
                if v is None or math.isnan(v) or v <= 0:
                    brand_med = get_median(brand_vals.get(brand, {}).get(feat, []), global_medians[feat])
                    r_copy[feat] = brand_med
                    imputed_counts[feat] += 1

            imputed_records.append(r_copy)

        logger.info("Imputation summary: %s", imputed_counts)
        return imputed_records, imputed_counts

    @classmethod
    def wrangle_dataset(cls, raw_records: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        """End-to-end data wrangling orchestrator."""
        cleaned = [cls.clean_record(r) for r in raw_records]
        deduped, dup_count = cls.drop_duplicates(cleaned)
        imputed, imp_counts = cls.impute_missing_values(deduped)

        metadata = {
            "initial_rows": len(raw_records),
            "duplicates_removed": dup_count,
            "missing_values_imputed": imp_counts,
            "final_clean_rows": len(imputed)
        }
        return imputed, metadata
