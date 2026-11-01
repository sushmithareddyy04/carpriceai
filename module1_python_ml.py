"""
Module 1: Python for Machine Learning
======================================
Focus: Clean, modular Python scripts with object-oriented and functional programming practices.
Includes:
- Data validation models & schemas
- Custom exception handling
- Config dataclasses
- Functional transformation helpers
- Structured logging
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Callable
import logging

# Configure structured logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("CarPriceAI.Module1")


class AutomotiveDataError(Exception):
    """Custom exception raised for invalid or corrupt automotive data inputs."""
    pass


class ModelInferenceError(Exception):
    """Custom exception raised during model prediction failures."""
    pass


@dataclass(frozen=True)
class CarDataConfig:
    """Immutable configuration dataclass defining dataset parameters and boundaries."""
    target_col: str = "selling_price"
    reference_year: int = 2026
    categorical_features: List[str] = field(default_factory=lambda: [
        "fuel", "seller_type", "transmission", "owner", "brand"
    ])
    numerical_features: List[str] = field(default_factory=lambda: [
        "year", "km_driven", "mileage", "engine", "max_power", "seats", "vehicle_age"
    ])
    min_price: float = 20000.0
    max_price: float = 25000000.0
    min_year: int = 1995
    max_year: int = 2026


class DataValidator:
    """
    Object-Oriented Validator for automotive feature records.
    Ensures input features conform to expected bounds and data types.
    """
    def __init__(self, config: Optional[CarDataConfig] = None):
        self.config = config or CarDataConfig()

    def validate_record(self, record: Dict[str, Any]) -> bool:
        """Validates a single car record against data boundaries."""
        year = record.get("year")
        if year is not None and not (self.config.min_year <= int(year) <= self.config.max_year):
            raise AutomotiveDataError(f"Year {year} is out of expected range ({self.config.min_year}-{self.config.max_year})")
        
        km = record.get("km_driven")
        if km is not None and int(km) < 0:
            raise AutomotiveDataError(f"Kilometers driven cannot be negative: {km}")

        fuel = record.get("fuel")
        allowed_fuels = {"Diesel", "Petrol", "CNG", "LPG", "Electric"}
        if fuel is not None and fuel not in allowed_fuels:
            raise AutomotiveDataError(f"Unsupported fuel type '{fuel}'. Allowed: {allowed_fuels}")

        trans = record.get("transmission")
        allowed_trans = {"Manual", "Automatic"}
        if trans is not None and trans not in allowed_trans:
            raise AutomotiveDataError(f"Unsupported transmission '{trans}'. Allowed: {allowed_trans}")

        return True

    def validate_batch(self, records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Functional batch validation filtering valid records."""
        valid_records = []
        for rec in records:
            try:
                if self.validate_record(rec):
                    valid_records.append(rec)
            except AutomotiveDataError as e:
                logger.warning("Dropping invalid record: %s", e)
        return valid_records


# Functional Programming Utilities
def apply_transform(data: List[Dict[str, Any]], key: str, func: Callable[[Any], Any]) -> List[Dict[str, Any]]:
    """Pure functional helper that maps a transform function across a list of dicts."""
    return [{**row, key: func(row[key])} if key in row else dict(row) for row in data]


def filter_by_condition(data: List[Dict[str, Any]], predicate: Callable[[Dict[str, Any]], bool]) -> List[Dict[str, Any]]:
    """Functional filter returning elements satisfying the given predicate."""
    return list(filter(predicate, data))
