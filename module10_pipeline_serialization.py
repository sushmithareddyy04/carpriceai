"""
Module 10: ML Projects & Model Serialization
=============================================
Focus: Serializing the trained champion model, scaler, and preprocessor using Python's
pickle library, and exporting metadata for deployment in Streamlit.
Includes:
- Model artifact serialization (.pkl)
- Preprocessor artifact serialization (.pkl)
- Comprehensive metadata serialization (JSON & PKL)
- Integrity checks and loading routines
"""

import pickle
import json
import os
from typing import Dict, Any, Tuple
import logging

logger = logging.getLogger("CarPriceAI.Module10")


class PipelineSerializer:
    """
    Manages persistence of ML model artifacts and pipeline components.
    """

    @staticmethod
    def save_artifacts(
        champion_model: Any,
        preprocessor: Any,
        metadata: Dict[str, Any],
        artifacts_dir: str = "models"
    ) -> Dict[str, str]:
        """Serializes model, preprocessor, and metadata to the target directory."""
        os.makedirs(artifacts_dir, exist_ok=True)

        model_path = os.path.join(artifacts_dir, "car_price_champion.pkl")
        preprocessor_path = os.path.join(artifacts_dir, "preprocessor.pkl")
        meta_json_path = os.path.join(artifacts_dir, "model_metadata.json")
        meta_pkl_path = os.path.join(artifacts_dir, "model_metadata.pkl")

        # 1. Save Champion Model
        with open(model_path, "wb") as f:
            pickle.dump(champion_model, f, protocol=pickle.HIGHEST_PROTOCOL)
        logger.info("Saved champion model to %s", model_path)

        # 2. Save Preprocessor
        with open(preprocessor_path, "wb") as f:
            pickle.dump(preprocessor, f, protocol=pickle.HIGHEST_PROTOCOL)
        logger.info("Saved preprocessor to %s", preprocessor_path)

        # 3. Save Metadata JSON
        with open(meta_json_path, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=4)
        logger.info("Saved metadata JSON to %s", meta_json_path)

        # 4. Save Metadata PKL
        with open(meta_pkl_path, "wb") as f:
            pickle.dump(metadata, f, protocol=pickle.HIGHEST_PROTOCOL)

        return {
            "model_path": model_path,
            "preprocessor_path": preprocessor_path,
            "metadata_json": meta_json_path,
            "metadata_pkl": meta_pkl_path
        }

    @staticmethod
    def load_artifacts(artifacts_dir: str = "models") -> Tuple[Any, Any, Dict[str, Any]]:
        """Loads and verifies serialized artifacts from disk."""
        model_path = os.path.join(artifacts_dir, "car_price_champion.pkl")
        preprocessor_path = os.path.join(artifacts_dir, "preprocessor.pkl")
        meta_json_path = os.path.join(artifacts_dir, "model_metadata.json")

        if not (os.path.exists(model_path) and os.path.exists(preprocessor_path)):
            raise FileNotFoundError(f"Model artifacts not found in directory: {artifacts_dir}")

        with open(model_path, "rb") as f:
            model = pickle.load(f)

        with open(preprocessor_path, "rb") as f:
            preprocessor = pickle.load(f)

        metadata = {}
        if os.path.exists(meta_json_path):
            with open(meta_json_path, "r", encoding="utf-8") as f:
                metadata = json.load(f)

        logger.info("Artifacts successfully loaded from %s", artifacts_dir)
        return model, preprocessor, metadata
