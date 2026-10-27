"""
Prediction module for JCAB.
Handles XGBoost model loading and preliminary ML predictions.
"""
import logging
from typing import Any, Tuple
import joblib
import numpy as np
import streamlit as st
from config.settings import (
    EXPECTED_EMBEDDING_DIMENSION,
    LEVEL_MODEL_PATH,
    SCORE_MODEL_PATH,
    ENCODER_MODEL_PATH,
)

logger = logging.getLogger("JCAB")

@st.cache_resource(show_spinner=False)
def load_ml_models() -> Tuple[Any, Any]:
    logger.info("Loading preliminary XGBoost models (synthetic data trained)")
    
    if not SCORE_MODEL_PATH.exists():
        raise FileNotFoundError(f"Score model not found:\n{SCORE_MODEL_PATH}")
    if not LEVEL_MODEL_PATH.exists():
        raise FileNotFoundError(f"Learner-level model not found:\n{LEVEL_MODEL_PATH}")

    if not ENCODER_MODEL_PATH.exists():
        raise FileNotFoundError(f"Learner-level encoder not found:\n{ENCODER_MODEL_PATH}")

    # Load models
    score_model = joblib.load(SCORE_MODEL_PATH)
    level_model = joblib.load(LEVEL_MODEL_PATH)
    encoder_model = joblib.load(ENCODER_MODEL_PATH)

    for name, model, path in (
        ("score", score_model, SCORE_MODEL_PATH),
        ("learner-level", level_model, LEVEL_MODEL_PATH),
    ):
        feature_count = getattr(model, "n_features_in_", None)
        if feature_count is None or feature_count != EXPECTED_EMBEDDING_DIMENSION:
            raise ValueError(
                f"Incompatible {name} model {path.name}: expected "
                f"{EXPECTED_EMBEDDING_DIMENSION} input features, received {feature_count}."
            )
        if not hasattr(model, "get_booster"):
            raise TypeError(f"Incompatible {name} model {path.name}: expected an XGBoost model.")

    class_count = getattr(level_model, "n_classes_", None)
    if class_count != len(encoder_model.classes_):
        raise ValueError(
            f"Incompatible learner-level model: expected {len(encoder_model.classes_)} "
            f"classes, received {class_count}."
        )

    logger.info(
        "Loaded score model=%s, learner-level model=%s, encoder=%s; input_shape=(1, %d)",
        type(score_model).__name__, type(level_model).__name__, type(encoder_model).__name__, EXPECTED_EMBEDDING_DIMENSION,
    )
    
    return score_model, level_model, encoder_model


try:
    import pkg_resources
    if not hasattr(pkg_resources, 'get_distribution'):
        class DistributionNotFound(Exception):
            pass
        def _mock_get_distribution(dist_name):
            raise DistributionNotFound(dist_name)
        pkg_resources.get_distribution = _mock_get_distribution
        pkg_resources.DistributionNotFound = DistributionNotFound
except ImportError:
    import types
    import sys
    pkg_resources = types.ModuleType("pkg_resources")
    class DistributionNotFound(Exception):
        pass
    def _mock_get_distribution(dist_name):
        raise DistributionNotFound(dist_name)
    pkg_resources.get_distribution = _mock_get_distribution
    pkg_resources.DistributionNotFound = DistributionNotFound
    sys.modules["pkg_resources"] = pkg_resources

import xgboost as xgb

def predict_score(embedding: np.ndarray, score_model: Any) -> float:
    """
    Predict numerical teacher score using the preliminary XGBoost Regressor.
    """
    try:
        # Use Booster directly to bypass sklearn wrapper attribute errors from older versions
        booster = score_model.get_booster()
        dmatrix = xgb.DMatrix(embedding)
        prediction = booster.predict(dmatrix)
        result = float(prediction[0])
        logger.info(
            "Score model=%s input_shape=%s prediction_output=%s",
            type(score_model).__name__, embedding.shape, result,
        )
        return result
    except Exception as e:
        logger.error(f"Score prediction failed: {e}")
        raise


def predict_level(embedding: np.ndarray, level_model: Any) -> Any:
    """
    Predict learner-level class using the preliminary XGBoost Classifier.
    """
    try:
        # Use Booster directly to bypass sklearn wrapper attribute errors
        booster = level_model.get_booster()
        dmatrix = xgb.DMatrix(embedding)
        probs = booster.predict(dmatrix)
        # Booster predict returns probabilities for classifiers; get the argmax
        predicted_class = int(np.argmax(probs[0]))
        logger.info(
            "Learner-level model=%s input_shape=%s prediction_output=%s",
            type(level_model).__name__, embedding.shape, predicted_class,
        )
        return predicted_class
    except Exception as e:
        logger.error(f"Level prediction failed: {e}")
        raise


def decode_learner_level(encoded_level: int, encoder: Any) -> str:
    """Decode the classifier output using the trained LabelEncoder."""
    if not isinstance(encoded_level, (int, np.integer)):
        raise TypeError(f"Encoded learner level must be an integer, received {encoded_level!r}.")
    try:
        return encoder.inverse_transform([int(encoded_level)])[0]
    except Exception as exc:
        raise ValueError(f"Unknown encoded learner level: {encoded_level}") from exc
