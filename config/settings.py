"""
Configuration settings for the JCAB Academic Prototype.
"""
from pathlib import Path
import os
from dotenv import load_dotenv

# Project structure
BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_DIR = BASE_DIR / "models"
DATA_DIR = BASE_DIR / "data"

# Load environment variables explicitly from project root
env_path = BASE_DIR / ".env"
load_dotenv(dotenv_path=env_path)

APP_NAME = "Adaptive Assessment"
APP_VERSION = "0.2.0-dev"

# ML Models
SCORE_MODEL_PATH = MODEL_DIR / "xgb_regressor.pkl"
LEVEL_MODEL_PATH = MODEL_DIR / "xgb_classifier.pkl"
ENCODER_MODEL_PATH = MODEL_DIR / "learner_level_encoder.pkl"
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
EXPECTED_EMBEDDING_DIMENSION = 384

# The classifier was trained with sklearn-style alphabetical encoding of these labels.
LEARNER_LEVEL_LABELS = (
    "Excellent",
    "Good",
    "Incorrect",
    "Partial",
    "Weak",
)

# Domain definitions for the Assessment Framework
DOMAINS = {
    "Java": {
        "status": "available",
        "description": "Object-Oriented Programming, JVM, and Core Java Concepts."
    },
    "Python": {
        "status": "coming_soon",
        "description": "Scripting, Data Science, and Web Backend Concepts."
    },
    "C++": {
        "status": "coming_soon",
        "description": "Memory Management, Pointers, and Systems Programming."
    },
    "DBMS": {
        "status": "coming_soon",
        "description": "Database Design, SQL, and Normalization."
    },
    "Operating Systems": {
        "status": "coming_soon",
        "description": "Processes, Memory, and File Systems."
    },
    "Computer Networks": {
        "status": "coming_soon",
        "description": "OSI Model, TCP/IP, and Protocols."
    }
}

# Development Dataset Info
DATASET_STATUS = "Synthetic Development Dataset"
RESEARCH_STATUS = "Preliminary Prototype"
TARGET_POPULATION = "MCA Students"

# Experimental Toggles
USE_LLM_FEEDBACK = os.environ.get("USE_LLM_FEEDBACK", "true").lower() in ("true", "1", "yes")
