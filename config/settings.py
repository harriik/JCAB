"""
Configuration settings for the JCAB Academic Prototype.
"""
from pathlib import Path

APP_NAME = "JCAB Academic Prototype"
APP_VERSION = "0.2.0-dev"

# Project structure
BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_DIR = BASE_DIR / "models"
DATA_DIR = BASE_DIR / "data"

# ML Models
SCORE_MODEL_PATH = MODEL_DIR / "xgb_regressor.pkl"
LEVEL_MODEL_PATH = MODEL_DIR / "xgb_classifier.pkl"
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
EXPECTED_EMBEDDING_DIMENSION = 384

# The classifier was trained with sklearn-style alphabetical encoding of these labels.
LEARNER_LEVEL_LABELS = (
    "Advanced",
    "Beginner",
    "Expert",
    "Intermediate",
    "Novice",
)

# Assessment topics (Controlled bank for the prototype)
TOPICS = [
    "OOP Fundamentals",
    "Inheritance",
    "Polymorphism",
    "Encapsulation",
    "Exception Handling",
    "Multithreading",
    "Collections",
    "JVM Architecture",
    "Garbage Collection",
    "Interfaces & Abstract Classes"
]

# Development Dataset Info
DATASET_STATUS = "Synthetic Development Dataset"
RESEARCH_STATUS = "Preliminary Prototype"
TARGET_POPULATION = "MCA Students"

# Experimental Toggles
USE_LLM_FEEDBACK = False  # Set to True to enable future LLM integration
