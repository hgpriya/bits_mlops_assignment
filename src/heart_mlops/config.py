"""Central configuration: paths, column schema and feature groups."""
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
FIGURES_DIR = PROJECT_ROOT / "reports" / "figures"

UCI_BASE_URL = (
    "https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/"
)
RAW_FILE_NAME = "processed.cleveland.data"
RAW_CSV = RAW_DIR / "heart_cleveland_raw.csv"
CLEAN_CSV = PROCESSED_DIR / "heart_clean.csv"

# The 14 attributes used by the UCI Cleveland subset, in file order.
COLUMNS = [
    "age", "sex", "cp", "trestbps", "chol", "fbs", "restecg",
    "thalach", "exang", "oldpeak", "slope", "ca", "thal", "num",
]

TARGET = "target"          # binary label we create from `num`
RAW_TARGET = "num"         # original 0-4 severity label

NUMERIC_FEATURES = ["age", "trestbps", "chol", "thalach", "oldpeak", "ca"]
BINARY_FEATURES = ["sex", "fbs", "exang"]
CATEGORICAL_FEATURES = ["cp", "restecg", "slope", "thal"]
ALL_FEATURES = NUMERIC_FEATURES + BINARY_FEATURES + CATEGORICAL_FEATURES

# Human-readable labels used in plots and the report.
FEATURE_LABELS = {
    "age": "Age (years)",
    "sex": "Sex (1 = male)",
    "cp": "Chest pain type",
    "trestbps": "Resting BP (mm Hg)",
    "chol": "Serum cholesterol (mg/dl)",
    "fbs": "Fasting sugar > 120 mg/dl",
    "restecg": "Resting ECG result",
    "thalach": "Max heart rate achieved",
    "exang": "Exercise-induced angina",
    "oldpeak": "ST depression (oldpeak)",
    "slope": "Slope of peak ST segment",
    "ca": "Major vessels coloured (0-3)",
    "thal": "Thalassemia test result",
    "target": "Heart disease (1 = present)",
}

# Clinically plausible ranges - values outside are treated as data errors.
VALID_RANGES = {
    "age": (18, 100),
    "trestbps": (70, 220),
    "chol": (100, 600),
    "thalach": (60, 220),
    "oldpeak": (0, 7),
    "ca": (0, 3),
}

# ---------------------------------------------------------------- modelling
MODELS_DIR = PROJECT_ROOT / "models"
MODEL_PATH = MODELS_DIR / "model.joblib"
MODEL_CARD_PATH = MODELS_DIR / "model_card.json"

SEED = 42
TEST_SIZE = 0.2
CV_FOLDS = 5
SELECTION_METRIC = "roc_auc"

MLFLOW_EXPERIMENT = "heart-disease-risk"
# SQLite backend: supports the MLflow UI and model registry without a server.
MLFLOW_TRACKING_URI = f"sqlite:///{(PROJECT_ROOT / 'mlflow.db').as_posix()}"
