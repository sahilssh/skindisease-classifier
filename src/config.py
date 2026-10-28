import os
from pathlib import Path

# Paths
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
MODELS_DIR = BASE_DIR / "models"

for directory in [DATA_DIR, RAW_DATA_DIR, PROCESSED_DATA_DIR, MODELS_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

# Image settings
IMAGE_SIZE = (224, 224)
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]

# HAM10000 7 Primary Diagnostic Categories
CLASSES = ["akiec", "bcc", "bkl", "df", "mel", "nv", "vasc"]
NUM_CLASSES = len(CLASSES)

CLASS_TO_IDX = {cls_name: idx for idx, cls_name in enumerate(CLASSES)}
IDX_TO_CLASS = {idx: cls_name for idx, cls_name in enumerate(CLASSES)}

# 8th Category for OOD / Low Confidence / Clear Skin
NOT_DETECTED_CLASS = "not_detected"
DEFAULT_CONFIDENCE_THRESHOLD = 0.55  # If max probability < 55%, mark as "Not Detected"

# Human-readable labels and descriptions
CLASS_NAMES = {
    "akiec": "Actinic Keratosis / Bowen’s Disease (akiec)",
    "bcc": "Basal Cell Carcinoma (bcc)",
    "bkl": "Benign Keratosis-like Lesions (bkl)",
    "df": "Dermatofibroma (df)",
    "mel": "Melanoma (mel)",
    "nv": "Melanocytic Nevus / Common Mole (nv)",
    "vasc": "Vascular Lesion (vasc)",
    "not_detected": "Not Detected / Inconclusive Lesion",
}

CLASS_RISK_LEVEL = {
    "akiec": {"risk": "Pre-cancerous", "severity": "warning", "action": "Consult a dermatologist for evaluation and potential cryotherapy/topical treatment."},
    "bcc": {"risk": "Malignant", "severity": "danger", "action": "Consult a dermatologist. Most common skin cancer; generally slow growing with high cure rates if treated early."},
    "bkl": {"risk": "Benign", "severity": "success", "action": "Harmless skin lesion (such as seborrheic keratosis). Regular routine skin self-checks recommended."},
    "df": {"risk": "Benign", "severity": "success", "action": "Benign cutaneous fibrous histiocytoma. No medical intervention typically needed unless symptomatic."},
    "mel": {"risk": "High-Risk Malignant", "severity": "danger", "action": "URGENT: High-risk malignant skin cancer. Prompt evaluation by an oncologist/dermatologist is strongly advised."},
    "nv": {"risk": "Benign", "severity": "success", "action": "Normal benign melanocytic mole. Monitor for changes following the ABCDE criteria."},
    "vasc": {"risk": "Benign", "severity": "success", "action": "Benign vascular growth (e.g. cherry angioma, pyogenic granuloma). Generally harmless."},
    "not_detected": {"risk": "Undetected / Inconclusive", "severity": "neutral", "action": "No specific skin lesion detected or confidence is too low to classify. Ensure good lighting, clear focus, and close framing."},
}

CLASS_DESCRIPTIONS = {
    "akiec": "Actinic keratoses and intraepithelial carcinoma (Bowen's disease) are common non-invasive variants of squamous cell carcinoma that can progress to invasive SCC if untreated.",
    "bcc": "Basal cell carcinoma is the most common form of skin cancer, arising from basal cells in the deepest layer of the epidermis. Rarely metastasizes, but can cause local tissue damage.",
    "bkl": "Benign keratosis-like lesions include solar lentigines (sun spots), seborrheic keratoses, and lichen-planus like keratoses. Common in older adults and completely non-cancerous.",
    "df": "Dermatofibromas are benign fibrous skin nodules that often develop on the lower legs, often feeling like firm small bumps under the skin surface.",
    "mel": "Melanoma develops from melanocytes and is the most dangerous form of skin cancer. Early detection and surgical excision offer the highest cure rates.",
    "nv": "Melanocytic nevi are common benign moles formed by clusters of pigmented melanocytes. Most adults have between 10 and 40 nevi.",
    "vasc": "Vascular lesions encompass cherry angiomas, angiokeratomas, pyogenic granulomas, and cutaneous hemorrhages resulting from blood vessel proliferation.",
    "not_detected": "The uploaded image did not meet the confidence threshold for any of the 7 known HAM10000 lesion categories. It may be healthy skin, an unrepresented condition, or blurry/non-dermatoscopic input.",
}

# Training Hyperparameters
DEFAULT_TRAIN_CONFIG = {
    "model_name": "mobilenetv3_large_100",  # Lightweight and fast for Streamlit hosting
    "batch_size": 32,
    "num_epochs": 15,
    "learning_rate": 3e-4,
    "weight_decay": 1e-4,
    "num_workers": 2,
    "seed": 42,
}
