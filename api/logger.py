import json
import os
from datetime import datetime, timezone

LOG_FILE = os.getenv("LOG_FILE", "logs/predictions.jsonl")

def log_prediction(input_data: dict, result: dict, inference_time_ms: float = None):
    """
    Enregistre chaque prédiction dans un fichier JSONL.
    Une ligne JSON par prédiction → facile à analyser avec pandas.
    """
    os.makedirs(os.path.dirname(LOG_FILE), exist_ok=True)

    record = {
        "timestamp":  datetime.now(timezone.utc).isoformat(),
        "input":      input_data,
        "score":      result["score"],
        "decision":   result["decision"],
        "seuil":      result["seuil"],
        "inference_time_ms": inference_time_ms,
    }

    with open(LOG_FILE, "a") as f:
        f.write(json.dumps(record) + "\n")