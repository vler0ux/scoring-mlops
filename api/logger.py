import json
import os
import time
from datetime import datetime

LOG_FILE = os.getenv("LOG_FILE", "logs/predictions.jsonl")

def log_prediction(input_data: dict, result: dict):
    """
    Enregistre chaque prédiction dans un fichier JSONL.
    Une ligne JSON par prédiction → facile à analyser avec pandas.
    """
    os.makedirs(os.path.dirname(LOG_FILE), exist_ok=True)

    record = {
        "timestamp":  datetime.utcnow().isoformat(),
        "input":      input_data,
        "score":      result["score"],
        "decision":   result["decision"],
        "seuil":      result["seuil"],
    }

    with open(LOG_FILE, "a") as f:
        f.write(json.dumps(record) + "\n")