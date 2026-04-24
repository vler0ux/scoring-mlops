import mlflow.pyfunc
import mlflow.lightgbm
import pandas as pd
import numpy as np
import json
import os

from features_engineering import compute_features

OPTIMAL_THRESHOLD = 0.519
MODEL = None

FEATURES_PATH = os.path.join(os.path.dirname(__file__), "feature_columns.json")
with open(FEATURES_PATH) as f:
    FEATURE_COLUMNS = json.load(f)

MEANS_PATH = os.path.join(os.path.dirname(__file__), "feature_means.json")
with open(MEANS_PATH) as f:
    FEATURE_MEANS = json.load(f)

EDUCATION_COLS = [
    "NAME_EDUCATION_TYPE_Academic degree",
    "NAME_EDUCATION_TYPE_Higher education",
    "NAME_EDUCATION_TYPE_Incomplete higher",
    "NAME_EDUCATION_TYPE_Lower secondary",
    "NAME_EDUCATION_TYPE_Secondary / secondary special",
]

def load_model(model_uri: str):
    global MODEL
    MODEL = mlflow.lightgbm.load_model(model_uri)  # ← mlflow.lightgbm au lieu de mlflow.pyfunc
    print(f"✅ Modèle chargé : {type(MODEL)}")


def predict(input_data: dict) -> dict:
    if MODEL is None:
        raise RuntimeError("Modèle non chargé.")

    df = pd.DataFrame([input_data])

    df['CODE_GENDER_1'] = (df['CODE_GENDER'] == 'M').astype(int)
    df = df.drop(columns=['CODE_GENDER'])

    education_val = df['NAME_EDUCATION_TYPE'].iloc[0]
    for col in EDUCATION_COLS:
        modalite = col.replace("NAME_EDUCATION_TYPE_", "")
        df[col] = 1 if education_val == modalite else 0
    df = df.drop(columns=['NAME_EDUCATION_TYPE'])

    # APRÈS — part des moyennes, écrase avec les valeurs saisies
    df_full = pd.DataFrame([FEATURE_MEANS.copy()])
    for col in df.columns:
        if col in df_full.columns:
            df_full[col] = df[col].values[0]
    df = df_full

    df = compute_features(df)

    # Garder uniquement les colonnes du modèle, dans le bon ordre
    df = df[FEATURE_COLUMNS]

    # ── Prédiction
    proba_raw = MODEL.predict_proba(df)
    print(f"DEBUG proba_raw = {proba_raw}")
    score = float(proba_raw[0][1]) if proba_raw.ndim == 1 else float(proba_raw[0][1])

    decision = "❌ Refusé" if score >= OPTIMAL_THRESHOLD else "✅ Accordé"

    return {
        "score":      round(score, 4),
        "seuil":      OPTIMAL_THRESHOLD,
        "decision":   decision,
        "risque_pct": round(score * 100, 1)
    }