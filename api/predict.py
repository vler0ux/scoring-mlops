import mlflow.pyfunc
import pandas as pd
import numpy as np
import json
import os

OPTIMAL_THRESHOLD = 0.519
MODEL = None

FEATURES_PATH = os.path.join(os.path.dirname(__file__), "feature_columns.json")
with open(FEATURES_PATH) as f:
    FEATURE_COLUMNS = json.load(f)
    
EDUCATION_COLS = [
    "NAME_EDUCATION_TYPE_Academic degree",
    "NAME_EDUCATION_TYPE_Higher education",
    "NAME_EDUCATION_TYPE_Incomplete higher",
    "NAME_EDUCATION_TYPE_Lower secondary",
    "NAME_EDUCATION_TYPE_Secondary / secondary special",
]

def load_model(model_uri: str):
    global MODEL
    MODEL = mlflow.pyfunc.load_model(model_uri)
    print(f"✅ Modèle chargé depuis : {model_uri}")


def predict(input_data: dict) -> dict:
    if MODEL is None:
        raise RuntimeError("Modèle non chargé.")

    df = pd.DataFrame([input_data])

    df['CODE_GENDER_1'] = (df['CODE_GENDER'] == 'M').astype(int)
    df = df.drop(columns=['CODE_GENDER'])

    education_val = df['NAME_EDUCATION_TYPE'].iloc[0]
    for col in EDUCATION_COLS:
        # col = "NAME_EDUCATION_TYPE_Higher education"
        # on extrait la modalité après le préfixe
        modalite = col.replace("NAME_EDUCATION_TYPE_", "")
        df[col] = 1 if education_val == modalite else 0
    df = df.drop(columns=['NAME_EDUCATION_TYPE'])

    df['CREDIT_INCOME_PERCENT']  = df['AMT_CREDIT']    / df['AMT_INCOME_TOTAL']
    df['ANNUITY_INCOME_PERCENT'] = df['AMT_ANNUITY']   / df['AMT_INCOME_TOTAL']
    df['CREDIT_TERM']            = df['AMT_ANNUITY']   / df['AMT_CREDIT']
    df['DAYS_EMPLOYED_PERCENT']  = df['DAYS_EMPLOYED'] / df['DAYS_BIRTH']
    
    for col in FEATURE_COLUMNS:
        if col not in df.columns:
            df[col] = np.nan

    # Garder uniquement les colonnes du modèle, dans le bon ordre
    df = df[FEATURE_COLUMNS]

    # ── Prédiction 
    proba = MODEL.predict(df)
    score = float(proba[0]) if proba.ndim == 1 else float(proba[0][1])

    decision = "❌ Refusé" if score >= OPTIMAL_THRESHOLD else "✅ Accordé"

    return {
        "score":      round(score, 4),
        "seuil":      OPTIMAL_THRESHOLD,
        "decision":   decision,
        "risque_pct": round(score * 100, 1)
    }