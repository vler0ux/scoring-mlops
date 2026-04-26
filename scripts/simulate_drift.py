import sys, os, random, time
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../api"))

from predict import load_model, predict
from logger import log_prediction

MODEL_URI = os.path.join(os.path.dirname(__file__), "../mlflow_model")
load_model(MODEL_URI)

random.seed(42)
np.random.seed(42)

EDUCATIONS = [
    "Higher education",
    "Secondary / secondary special",
    "Incomplete higher",
    "Lower secondary",
    "Academic degree",
]

n = 200
for i in range(n):
    drift_factor = i / n

    # AMT_INCOME_TOTAL : drift vers le haut (45k → 150k)
    income = np.random.normal(45000 + drift_factor * 105000, 15000)
    income = max(income, 10000)

    # AMT_CREDIT : proportionnel au revenu
    credit = income * np.random.uniform(1.5, 4.0)

    # AMT_ANNUITY : ~10% du crédit par an
    annuity = credit / np.random.uniform(8, 15)

    # Âge : drift vers des clients plus jeunes (35 ans → 22 ans)
    age_ans = max(18, np.random.normal(35 - drift_factor * 13, 5))
    days_birth = float(int(age_ans * 365.25))

    # Ancienneté
    anciennete_ans = np.random.uniform(0, min(age_ans - 18, 20))
    days_employed = float(int(anciennete_ans * 365.25))

    # EXT_SOURCE_2 : drift vers le bas (0.55 → 0.30)
    ext2 = np.clip(np.random.normal(0.55 - drift_factor * 0.25, 0.1), 0, 1)
    ext1 = np.clip(np.random.normal(0.50, 0.15), 0, 1)
    ext3 = np.clip(np.random.normal(0.55, 0.12), 0, 1)

    input_data = {
        "AMT_INCOME_TOTAL":    round(float(income), 2),
        "AMT_CREDIT":          round(float(credit), 2),
        "AMT_ANNUITY":         round(float(annuity), 2),
        "DAYS_BIRTH":          days_birth,
        "DAYS_EMPLOYED":       days_employed,
        "EXT_SOURCE_1":        round(float(ext1), 4),
        "EXT_SOURCE_2":        round(float(ext2), 4),
        "EXT_SOURCE_3":        round(float(ext3), 4),
        "CODE_GENDER":         random.choice(["M", "F"]),
        "NAME_EDUCATION_TYPE": random.choice(EDUCATIONS),
    }

    try:
        t0 = time.time()
        result = predict(input_data)
        inference_time_ms = round((time.time() - t0) * 1000, 2)

        log_prediction(input_data, result, inference_time_ms)

        if i % 20 == 0:
            print(f"[{i+1}/200] income={income:.0f}€ | score={result['score']:.3f} | "
                f"latence={inference_time_ms:.1f}ms | {result['decision']}")
    except Exception as e:
        print(f"[{i+1}] ⚠️ Erreur : {e}")

print(f"\n✅ Simulation terminée — logs dans logs/predictions.jsonl")