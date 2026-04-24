# tests/test_predict.py

import pytest
import sys
import os
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'api'))
from predict import load_model, predict, OPTIMAL_THRESHOLD, MODEL

# ── Chargement du modèle une seule fois pour tous les tests ──────────────────
@pytest.fixture(scope="session", autouse=True)
def setup_model():
    load_model(os.getenv("MODEL_URI", "./mlflow_model"))

# ── Profils de test ──────────────────────────────────────────────────────────
PROFIL_RISQUE = {
    "AMT_INCOME_TOTAL": 45000.0,
    "AMT_CREDIT": 500000.0,
    "AMT_ANNUITY": 35000.0,
    "DAYS_BIRTH": 8000,        # ~22 ans
    "DAYS_EMPLOYED": 365243,   # sans emploi
    "EXT_SOURCE_1": 0.05,
    "EXT_SOURCE_2": 0.05,
    "EXT_SOURCE_3": 0.05,
    "CODE_GENDER": "M",
    "NAME_EDUCATION_TYPE": "Lower secondary",
}

PROFIL_SUR = {
    "AMT_INCOME_TOTAL": 250000.0,
    "AMT_CREDIT": 200000.0,
    "AMT_ANNUITY": 8000.0,
    "DAYS_BIRTH": 18000,       # ~49 ans
    "DAYS_EMPLOYED": 5000,     # ~13 ans ancienneté
    "EXT_SOURCE_1": 0.85,
    "EXT_SOURCE_2": 0.90,
    "EXT_SOURCE_3": 0.85,
    "CODE_GENDER": "F",
    "NAME_EDUCATION_TYPE": "Higher education",
}


# ── 1. Tests fonctionnels ────────────────────────────────────────────────────
class TestFonctionnel:

    def test_profil_risque_refuse(self):
        result = predict(PROFIL_RISQUE)
        assert result["score"] >= OPTIMAL_THRESHOLD, \
            f"Profil risqué devrait être refusé, score={result['score']}"

    def test_profil_sur_accorde(self):
        result = predict(PROFIL_SUR)
        assert result["score"] < OPTIMAL_THRESHOLD, \
            f"Profil sûr devrait être accordé, score={result['score']}"

    def test_score_entre_0_et_1(self):
        result = predict(PROFIL_SUR)
        assert 0.0 <= result["score"] <= 1.0

    def test_risque_pct_coherent(self):
        result = predict(PROFIL_SUR)
        assert result["risque_pct"] == round(result["score"] * 100, 1)


# ── 2. Tests entrées invalides ───────────────────────────────────────────────
class TestEntreesInvalides:

    def test_revenu_nul(self):
        data = PROFIL_SUR.copy()
        data["AMT_INCOME_TOTAL"] = 0
        with pytest.raises(Exception):
            predict(data)

    def test_revenu_negatif(self):
        data = PROFIL_SUR.copy()
        data["AMT_INCOME_TOTAL"] = -5000
        with pytest.raises(Exception):
            predict(data)

    def test_age_trop_jeune(self):
        data = PROFIL_SUR.copy()
        data["DAYS_BIRTH"] = 3000  # ~8 ans
        with pytest.raises(Exception):
            predict(data)

    def test_ext_source_hors_plage(self):
        data = PROFIL_SUR.copy()
        data["EXT_SOURCE_2"] = 1.5  # > 1
        with pytest.raises(Exception):
            predict(data)

    def test_mensualite_superieure_revenu(self):
        data = PROFIL_SUR.copy()
        data["AMT_ANNUITY"] = 300000  # > revenu annuel
        with pytest.raises(Exception):
            predict(data)

    def test_genre_invalide(self):
        data = PROFIL_SUR.copy()
        data["CODE_GENDER"] = "X"
        with pytest.raises(Exception):
            predict(data)


# ── 3. Tests valeurs manquantes ──────────────────────────────────────────────
class TestValeursManquantes:

    def test_ext_source_3_none(self):
        data = PROFIL_SUR.copy()
        data["EXT_SOURCE_3"] = None
        result = predict(data)
        assert 0.0 <= result["score"] <= 1.0

    def test_ext_source_1_et_3_none(self):
        data = PROFIL_SUR.copy()
        data["EXT_SOURCE_1"] = None
        data["EXT_SOURCE_3"] = None
        result = predict(data)
        assert 0.0 <= result["score"] <= 1.0


# ── 4. Tests seuil métier ────────────────────────────────────────────────────
class TestSeuilMetier:

    def test_seuil_valeur(self):
        assert OPTIMAL_THRESHOLD == 0.519

    def test_decision_accordee(self):
        result = predict(PROFIL_SUR)
        if result["score"] < OPTIMAL_THRESHOLD:
            assert result["decision"] == "✅ Accordé"

    def test_decision_refusee(self):
        result = predict(PROFIL_RISQUE)
        if result["score"] >= OPTIMAL_THRESHOLD:
            assert result["decision"] == "❌ Refusé"


# ── 5. Tests performance ─────────────────────────────────────────────────────
class TestPerformance:

    def test_temps_inference_inferieur_1s(self):
        start = time.time()
        predict(PROFIL_SUR)
        elapsed = time.time() - start
        assert elapsed < 1.0, f"Temps d'inférence trop long : {elapsed:.2f}s"

    def test_modele_charge_une_seule_fois(self):
        from predict import MODEL
        assert MODEL is not None, "Le modèle doit être chargé au démarrage"