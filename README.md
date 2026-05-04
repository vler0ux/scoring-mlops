# Scoring Crédit — Prêt à Dépenser
## Projet MLOps P8 — Mise en production du modèle de scoring

> Déploiement et suivi en production d'un modèle de scoring de risque de crédit.

---

## Présentation

Ce projet met en production un modèle de scoring basé sur le dataset **Home Credit Default Risk**.
Il expose le modèle via une **API Gradio**, conteneurisée avec **Docker**, avec un pipeline **CI/CD GitHub Actions** et un **dashboard de monitoring** Streamlit + Evidently AI.

- **Modèle** : LightGBM 4.6.0, entraîné et versionné dans MLflow (P6)
- **Seuil métier** : 0.519 (optimisé pour minimiser le coût métier — FN coûte 10× FP)
- **Décision** : score ≥ 0.519 → crédit refusé / score < 0.519 → crédit accordé

---

## Structure du projet

```
scoring-mlops/
├── api/
│   ├── app.py              # Interface Gradio + chargement modèle
│   ├── predict.py          # Logique d'inférence + seuil métier
│   └── logger.py           # Logging JSON des prédictions
├── tests/
│   ├── test_api.py         # Tests unitaires de l'API
│   └── test_predict.py     # Tests du pipeline d'inférence
├── monitoring/
│   └── drift_analysis.py   # Détection de drift standalone avec Evidently AI
├── scripts/
│   └── prepare_data.py     # Pipeline de nettoyage + feature engineering
├── data/
│   └── app_train_final.parquet  # Référence de drift (features d'entraînement)
├── mlflow_model/           # Artefacts du modèle champion (depuis P6)
│   ├── MLmodel
│   ├── model.pkl
│   ├── requirements.txt
│   └── conda.yaml
├── .github/
│   └── workflows/
│       └── ci_cd.yml       # Pipeline GitHub Actions
├── logs/                   # Prédictions loggées en JSONL (généré à l'exécution)
├── dashboard.py            # Dashboard Streamlit de monitoring
├── Dockerfile
├── requirements.txt
└── README.md
```

---

## Installation

```bash
git clone https://github.com/vler0ux/scoring-mlops.git
cd scoring-mlops
python -m venv venv
source venv/bin/activate    # Windows : venv\Scripts\activate
pip install -r requirements.txt
```

---

## 1. Lancer l'API Gradio

L'API expose une interface Gradio pour scorer un client et loggue chaque prédiction dans `logs/predictions.jsonl`.

### En local

```bash
source venv/bin/activate
python api/app.py
```

Interface accessible sur **http://localhost:7860**

### Via Docker

```bash
# Construire l'image
docker build -t scoring-api .

# Lancer (sans persistance des logs)
docker run -p 7860:7860 scoring-api

# Lancer avec persistance des logs sur la machine hôte
docker run -p 7860:7860 -v $(pwd)/logs:/app/logs scoring-api
```

Interface accessible sur **http://localhost:7860**

### Champs du formulaire

| Champ | Description | Exemple |
|---|---|---|
| Revenu annuel (€) | `AMT_INCOME_TOTAL` | 135 000 |
| Montant du crédit (€) | `AMT_CREDIT` | 200 000 |
| Mensualité (€) | `AMT_ANNUITY` | 8 000 |
| Âge (années) | converti en `DAYS_BIRTH` | 35 |
| Ancienneté emploi (années) | converti en `DAYS_EMPLOYED` | 8 |
| Sans emploi / Retraité | force `DAYS_EMPLOYED = 365243` | case à cocher |
| EXT_SOURCE_1/2/3 | Scores externes de crédit (0–1) | 0.5 / 0.6 / 0.7 |
| Genre | `CODE_GENDER` | M / F |
| Niveau d'éducation | `NAME_EDUCATION_TYPE` | Higher education |

**Résultat retourné :**
- **Décision** : ✅ Accordé / ❌ Refusé
- **Score** : probabilité de défaut (entre 0 et 1)
- **Risque %** : score exprimé en pourcentage
- **Seuil utilisé** : 0.519

---

## 2. Logs des prédictions (JSONL)

Chaque prédiction est enregistrée dans `logs/predictions.jsonl`, une ligne par requête :

```json
{
  "timestamp": "2025-01-15T14:32:01.123456+00:00",
  "input": {
    "AMT_INCOME_TOTAL": 135000.0,
    "AMT_CREDIT": 200000.0,
    "AMT_ANNUITY": 8000.0,
    "DAYS_BIRTH": 12783.0,
    "DAYS_EMPLOYED": 2922.0,
    "EXT_SOURCE_1": 0.5,
    "EXT_SOURCE_2": 0.6,
    "EXT_SOURCE_3": 0.7,
    "CODE_GENDER": "M",
    "NAME_EDUCATION_TYPE": "Higher education"
  },
  "score": 0.312,
  "decision": "✅ Crédit ACCORDÉ",
  "seuil": 0.519,
  "inference_time_ms": 42.7
}
```

Ce fichier est la source de données du dashboard de monitoring.

---

## 3. Dashboard Streamlit + Evidently

Le dashboard lit `logs/predictions.jsonl` et affiche :
- Distribution des scores et évolution temporelle
- Latence de l'API (moyenne, p95, alerte configurable)
- Taux de décisions accordées / refusées
- Indicateur de dérive temporelle (1ère vs 2ème moitié des requêtes)
- Rapport Evidently AI complet (drift vs données d'entraînement)

> **Prérequis** : avoir lancé l'API et effectué au moins quelques prédictions pour que `logs/predictions.jsonl` existe. En l'absence de logs, le dashboard s'affiche en **mode démo** avec des données synthétiques.

```bash
source venv/bin/activate
streamlit run dashboard.py
```

Dashboard accessible sur **http://localhost:8501**

### Analyse de drift standalone

Pour générer un rapport Evidently sans lancer le dashboard :

```bash
python monitoring/drift_analysis.py
```

Compare `logs/predictions.jsonl` avec `data/app_train_final.parquet` (référence d'entraînement).

---

## 4. Tests

```bash
# Lancer tous les tests
pytest tests/ -v

# Avec rapport de couverture
pytest tests/ -v --cov=api --cov-report=term-missing
```

Les tests couvrent :
- Prédiction avec des données valides
- Gestion des valeurs manquantes
- Rejet des types incorrects
- Rejet des valeurs hors plage
- Temps de réponse de l'API

---

## 5. CI/CD

Le pipeline GitHub Actions (`.github/workflows/ci_cd.yml`) se déclenche à chaque push sur `main` :

1. **test** — `pytest` sur l'ensemble des tests unitaires
2. **build** — construction et validation de l'image Docker
3. **deploy** — push du code vers Hugging Face Spaces (déclenche le build Docker côté HF)

**Secret requis** : `HF_TOKEN` (token Hugging Face avec droits write) à configurer dans GitHub → Settings → Secrets → Actions.

---

## Variables d'environnement

| Variable | Valeur par défaut | Description |
|---|---|---|
| `MODEL_URI` | `./mlflow_model` | Chemin vers les artefacts MLflow |
| `LOG_FILE` | `./logs/predictions.jsonl` | Fichier de log des prédictions |

```bash
MODEL_URI=./mlflow_model LOG_FILE=./logs/predictions.jsonl python api/app.py
```

---

## Pipeline de données

Pour régénérer le fichier de référence de drift à partir de nouvelles données brutes :

```bash
python scripts/prepare_data.py \
  --input  data/application_train.csv \
  --output data/app_train_final.parquet
```

---

## Projet source (P6)

- Dataset : [Home Credit Default Risk](https://www.kaggle.com/c/home-credit-default-risk)
- Modèle champion : LightGBM v4.6.0, AUC validation = **0.767**
- Alias MLflow : `champion` (version 14)
- Repo P6 : **[lien vers ton repo P6]**

---

## Auteur

**Véronique LEROUX** — Projet MLOps P8 — OpenClassrooms — Formation Data Scientist
