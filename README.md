# Scoring Crédit — Prêt à Dépenser
## Projet MLOps P8 — Mise en production du modèle de scoring

> le déploiement et le suivi en production d'un modèle de scoring.

---

## Présentation

Ce projet met en production un modèle de scoring de risque de crédit basé sur le dataset **Home Credit Default Risk**.  
Il expose le modèle via une **API Gradio**, conteneurisée avec **Docker**, avec un pipeline **CI/CD GitHub Actions** et un **dashboard de monitoring** des données de production.

- **Modèle** : LightGBM 4.6.0, entraîné et versionné dans MLflow (P6)
- **Seuil métier** : 0.519 (optimisé pour minimiser le coût métier — FN coûte 10× FP)
- **Décision** : score ≥ 0.519 → crédit refusé / score < 0.519 → crédit accordé

---

## Structure du projet

```
scoring-mlops/
│
├── api/
│   ├── app.py              # Interface Gradio + chargement modèle
│   ├── predict.py          # Logique d'inférence + seuil métier
│   └── logger.py           # Logging JSON des prédictions
│
├── tests/
│   ├── test_api.py         # Tests unitaires de l'API
│   └── test_predict.py     # Tests du pipeline d'inférence
│
├── monitoring/
│   ├── dashboard.py        # Dashboard Streamlit (scores, latence)
│   └── drift_analysis.py   # Détection de drift avec Evidently AI
│
├── scripts/
│   └── prepare_data.py     # Pipeline de nettoyage + feature engineering
│
├── data/
│   └── reference_data.csv  # Échantillon de référence pour le drift
│
├── mlflow_model/           # Artefacts du modèle champion (depuis P6)
│   ├── MLmodel
│   ├── model.pkl
│   ├── requirements.txt
│   └── conda.yaml
│
├── .github/
│   └── workflows/
│       └── ci_cd.yml       # Pipeline GitHub Actions
│
├── logs/                   # Prédictions loggées en JSONL (généré à l'exécution)
│
├── Dockerfile
├── dashboard.py
├── .gitignore
├── mlflow.db
├── .dockerignore
├── requirements.txt
└── README.md
```

## Fichiers de données (`data/`)

| Fichier | Taille | Description |
|---|---|---|
| `application_train.csv` | 159 Mo | Dataset brut d'entraînement Home Credit (307 511 clients) |
| `application_test.csv` | 26 Mo | Dataset brut de test Home Credit |
| `bureau.csv` | 163 Mo | Historique des crédits bureau pour chaque client |
| `previous_application.csv` | 387 Mo | Historique des demandes de crédit précédentes |
| `app_train_clean.parquet` | 30 Mo | Dataset nettoyé (248 colonnes) — après suppression des anomalies et encodage |
| `app_train_final.parquet` | 62 Mo | Dataset final d'entraînement (267 colonnes) — clean + features polynomiales (EXT_SOURCE^2, interactions) |
| `df_train_enrichi.csv` | 65 Mo | Dataset d'entraînement enrichi (split train) |
| `df_test_enrichi.csv` | 58 Mo | Dataset de test enrichi (split test) |
| `HomeCredit_columns_description.csv` | 37 Ko | Description de toutes les variables du dataset Home Credit |

> **Pour le monitoring drift**, le fichier de référence utilisé est `app_train_final.parquet` —
> il contient les mêmes features que celles vues par le modèle à l'entraînement,
> incluant les features polynomiales sur `EXT_SOURCE` et `DAYS_BIRTH`.
---

## Lancer l'API

### Option 1 — En local (sans Docker)

```bash
# 1. Cloner le repo
git clone https://github.com/vler0ux/scoring-mlops.git
cd scoring-mlops

# 2. Créer l'environnement et installer les dépendances
python -m venv venv
source venv/bin/activate       # Windows : venv\Scripts\activate
pip install -r requirements.txt

# 3. Lancer l'API
python api/app.py
```

L'interface est accessible sur **http://localhost:7860**

### Option 2 — Via Docker (recommandé)

```bash
# 1. Construire l'image
docker build -t scoring-api .

# 2. Lancer le conteneur
docker run -p 7860:7860 scoring-api
```

L'interface est accessible sur **http://localhost:7860**

### Option 3 — Docker Compose

```bash
docker-compose up --build
```

---

## Utilisation de l'API

L'interface Gradio demande les informations suivantes pour un client :

| Champ | Description | Exemple |
|---|---|---|
| `SK_ID_CURR` | Identifiant client | 100001 |
| `AMT_INCOME_TOTAL` | Revenu annuel (€) | 135000 |
| `AMT_CREDIT` | Montant du crédit (€) | 568800 |
| `AMT_ANNUITY` | Mensualité (€) | 20250 |
| `DAYS_BIRTH` | Âge en jours (négatif) | -12000 |
| `DAYS_EMPLOYED` | Ancienneté emploi en jours | -3000 |
| `EXT_SOURCE_1/2/3` | Scores externes de crédit | 0.5 / 0.6 / 0.7 |
| `CODE_GENDER` | Genre | M / F |
| `NAME_EDUCATION_TYPE` | Niveau d'éducation | Higher education |

**Résultat retourné :**
- **Décision** : ✅ Accordé / ❌ Refusé
- **Score** : probabilité de défaut (entre 0 et 1)
- **Risque %** : score exprimé en pourcentage
- **Seuil utilisé** : 0.519

---

## Pipeline de données

Si tu disposes de nouvelles données brutes, le script `prepare_data.py` reproduit
l'intégralité du pipeline de nettoyage et de feature engineering du projet P6 :

```bash
python scripts/prepare_data.py \
  --input  data/application_train.csv \
  --output data/reference_data.csv
```

Le fichier de sortie est utilisé comme **référence de drift** par Evidently AI.

---

## Monitoring

### Lancer le dashboard Streamlit

```bash
streamlit run monitoring/dashboard.py
```

Le dashboard affiche :
- Distribution des scores prédits en production
- Évolution de la latence de l'API
- Alertes de data drift (features vs référence d'entraînement)
- Taux de décisions accordées / refusées

### Analyse de drift

```bash
python monitoring/drift_analysis.py
```

Compare les données de production (`logs/predictions.jsonl`) avec la référence
(`data/reference_data.csv`) en utilisant **Evidently AI**.

---

## Tests

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
- Rejet des valeurs hors plage (âge négatif, revenu nul...)
- Temps de réponse de l'API

---

## CI/CD

Le pipeline GitHub Actions (`.github/workflows/ci_cd.yml`) se déclenche à chaque
push sur `main` et exécute dans l'ordre :

1. **Tests** — `pytest` sur l'ensemble des tests unitaires
2. **Build** — construction de l'image Docker si les tests passent
3. **Déploiement** — push de l'image vers Hugging Face Spaces

---

## Variables d'environnement

| Variable | Valeur par défaut | Description |
|---|---|---|
| `MODEL_URI` | `./mlflow_model` | Chemin vers les artefacts MLflow |
| `LOG_FILE` | `./logs/predictions.jsonl` | Fichier de log des prédictions |

---

## Projet source (P6)

Le modèle utilisé dans ce projet a été développé, entraîné et versionné dans le
projet P6 (MLflow) disponible ici :  
🔗 **[lien vers ton repo P6]**

- Dataset : [Home Credit Default Risk](https://www.kaggle.com/c/home-credit-default-risk)
- Modèle champion : LightGBM v4.6.0, AUC validation = **0.767**
- Alias MLflow : `champion` (version 14)

---

## Auteur

**Véronique LEROUX** — Projet MLOps P8 — OpenClassrooms  
Formation Data Scientist