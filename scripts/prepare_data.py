import numpy as np
import pandas as pd
import argparse
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import PolynomialFeatures


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Pipeline complet de nettoyage + feature engineering (extrait des notebooks P6)."""

    df = df.copy()

    # ── 1. Nettoyage ──────────────────────────────────────────────────────────

    # CODE_GENDER : remplacer XNA par le mode
    mode_gender = df['CODE_GENDER'].mode()[0]
    df['CODE_GENDER'] = df['CODE_GENDER'].replace('XNA', mode_gender)

    # DAYS_EMPLOYED : anomalie 365243 (sans-emploi) → NaN
    df['DAYS_EMPLOYED'].replace({365243: np.nan}, inplace=True)

    # DAYS_BIRTH : valeur absolue
    df['DAYS_BIRTH'] = abs(df['DAYS_BIRTH'])

    # LabelEncoding des colonnes binaires (2 modalités)
    le = LabelEncoder()
    for col in df.columns:
        if df[col].dtype == 'object' and len(df[col].unique()) <= 2:
            le.fit(df[col])
            df[col] = le.transform(df[col])

    # OHE explicite pour CODE_GENDER
    ohe = OneHotEncoder(sparse_output=False, handle_unknown='ignore', drop='first')
    gender_encoded = ohe.fit_transform(df[['CODE_GENDER']])
    gender_cols = ohe.get_feature_names_out(['CODE_GENDER'])
    df = df.drop(columns=['CODE_GENDER'])
    df = pd.concat(
        [df, pd.DataFrame(gender_encoded, columns=gender_cols, index=df.index)],
        axis=1
    )

    # get_dummies pour les autres colonnes catégorielles restantes
    df = pd.get_dummies(df)

    # ── 2. Feature Engineering ────────────────────────────────────────────────

    # Polynomial features sur EXT_SOURCE + DAYS_BIRTH
    poly_cols = ['EXT_SOURCE_1', 'EXT_SOURCE_2', 'EXT_SOURCE_3', 'DAYS_BIRTH']
    poly_cols_present = [c for c in poly_cols if c in df.columns]

    imputer = SimpleImputer(strategy='median')
    poly_input = pd.DataFrame(
        imputer.fit_transform(df[poly_cols_present]),
        columns=poly_cols_present,
        index=df.index
    )

    poly_transformer = PolynomialFeatures(degree=2)
    poly_array = poly_transformer.fit_transform(poly_input)
    poly_names = poly_transformer.get_feature_names_out(poly_cols_present)
    poly_df = pd.DataFrame(poly_array, columns=poly_names, index=df.index)

    # Ajouter seulement les nouvelles colonnes (éviter les doublons)
    new_cols = [c for c in poly_names if c not in df.columns and c != '1']
    df = pd.concat([df, poly_df[new_cols]], axis=1)

    # Domain knowledge features (ratios métier)
    df['CREDIT_INCOME_PERCENT']  = df['AMT_CREDIT']  / df['AMT_INCOME_TOTAL']
    df['ANNUITY_INCOME_PERCENT'] = df['AMT_ANNUITY'] / df['AMT_INCOME_TOTAL']
    df['CREDIT_TERM']            = df['AMT_ANNUITY'] / df['AMT_CREDIT']
    df['DAYS_EMPLOYED_PERCENT']  = df['DAYS_EMPLOYED'] / df['DAYS_BIRTH']

    return df


def main(input_path: str, output_path: str):
    print(f"Chargement de {input_path}...")
    df_raw = pd.read_csv(input_path)

    print("Nettoyage et feature engineering...")
    df_clean = clean_data(df_raw)

    # Retirer TARGET si présente (on ne veut que les features)
    df_out = df_clean.drop(columns=['TARGET'], errors='ignore')

    # Échantillon représentatif pour le monitoring drift
    sample = df_out.sample(n=min(1000, len(df_out)), random_state=42)
    sample.to_csv(output_path, index=False)
    print(f"✓ {output_path} généré ({len(sample)} lignes, {df_out.shape[1]} colonnes)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prépare les données pour le monitoring drift")
    parser.add_argument("--input",  default="data/raw_data.csv",          help="Fichier CSV brut")
    parser.add_argument("--output", default="data/reference_data.csv",    help="Fichier de sortie")
    args = parser.parse_args()
    main(args.input, args.output)