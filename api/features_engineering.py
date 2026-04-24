import pandas as pd

def compute_features(df: pd.DataFrame) -> pd.DataFrame:
    """Calcule toutes les features dérivées et polynomial features."""

    # Features dérivées
    df['CREDIT_INCOME_PERCENT']  = df['AMT_CREDIT']    / df['AMT_INCOME_TOTAL']
    df['ANNUITY_INCOME_PERCENT'] = df['AMT_ANNUITY']   / df['AMT_INCOME_TOTAL']
    df['CREDIT_TERM']            = df['AMT_ANNUITY']   / df['AMT_CREDIT']
    df['DAYS_EMPLOYED_PERCENT']  = df['DAYS_EMPLOYED'] / df['DAYS_BIRTH']
    df['AGE_YEARS']              = df['DAYS_BIRTH']    / 365.25
    df['EMPLOYED_YEARS']         = df['DAYS_EMPLOYED'] / 365.25
    df['EXT_SOURCE_MEAN']        = df[['EXT_SOURCE_1', 'EXT_SOURCE_2', 'EXT_SOURCE_3']].mean(axis=1)
    df['EXT_SOURCE_MIN']         = df[['EXT_SOURCE_1', 'EXT_SOURCE_2', 'EXT_SOURCE_3']].min(axis=1)
    df['ANNUITY_INCOME_RATIO']   = df['AMT_ANNUITY']  / df['AMT_INCOME_TOTAL']
    if 'AMT_GOODS_PRICE' in df.columns and df['AMT_GOODS_PRICE'].notna().all():
        df['CREDIT_GOODS_RATIO'] = df['AMT_CREDIT'] / df['AMT_GOODS_PRICE']
    else:
        df['CREDIT_GOODS_RATIO'] = None  # remplacé par la moyenne de FEATURE_MEANS
    df['CREDIT_INCOME_RATIO']    = df['AMT_CREDIT']   / df['AMT_INCOME_TOTAL']

    # Polynomial features
    df['EXT_SOURCE_1^2']            = df['EXT_SOURCE_1'] ** 2
    df['EXT_SOURCE_1 EXT_SOURCE_2'] = df['EXT_SOURCE_1'] * df['EXT_SOURCE_2']
    df['EXT_SOURCE_1 EXT_SOURCE_3'] = df['EXT_SOURCE_1'] * df['EXT_SOURCE_3']
    df['EXT_SOURCE_1 DAYS_BIRTH']   = df['EXT_SOURCE_1'] * df['DAYS_BIRTH']
    df['EXT_SOURCE_1 AGE_YEARS']    = df['EXT_SOURCE_1'] * df['AGE_YEARS']
    df['EXT_SOURCE_2^2']            = df['EXT_SOURCE_2'] ** 2
    df['EXT_SOURCE_2 EXT_SOURCE_3'] = df['EXT_SOURCE_2'] * df['EXT_SOURCE_3']
    df['EXT_SOURCE_2 DAYS_BIRTH']   = df['EXT_SOURCE_2'] * df['DAYS_BIRTH']
    df['EXT_SOURCE_2 AGE_YEARS']    = df['EXT_SOURCE_2'] * df['AGE_YEARS']
    df['EXT_SOURCE_3^2']            = df['EXT_SOURCE_3'] ** 2
    df['EXT_SOURCE_3 DAYS_BIRTH']   = df['EXT_SOURCE_3'] * df['DAYS_BIRTH']
    df['EXT_SOURCE_3 AGE_YEARS']    = df['EXT_SOURCE_3'] * df['AGE_YEARS']
    df['DAYS_BIRTH^2']              = df['DAYS_BIRTH'] ** 2
    df['DAYS_BIRTH AGE_YEARS']      = df['DAYS_BIRTH'] * df['AGE_YEARS']
    df['AGE_YEARS^2']               = df['AGE_YEARS'] ** 2

    return df