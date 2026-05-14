"""
Preprocesamiento — Adult Census Income Dataset
Limpieza, encoding, StandardScaler, split 80/20
Exporta data/processed/X_train.csv, X_test.csv, y_train.csv, y_test.csv
"""
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
import os

BASE_DIR  = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_RAW  = os.path.join(BASE_DIR, 'data', 'raw', 'adult.csv')
DATA_PROC = os.path.join(BASE_DIR, 'data', 'processed')
os.makedirs(DATA_PROC, exist_ok=True)

RANDOM_STATE = 42

# ---------------------------------------------------------------------------
print("Cargando dataset raw...")
df = pd.read_csv(DATA_RAW)
print(f"Shape inicial: {df.shape}")

# ---------------------------------------------------------------------------
# 1. Limpieza
n_dup = df.duplicated().sum()
df = df.drop_duplicates()
print(f"Duplicados eliminados: {n_dup}")

n_nulos_total = df.isnull().sum().sum()
df = df.dropna()
print(f"Filas con nulos eliminadas (valores nulos presentes: {n_nulos_total})")
print(f"Shape tras limpieza: {df.shape}")

# ---------------------------------------------------------------------------
# 2. Seleccion de features
# fnlwgt: peso censal, no es una variable predictiva real
# education: redundante con education_num (version ordinal)
df = df.drop(columns=['fnlwgt', 'education'])
print(f"\nColumnas tras eliminar redundantes: {list(df.columns)}")

# ---------------------------------------------------------------------------
# 3. Encoding del target
# <=50K -> 0  |  >50K -> 1
df['income'] = df['income'].str.strip().str.rstrip('.')
df['income'] = (df['income'] == '>50K').astype(int)
print(f"\nBalance del target:\n{df['income'].value_counts()}")
print(df['income'].value_counts(normalize=True).round(3))

# ---------------------------------------------------------------------------
# 4. Encoding de categoricas con LabelEncoder
CATEG_COLS = ['workclass', 'marital_status', 'occupation',
              'relationship', 'race', 'sex', 'native_country']

le = LabelEncoder()
for col in CATEG_COLS:
    df[col] = le.fit_transform(df[col].astype(str))

print(f"\nFeatures finales: {[c for c in df.columns if c != 'income']}")

# ---------------------------------------------------------------------------
# 5. Split features / target
X = df.drop(columns=['income'])
y = df['income']

print(f"\nShape X: {X.shape}  |  Shape y: {y.shape}")

# ---------------------------------------------------------------------------
# 6. Train / test split — 80/20, estratificado, semilla fija
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
)
print(f"\nX_train: {X_train.shape}  X_test: {X_test.shape}")
print(f"y_train balance:\n{y_train.value_counts(normalize=True).round(3)}")
print(f"y_test  balance:\n{y_test.value_counts(normalize=True).round(3)}")

# ---------------------------------------------------------------------------
# 7. StandardScaler — obligatorio para SVM
scaler = StandardScaler()
X_train_sc = pd.DataFrame(
    scaler.fit_transform(X_train), columns=X.columns, index=X_train.index
)
X_test_sc = pd.DataFrame(
    scaler.transform(X_test), columns=X.columns, index=X_test.index
)

# ---------------------------------------------------------------------------
# 8. Exportar
X_train_sc.to_csv(os.path.join(DATA_PROC, 'X_train.csv'), index=False)
X_test_sc.to_csv( os.path.join(DATA_PROC, 'X_test.csv'),  index=False)
y_train.reset_index(drop=True).to_csv(os.path.join(DATA_PROC, 'y_train.csv'), index=False)
y_test.reset_index(drop=True).to_csv(  os.path.join(DATA_PROC, 'y_test.csv'),  index=False)

print(f"\n✅ Datos procesados guardados en: {DATA_PROC}")
print(f"  X_train.csv : {X_train_sc.shape}")
print(f"  X_test.csv  : {X_test_sc.shape}")
print(f"  y_train.csv : {y_train.shape}")
print(f"  y_test.csv  : {y_test.shape}")
