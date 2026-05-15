"""
sensibilidad.py — Analisis de sensibilidad de hiperparametros para TSVM y LapSVM.

Grid search sobre:
  TSVM  : C_L [0.1, 1, 10] x C_U_init [0.001, 0.01, 0.1] x kernel ['linear', 'rbf']
  LapSVM: gamma_I [0.01, 0.1, 1.0] x n_neighbors [5, 7, 10] x C [0.1, 1, 10]

Para cada combinacion se registra accuracy y F1 en train y test.
El grid usa escenario 10% de etiquetas sobre una submuestra estratificada de 4000
muestras para mantener tiempos tractables; la tendencia de sensibilidad es la misma.
Exporta resultados/sensibilidad_grid.csv.
"""
import os
import sys
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROC     = os.path.join(BASE_DIR, 'data', 'processed')
RES      = os.path.join(BASE_DIR, 'resultados')
os.makedirs(RES, exist_ok=True)

sys.path.insert(0, os.path.join(BASE_DIR, 'modelos'))
sys.path.insert(0, os.path.join(BASE_DIR, 'modelos', 'semisupervisados'))

from utils  import simular_etiquetas, evaluar_modelo
from tsvm   import TSVM
from lapsvm import LapSVM

RANDOM_STATE   = 42
PCT_ETIQUETADO = 0.10   # escenario fijo para el grid
N_SAMPLE       = 4000   # submuestra estratificada para hacer el grid tratable

# Grids de hiperparametros segun el plan
TSVM_CL       = [0.1, 1, 10]
TSVM_CU       = [0.001, 0.01, 0.1]
TSVM_KERNELS  = ['linear', 'rbf']

LAPSVM_GAMMA  = [0.01, 0.1, 1.0]
LAPSVM_KNN    = [5, 7, 10]
LAPSVM_C      = [0.1, 1, 10]


def _subsample(X, y, n, random_state=42):
    """Submuestra estratificada de n filas."""
    sss = StratifiedShuffleSplit(n_splits=1, train_size=n, random_state=random_state)
    idx, _ = next(sss.split(X, y))
    return X[idx], y[idx]


def run_tsvm_grid(X_labeled, y_labeled, X_unlabeled, X_test, y_test):
    resultados = []
    combos = [(cl, cu, k) for cl in TSVM_CL for cu in TSVM_CU for k in TSVM_KERNELS]
    total = len(combos)
    for i, (C_L, C_U_init, kernel) in enumerate(combos, 1):
        print(f"  TSVM [{i:02d}/{total}]: C_L={C_L}, C_U_init={C_U_init}, kernel={kernel}")
        tsvm = TSVM(kernel=kernel, C_L=C_L, C_U_init=C_U_init,
                    C_U_max=C_L, max_iter=8, random_state=RANDOM_STATE)
        try:
            tsvm.fit(X_labeled, y_labeled, X_unlabeled)
            y_pred_test  = tsvm.predict(X_test)
            y_pred_train = tsvm.predict(X_labeled)
            met_test  = evaluar_modelo(y_test,    y_pred_test)
            met_train = evaluar_modelo(y_labeled, y_pred_train)
            print(f"    acc_test={met_test['accuracy']:.4f} | acc_train={met_train['accuracy']:.4f}")
            resultados.append({
                'modelo':         'TSVM',
                'C_L':            C_L,
                'C_U_init':       C_U_init,
                'kernel':         kernel,
                'gamma_I':        np.nan,
                'n_neighbors':    np.nan,
                'C':              np.nan,
                'accuracy_test':  met_test['accuracy'],
                'f1_test':        met_test['f1'],
                'accuracy_train': met_train['accuracy'],
                'f1_train':       met_train['f1'],
            })
        except Exception as e:
            print(f"    ERROR: {e}")
    return resultados


def run_lapsvm_grid(X_labeled, y_labeled, X_unlabeled, X_test, y_test):
    resultados = []
    combos = [(g, k, c) for g in LAPSVM_GAMMA for k in LAPSVM_KNN for c in LAPSVM_C]
    total = len(combos)
    for i, (gamma_I, n_neighbors, C) in enumerate(combos, 1):
        print(f"  LapSVM [{i:02d}/{total}]: gamma_I={gamma_I}, n_neighbors={n_neighbors}, C={C}")
        lapsvm = LapSVM(C=C, gamma_I=gamma_I, n_neighbors=n_neighbors,
                        max_unlabeled=2000, random_state=RANDOM_STATE)
        try:
            lapsvm.fit(X_labeled, y_labeled, X_unlabeled)
            y_pred_test  = lapsvm.predict(X_test)
            y_pred_train = lapsvm.predict(X_labeled)
            met_test  = evaluar_modelo(y_test,    y_pred_test)
            met_train = evaluar_modelo(y_labeled, y_pred_train)
            print(f"    acc_test={met_test['accuracy']:.4f} | acc_train={met_train['accuracy']:.4f}")
            resultados.append({
                'modelo':         'LapSVM',
                'C_L':            np.nan,
                'C_U_init':       np.nan,
                'kernel':         np.nan,
                'gamma_I':        gamma_I,
                'n_neighbors':    n_neighbors,
                'C':              C,
                'accuracy_test':  met_test['accuracy'],
                'f1_test':        met_test['f1'],
                'accuracy_train': met_train['accuracy'],
                'f1_train':       met_train['f1'],
            })
        except Exception as e:
            print(f"    ERROR: {e}")
    return resultados


if __name__ == '__main__':
    print("Cargando datos procesados...")
    X_train = pd.read_csv(os.path.join(PROC, 'X_train.csv')).values
    X_test  = pd.read_csv(os.path.join(PROC, 'X_test.csv')).values
    y_train = pd.read_csv(os.path.join(PROC, 'y_train.csv')).values.ravel()
    y_test  = pd.read_csv(os.path.join(PROC, 'y_test.csv')).values.ravel()
    print(f"  X_train completo: {X_train.shape} | X_test: {X_test.shape}")

    print(f"\nSubmuestreando a {N_SAMPLE} filas (estratificado)...")
    X_sub, y_sub = _subsample(X_train, y_train, N_SAMPLE, RANDOM_STATE)
    print(f"  Submuestra: {X_sub.shape}")

    y_sim        = simular_etiquetas(X_sub, y_sub, PCT_ETIQUETADO, random_state=RANDOM_STATE)
    labeled_mask = y_sim != -1
    X_labeled    = X_sub[labeled_mask]
    y_labeled    = y_sim[labeled_mask]
    X_unlabeled  = X_sub[~labeled_mask]
    print(f"  Etiquetados: {labeled_mask.sum()} | No etiquetados: {(~labeled_mask).sum()}")

    resultados = []

    print(f"\n{'='*60}")
    print("Grid search TSVM")
    print(f"{'='*60}")
    resultados += run_tsvm_grid(X_labeled, y_labeled, X_unlabeled, X_test, y_test)

    print(f"\n{'='*60}")
    print("Grid search LapSVM")
    print(f"{'='*60}")
    resultados += run_lapsvm_grid(X_labeled, y_labeled, X_unlabeled, X_test, y_test)

    out_path = os.path.join(RES, 'sensibilidad_grid.csv')
    pd.DataFrame(resultados).to_csv(out_path, index=False)
    print(f"\nSensibilidad exportada: {out_path}")
    print(f"Total combinaciones: {len(resultados)}")

    df = pd.DataFrame(resultados)
    print("\nResumen TSVM (acc_test promedio por C_L):")
    tsvm = df[df['modelo'] == 'TSVM']
    print(tsvm.groupby('C_L')[['accuracy_test', 'accuracy_train']].mean().round(4).to_string())
    print("\nResumen LapSVM (acc_test promedio por gamma_I):")
    lapsvm = df[df['modelo'] == 'LapSVM']
    print(lapsvm.groupby('gamma_I')[['accuracy_test', 'accuracy_train']].mean().round(4).to_string())
