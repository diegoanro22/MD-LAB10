"""
utils.py — Funciones compartidas para modelos supervisado y semi-supervisado.
"""
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score


def simular_etiquetas(X_train, y_train, porcentaje, random_state=42):
    """
    Devuelve y_sim donde solo `porcentaje` de muestras conservan su etiqueta
    real; el resto se marca con -1 (no etiquetado). Muestreo estratificado.
    """
    y_arr = np.asarray(y_train)
    n_labeled = max(2, int(len(y_arr) * porcentaje))

    sss = StratifiedShuffleSplit(n_splits=1, train_size=n_labeled, random_state=random_state)
    labeled_idx, _ = next(sss.split(X_train, y_arr))

    y_sim = np.full(len(y_arr), -1, dtype=int)
    y_sim[labeled_idx] = y_arr[labeled_idx]
    return y_sim


def evaluar_modelo(y_true, y_pred):
    """Retorna dict con accuracy, F1 (weighted), precision y recall."""
    return {
        'accuracy':  accuracy_score(y_true, y_pred),
        'f1':        f1_score(y_true, y_pred, average='weighted', zero_division=0),
        'precision': precision_score(y_true, y_pred, average='weighted', zero_division=0),
        'recall':    recall_score(y_true, y_pred, average='weighted', zero_division=0),
    }


def exportar_resultados(resultados, filepath):
    """Guarda lista de dicts de métricas como CSV en filepath."""
    pd.DataFrame(resultados).to_csv(filepath, index=False)
    print(f"Resultados exportados: {filepath}")
