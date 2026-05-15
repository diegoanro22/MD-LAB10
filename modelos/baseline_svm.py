"""
baseline_svm.py — SVC supervisado (baseline) entrenado SOLO con datos etiquetados.

Entrenamos SVC(kernel='rbf') unicamente con los datos que tienen etiqueta real.
Los datos no etiquetados se ignoran completamente — ese es el punto de comparacion
con TSVM y LapSVM que si los aprovechan.

Escenarios: 5%, 10%, 20% de etiquetas.
Exporta resultados/baseline_metrics.csv.
"""
import os
import sys
import numpy as np
import pandas as pd
from sklearn.svm import SVC

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from utils import simular_etiquetas, evaluar_modelo, exportar_resultados

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROC     = os.path.join(BASE_DIR, 'data', 'processed')
RES      = os.path.join(BASE_DIR, 'resultados')
os.makedirs(RES, exist_ok=True)

RANDOM_STATE = 42
PORCENTAJES  = [0.05, 0.10, 0.20]

print("Cargando datos procesados...")
X_train = pd.read_csv(os.path.join(PROC, 'X_train.csv')).values
X_test  = pd.read_csv(os.path.join(PROC, 'X_test.csv')).values
y_train = pd.read_csv(os.path.join(PROC, 'y_train.csv')).values.ravel()
y_test  = pd.read_csv(os.path.join(PROC, 'y_test.csv')).values.ravel()
print(f"  X_train: {X_train.shape} | X_test: {X_test.shape}")

resultados = []

for pct in PORCENTAJES:
    print(f"\n{'='*50}")
    print(f"Escenario: {int(pct*100)}% etiquetado (Baseline SVC)")

    y_sim        = simular_etiquetas(X_train, y_train, pct, random_state=RANDOM_STATE)
    labeled_mask = y_sim != -1
    X_labeled    = X_train[labeled_mask]
    y_labeled    = y_sim[labeled_mask]

    print(f"  Etiquetados: {labeled_mask.sum()} | No etiquetados (ignorados): {(~labeled_mask).sum()}")

    clf = SVC(kernel='rbf', C=1.0, random_state=RANDOM_STATE)
    clf.fit(X_labeled, y_labeled)
    y_pred = clf.predict(X_test)

    metricas = evaluar_modelo(y_test, y_pred)
    metricas['porcentaje_etiquetado'] = pct
    metricas['n_etiquetados']         = int(labeled_mask.sum())
    metricas['algoritmo']             = 'Baseline_SVC'
    resultados.append(metricas)

    print(f"  Accuracy:  {metricas['accuracy']:.4f}")
    print(f"  F1:        {metricas['f1']:.4f}")
    print(f"  Precision: {metricas['precision']:.4f}")
    print(f"  Recall:    {metricas['recall']:.4f}")

out_path = os.path.join(RES, 'baseline_metrics.csv')
exportar_resultados(resultados, out_path)
print(f"\n{'='*50}")
print("Resumen Baseline SVC:")
df_res = pd.DataFrame(resultados)
print(df_res[['algoritmo', 'porcentaje_etiquetado', 'n_etiquetados',
              'accuracy', 'f1']].to_string(index=False))
