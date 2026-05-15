"""
comparacion_etiquetas.py — Tabla comparativa 3 modelos x 3 escenarios.

Carga baseline_metrics.csv, tsvm_metrics.csv y lapsvm_metrics.csv y genera
una tabla unificada con accuracy y F1 para los 3 modelos en los 3 escenarios.
Exporta resultados/comparacion_final.csv.
"""
import os
import pandas as pd

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES      = os.path.join(BASE_DIR, 'resultados')

print("Cargando metricas de los 3 modelos...")
dfs = []
for fname in ['baseline_metrics.csv', 'tsvm_metrics.csv', 'lapsvm_metrics.csv']:
    path = os.path.join(RES, fname)
    df   = pd.read_csv(path)
    dfs.append(df)
    print(f"  {fname}: {len(df)} filas")

comparacion = pd.concat(dfs, ignore_index=True)
comparacion['porcentaje_etiquetado'] = (comparacion['porcentaje_etiquetado'] * 100).astype(int)
comparacion = comparacion.sort_values(['porcentaje_etiquetado', 'algoritmo']).reset_index(drop=True)

out_path = os.path.join(RES, 'comparacion_final.csv')
comparacion.to_csv(out_path, index=False)

print(f"\n{'='*60}")
print("Tabla comparativa — Accuracy y F1 por modelo y escenario:")
print(f"{'='*60}")
print(comparacion[['algoritmo', 'porcentaje_etiquetado', 'n_etiquetados',
                    'accuracy', 'f1']].to_string(index=False))
print(f"\nExportado: {out_path}")
