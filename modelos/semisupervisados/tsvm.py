"""
tsvm.py — Transductive SVM (TSVM) implementado manualmente.

Funcion objetivo (Joachims, 1999):
    min 1/2 ||w||^2 + C_L * sum(xi_i) + C_U * sum(xi_j*)
    i = datos etiquetados, j = datos no etiquetados

Mecanismo: SVC de sklearn con sample_weight simula C_L y C_U por muestra.
C_U sube de forma exponencial (schedule de Joachims) hasta C_U_max.
Exporta resultados/tsvm_metrics.csv con escenarios 5%, 10%, 20%.
"""
import os
import sys
import numpy as np
import pandas as pd
from sklearn.svm import SVC

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils import simular_etiquetas, evaluar_modelo, exportar_resultados

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
PROC     = os.path.join(BASE_DIR, 'data', 'processed')
RES      = os.path.join(BASE_DIR, 'resultados')
os.makedirs(RES, exist_ok=True)

RANDOM_STATE = 42
PORCENTAJES  = [0.05, 0.10, 0.20]


class TSVM:
    """
    Transductive SVM: maximiza el margen incluyendo datos no etiquetados.

    Funcion objetivo implementada:
        min 1/2 ||w||^2 + C_L * sum(xi_i) + C_U * sum(xi_j*)

    sample_weight de sklearn escala C por muestra, mapeando directamente
    a C_L (etiquetados) y C_U (no etiquetados) del paper original.
    C_U sube de forma exponencial en cada iteracion (schedule de Joachims).
    """

    def __init__(self, kernel='rbf', C_L=1.0, C_U_init=0.01, C_U_max=1.0,
                 max_iter=15, random_state=42):
        self.kernel       = kernel
        self.C_L          = C_L
        self.C_U_init     = C_U_init
        self.C_U_max      = C_U_max
        self.max_iter     = max_iter
        self.random_state = random_state
        self.model_       = None

    def fit(self, X_labeled, y_labeled, X_unlabeled):
        n_lab   = len(y_labeled)
        n_unlab = len(X_unlabeled)
        X_all   = np.vstack([X_labeled, X_unlabeled])

        # Paso 1: SVM inicial entrenado solo con datos etiquetados
        svm0 = SVC(kernel=self.kernel, C=self.C_L, random_state=self.random_state)
        svm0.fit(X_labeled, y_labeled)

        # Paso 2: pseudo-etiquetas iniciales para los no etiquetados
        y_pseudo = svm0.predict(X_unlabeled)

        # Paso 3: refinamiento iterativo con C_U creciente
        C_U = self.C_U_init
        clf = svm0

        for it in range(self.max_iter):
            y_all   = np.concatenate([y_labeled, y_pseudo])
            weights = np.concatenate([
                np.ones(n_lab)   * self.C_L,
                np.ones(n_unlab) * C_U,
            ])

            clf = SVC(kernel=self.kernel, C=1.0, random_state=self.random_state)
            clf.fit(X_all, y_all, sample_weight=weights)

            y_new     = clf.predict(X_unlabeled)
            n_changed = int((y_new != y_pseudo).sum())
            y_pseudo  = y_new

            print(f"    iter {it+1:02d} | C_U={C_U:.4f} | etiquetas cambiadas: {n_changed}")

            C_U = min(C_U * 2.0, self.C_U_max)
            if n_changed == 0 and C_U >= self.C_U_max:
                print("    Convergencia alcanzada.")
                break

        self.model_ = clf
        return self

    def predict(self, X):
        return self.model_.predict(X)


# ---------------------------------------------------------------------------
if __name__ == '__main__':
    print("Cargando datos procesados...")
    X_train = pd.read_csv(os.path.join(PROC, 'X_train.csv')).values
    X_test  = pd.read_csv(os.path.join(PROC, 'X_test.csv')).values
    y_train = pd.read_csv(os.path.join(PROC, 'y_train.csv')).values.ravel()
    y_test  = pd.read_csv(os.path.join(PROC, 'y_test.csv')).values.ravel()
    print(f"  X_train: {X_train.shape} | X_test: {X_test.shape}")

    resultados = []

    for pct in PORCENTAJES:
        print(f"\n{'='*50}")
        print(f"Escenario: {int(pct*100)}% etiquetado (TSVM)")

        y_sim        = simular_etiquetas(X_train, y_train, pct, random_state=RANDOM_STATE)
        labeled_mask = y_sim != -1
        X_labeled    = X_train[labeled_mask]
        y_labeled    = y_sim[labeled_mask]
        X_unlabeled  = X_train[~labeled_mask]

        print(f"  Etiquetados: {labeled_mask.sum()} | No etiquetados: {(~labeled_mask).sum()}")

        tsvm = TSVM(kernel='rbf', C_L=1.0, C_U_init=0.01, C_U_max=1.0,
                    max_iter=15, random_state=RANDOM_STATE)
        tsvm.fit(X_labeled, y_labeled, X_unlabeled)
        y_pred = tsvm.predict(X_test)

        metricas = evaluar_modelo(y_test, y_pred)
        metricas['porcentaje_etiquetado'] = pct
        metricas['n_etiquetados']         = int(labeled_mask.sum())
        metricas['algoritmo']             = 'TSVM'
        resultados.append(metricas)

        print(f"  Accuracy:  {metricas['accuracy']:.4f}")
        print(f"  F1:        {metricas['f1']:.4f}")
        print(f"  Precision: {metricas['precision']:.4f}")
        print(f"  Recall:    {metricas['recall']:.4f}")

    out_path = os.path.join(RES, 'tsvm_metrics.csv')
    exportar_resultados(resultados, out_path)
    print(f"\n{'='*50}")
    print("Resumen TSVM:")
    df_res = pd.DataFrame(resultados)
    print(df_res[['algoritmo', 'porcentaje_etiquetado', 'n_etiquetados',
                  'accuracy', 'f1']].to_string(index=False))
