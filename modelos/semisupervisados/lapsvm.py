"""
lapsvm.py — Laplacian SVM (LapSVM) implementado manualmente.

Funcion objetivo (Belkin et al., 2006):
    min 1/nL * sum V(yi, f(xi)) + gamma_A * ||f||^2_H + gamma_I * f^T L f
    L = Laplaciano del grafo k-NN construido sobre etiquetados + no etiquetados

Mecanismo: kernel RBF modificado con regularizacion Laplaciana (precomputed kernel).
El grafo k-NN captura la geometria/manifold de los datos no etiquetados.
Para datasets grandes se usa un subconjunto de no etiquetados (max_unlabeled).
Exporta resultados/lapsvm_metrics.csv con escenarios 5%, 10%, 20%.
"""
import os
import sys
import numpy as np
import pandas as pd
from sklearn.svm import SVC
from sklearn.neighbors import kneighbors_graph
from sklearn.metrics.pairwise import rbf_kernel
import scipy.sparse as sp

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils import simular_etiquetas, evaluar_modelo, exportar_resultados

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
PROC     = os.path.join(BASE_DIR, 'data', 'processed')
RES      = os.path.join(BASE_DIR, 'resultados')
os.makedirs(RES, exist_ok=True)

RANDOM_STATE = 42
PORCENTAJES  = [0.05, 0.10, 0.20]


class LapSVM:
    """
    Laplacian SVM: regularizacion via grafo Laplaciano sobre todos los datos.

    Funcion objetivo:
        min 1/nL * sum V(yi,f(xi)) + gamma_A*||f||^2_H + gamma_I * f^T L f

    El termino gamma_I * f^T L f obliga a que la funcion f sea suave sobre
    el grafo: puntos cercanos en la geometria del dato deben tener predicciones
    similares (smoothness assumption).

    Implementacion: kernel RBF modificado K_lap = K + gamma_I * L @ K,
    entrenado con SVC(kernel='precomputed'). El Laplaciano L se construye
    sobre etiquetados + no etiquetados (semi-supervisado).
    Para datasets grandes se submuestrea max_unlabeled puntos no etiquetados.
    """

    def __init__(self, C=1.0, gamma_I=0.1, n_neighbors=7,
                 max_unlabeled=3000, random_state=42):
        self.C             = C
        self.gamma_I       = gamma_I
        self.n_neighbors   = n_neighbors
        self.max_unlabeled = max_unlabeled
        self.random_state  = random_state

    def _laplaciano(self, X):
        W = kneighbors_graph(X, self.n_neighbors, mode='connectivity',
                             include_self=False)
        W = (W + W.T) / 2           # simetrizar
        D = sp.diags(np.array(W.sum(axis=1)).ravel())
        return (D - W).toarray()    # Laplaciano no normalizado

    def fit(self, X_labeled, y_labeled, X_unlabeled):
        rng = np.random.RandomState(self.random_state)

        # Submuestrear no etiquetados si exceden max_unlabeled
        if len(X_unlabeled) > self.max_unlabeled:
            idx        = rng.choice(len(X_unlabeled), self.max_unlabeled, replace=False)
            X_unlabeled = X_unlabeled[idx]

        n_lab  = len(X_labeled)
        X_all  = np.vstack([X_labeled, X_unlabeled])

        # Grafo k-NN sobre todos los datos -> Laplaciano L
        L = self._laplaciano(X_all)

        # Kernel RBF K (n x n)
        K = rbf_kernel(X_all)

        # Kernel modificado: K_lap = K + gamma_I * L @ K
        # Implementa el termino de regularizacion Laplaciana de la funcion objetivo
        K_lap = K + self.gamma_I * (L @ K)

        # Bloque etiquetado para entrenar SVC
        K_train = K_lap[:n_lab, :n_lab]

        self.svc_ = SVC(kernel='precomputed', C=self.C, random_state=self.random_state)
        self.svc_.fit(K_train, y_labeled)

        # Guardar para prediccion
        self.X_all_  = X_all
        self.L_      = L
        self.n_lab_  = n_lab
        return self

    def predict(self, X_test):
        # Kernel entre test y todos los datos de entrenamiento
        K_test = rbf_kernel(X_test, self.X_all_)
        # Aplicar suavizado Laplaciano: K_test_lap ~= K_test + gamma_I * K_test @ L
        K_test_lap = K_test + self.gamma_I * (K_test @ self.L_)
        # SVC necesita (n_test, n_labeled)
        return self.svc_.predict(K_test_lap[:, :self.n_lab_])


# ---------------------------------------------------------------------------
if __name__ == '__main__' or __name__ != '__main__':
    print("Cargando datos procesados...")
    X_train = pd.read_csv(os.path.join(PROC, 'X_train.csv')).values
    X_test  = pd.read_csv(os.path.join(PROC, 'X_test.csv')).values
    y_train = pd.read_csv(os.path.join(PROC, 'y_train.csv')).values.ravel()
    y_test  = pd.read_csv(os.path.join(PROC, 'y_test.csv')).values.ravel()
    print(f"  X_train: {X_train.shape} | X_test: {X_test.shape}")

    resultados = []

    for pct in PORCENTAJES:
        print(f"\n{'='*50}")
        print(f"Escenario: {int(pct*100)}% etiquetado (LapSVM)")

        y_sim        = simular_etiquetas(X_train, y_train, pct, random_state=RANDOM_STATE)
        labeled_mask = y_sim != -1
        X_labeled    = X_train[labeled_mask]
        y_labeled    = y_sim[labeled_mask]
        X_unlabeled  = X_train[~labeled_mask]

        print(f"  Etiquetados: {labeled_mask.sum()} | No etiquetados usados: min({(~labeled_mask).sum()}, 3000)")

        lapsvm = LapSVM(C=1.0, gamma_I=0.1, n_neighbors=7,
                        max_unlabeled=3000, random_state=RANDOM_STATE)
        lapsvm.fit(X_labeled, y_labeled, X_unlabeled)
        y_pred = lapsvm.predict(X_test)

        metricas = evaluar_modelo(y_test, y_pred)
        metricas['porcentaje_etiquetado'] = pct
        metricas['n_etiquetados']         = int(labeled_mask.sum())
        metricas['algoritmo']             = 'LapSVM'
        resultados.append(metricas)

        print(f"  Accuracy:  {metricas['accuracy']:.4f}")
        print(f"  F1:        {metricas['f1']:.4f}")
        print(f"  Precision: {metricas['precision']:.4f}")
        print(f"  Recall:    {metricas['recall']:.4f}")

    out_path = os.path.join(RES, 'lapsvm_metrics.csv')
    exportar_resultados(resultados, out_path)
    print(f"\n{'='*50}")
    print("Resumen LapSVM:")
    df_res = pd.DataFrame(resultados)
    print(df_res[['algoritmo', 'porcentaje_etiquetado', 'n_etiquetados',
                  'accuracy', 'f1']].to_string(index=False))
