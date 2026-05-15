import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
from sklearn.svm import SVC

# Agregar directorio raiz para importar modelos
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from modelos.utils import simular_etiquetas
from modelos.semisupervisados.tsvm import TSVM
from modelos.semisupervisados.lapsvm import LapSVM

def generar_graficos_modelos():
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    GRAFICOS_DIR = os.path.join(BASE_DIR, 'visualizaciones', 'graficos', 'modelos')
    os.makedirs(GRAFICOS_DIR, exist_ok=True)
    
    # 1. Curvas de desempeno: accuracy y F1 vs % de etiquetas
    print("Generando curvas de desempeno...")
    df = pd.read_csv(os.path.join(BASE_DIR, 'resultados', 'comparacion_final.csv'))
    
    plt.figure(figsize=(10, 6))
    sns.lineplot(data=df, x='porcentaje_etiquetado', y='accuracy', hue='algoritmo', marker='o')
    plt.title('Accuracy vs Porcentaje de Etiquetas')
    plt.xlabel('Porcentaje Etiquetado (%)')
    plt.ylabel('Accuracy')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(GRAFICOS_DIR, 'curva_accuracy.png'), dpi=300)
    plt.close()
    
    plt.figure(figsize=(10, 6))
    sns.lineplot(data=df, x='porcentaje_etiquetado', y='f1', hue='algoritmo', marker='o')
    plt.title('F1 Score vs Porcentaje de Etiquetas')
    plt.xlabel('Porcentaje Etiquetado (%)')
    plt.ylabel('F1 Score')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(GRAFICOS_DIR, 'curva_f1.png'), dpi=300)
    plt.close()

    # 2. Matrices de confusion y 3. Grafico prediccion vs valor real
    print("Cargando datos para generar predicciones (matrices de confusion y pred vs real)...")
    X_train = pd.read_csv(os.path.join(BASE_DIR, 'data', 'processed', 'X_train.csv')).values
    X_test  = pd.read_csv(os.path.join(BASE_DIR, 'data', 'processed', 'X_test.csv')).values
    y_train = pd.read_csv(os.path.join(BASE_DIR, 'data', 'processed', 'y_train.csv')).values.ravel()
    y_test  = pd.read_csv(os.path.join(BASE_DIR, 'data', 'processed', 'y_test.csv')).values.ravel()
    
    porcentajes = [0.05, 0.10, 0.20]
    
    fig_cm, axes_cm = plt.subplots(3, 3, figsize=(18, 18))
    
    for j, pct in enumerate(porcentajes):
        print(f"\nEntrenando modelos para escenario {int(pct*100)}%...")
        y_sim = simular_etiquetas(X_train, y_train, pct, random_state=42)
        labeled_mask = y_sim != -1
        X_labeled = X_train[labeled_mask]
        y_labeled = y_sim[labeled_mask]
        X_unlabeled = X_train[~labeled_mask]
        
        # Baseline
        print("  -> Baseline...")
        clf_base = SVC(kernel='rbf', C=1.0, random_state=42)
        clf_base.fit(X_labeled, y_labeled)
        y_pred_base = clf_base.predict(X_test)
        
        cm_base = confusion_matrix(y_test, y_pred_base)
        sns.heatmap(cm_base, annot=True, fmt='d', cmap='Blues', ax=axes_cm[0, j])
        axes_cm[0, j].set_title(f'Baseline {int(pct*100)}%')
        axes_cm[0, j].set_xlabel('Predicted')
        axes_cm[0, j].set_ylabel('True')
        
        # TSVM
        print("  -> TSVM...")
        tsvm = TSVM(kernel='rbf', C_L=1.0, C_U_init=0.01, C_U_max=1.0, max_iter=15, random_state=42)
        tsvm.fit(X_labeled, y_labeled, X_unlabeled)
        y_pred_tsvm = tsvm.predict(X_test)
        
        cm_tsvm = confusion_matrix(y_test, y_pred_tsvm)
        sns.heatmap(cm_tsvm, annot=True, fmt='d', cmap='Greens', ax=axes_cm[1, j])
        axes_cm[1, j].set_title(f'TSVM {int(pct*100)}%')
        axes_cm[1, j].set_xlabel('Predicted')
        axes_cm[1, j].set_ylabel('True')
        
        # LapSVM
        print("  -> LapSVM...")
        lapsvm = LapSVM(C=1.0, gamma_I=0.1, n_neighbors=7, max_unlabeled=3000, random_state=42)
        lapsvm.fit(X_labeled, y_labeled, X_unlabeled)
        y_pred_lap = lapsvm.predict(X_test)
        
        cm_lap = confusion_matrix(y_test, y_pred_lap)
        sns.heatmap(cm_lap, annot=True, fmt='d', cmap='Oranges', ax=axes_cm[2, j])
        axes_cm[2, j].set_title(f'LapSVM {int(pct*100)}%')
        axes_cm[2, j].set_xlabel('Predicted')
        axes_cm[2, j].set_ylabel('True')

        # Prediccion vs Valor Real
        df_pred = pd.DataFrame({
            'Real': y_test,
            'Baseline': y_pred_base,
            'TSVM': y_pred_tsvm,
            'LapSVM': y_pred_lap
        })
        
        df_melt = df_pred.melt(var_name='Modelo', value_name='Clase')
        plt.figure(figsize=(10, 6))
        sns.countplot(data=df_melt, x='Clase', hue='Modelo')
        plt.title(f'Distribucion de Clases Predichas vs Real (Escenario {int(pct*100)}%)')
        plt.tight_layout()
        plt.savefig(os.path.join(GRAFICOS_DIR, f'pred_vs_real_{int(pct*100)}pct.png'), dpi=300)
        plt.close()
        
    plt.tight_layout()
    fig_cm.savefig(os.path.join(GRAFICOS_DIR, 'matrices_confusion_3x3.png'), dpi=300)
    plt.close(fig_cm)
    
    print(f"Graficos de modelos generados exitosamente en {GRAFICOS_DIR}")

if __name__ == "__main__":
    generar_graficos_modelos()
