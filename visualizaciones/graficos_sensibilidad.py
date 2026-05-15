"""
graficos_sensibilidad.py — Visualizaciones del analisis de sensibilidad de hiperparametros.

Genera:
  1. Heatmap C_L vs C_U_init (TSVM) y gamma_I vs n_neighbors (LapSVM)
  2. Curvas accuracy train vs test por C_L (TSVM) y gamma_I (LapSVM)

"""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

BASE_DIR   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES        = os.path.join(BASE_DIR, 'resultados')
OUT_DIR    = os.path.join(BASE_DIR, 'visualizaciones', 'graficos', 'sensibilidad')
os.makedirs(OUT_DIR, exist_ok=True)

CSV_PATH = os.path.join(RES, 'sensibilidad_grid.csv')

sns.set_theme(style='whitegrid', font_scale=1.1)


def load_data():
    df = pd.read_csv(CSV_PATH)
    tsvm   = df[df['modelo'] == 'TSVM'].copy()
    lapsvm = df[df['modelo'] == 'LapSVM'].copy()
    return tsvm, lapsvm


def plot_heatmaps(tsvm, lapsvm):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle('Sensibilidad de Hiperparametros — Accuracy en Test', fontsize=14, fontweight='bold')

    # --- TSVM: C_L vs C_U_init (promediado sobre kernel) ---
    pivot_tsvm = (
        tsvm.groupby(['C_L', 'C_U_init'])['accuracy_test']
        .mean()
        .unstack('C_U_init')
    )
    sns.heatmap(
        pivot_tsvm,
        ax=axes[0],
        annot=True, fmt='.3f',
        cmap='YlOrRd',
        linewidths=0.5,
        cbar_kws={'label': 'Accuracy Test'},
    )
    axes[0].set_title('TSVM: C_L vs C_U_init\n(promedio sobre kernel linear/rbf)')
    axes[0].set_xlabel('C_U_init')
    axes[0].set_ylabel('C_L')
    axes[0].invert_yaxis()

    # --- LapSVM: gamma_I vs n_neighbors (promediado sobre C) ---
    pivot_lapsvm = (
        lapsvm.groupby(['gamma_I', 'n_neighbors'])['accuracy_test']
        .mean()
        .unstack('n_neighbors')
    )
    sns.heatmap(
        pivot_lapsvm,
        ax=axes[1],
        annot=True, fmt='.3f',
        cmap='YlOrRd',
        linewidths=0.5,
        cbar_kws={'label': 'Accuracy Test'},
    )
    axes[1].set_title('LapSVM: gamma_I vs n_neighbors\n(promedio sobre C = 0.1/1/10)')
    axes[1].set_xlabel('n_neighbors')
    axes[1].set_ylabel('gamma_I')
    axes[1].invert_yaxis()

    plt.tight_layout()
    out = os.path.join(OUT_DIR, 'heatmaps.png')
    plt.savefig(out, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Guardado: {out}")


def plot_curvas_validacion(tsvm, lapsvm):
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle('Curvas de Validacion — Train vs Test', fontsize=14, fontweight='bold')

    # --- TSVM: accuracy vs C_L ---
    tsvm_agg = (
        tsvm.groupby('C_L')[['accuracy_train', 'accuracy_test']]
        .mean()
        .reset_index()
    )
    ax = axes[0]
    ax.plot(tsvm_agg['C_L'], tsvm_agg['accuracy_train'],
            marker='o', linewidth=2, label='Train')
    ax.plot(tsvm_agg['C_L'], tsvm_agg['accuracy_test'],
            marker='s', linewidth=2, linestyle='--', label='Test')
    ax.fill_between(
        tsvm_agg['C_L'],
        tsvm_agg['accuracy_train'],
        tsvm_agg['accuracy_test'],
        alpha=0.15, color='gray', label='Brecha (posible overfitting)',
    )
    ax.set_xscale('log')
    ax.set_xlabel('C_L (escala log)')
    ax.set_ylabel('Accuracy promedio')
    ax.set_title('TSVM — Sensibilidad a C_L\n(promedio sobre C_U_init y kernel)')
    ax.legend(fontsize=9)
    ax.set_ylim(0, 1.05)
    _xticks(ax, tsvm_agg['C_L'].tolist())

    # --- LapSVM: accuracy vs gamma_I ---
    lapsvm_agg = (
        lapsvm.groupby('gamma_I')[['accuracy_train', 'accuracy_test']]
        .mean()
        .reset_index()
    )
    ax = axes[1]
    ax.plot(lapsvm_agg['gamma_I'], lapsvm_agg['accuracy_train'],
            marker='o', linewidth=2, label='Train')
    ax.plot(lapsvm_agg['gamma_I'], lapsvm_agg['accuracy_test'],
            marker='s', linewidth=2, linestyle='--', label='Test')
    ax.fill_between(
        lapsvm_agg['gamma_I'],
        lapsvm_agg['accuracy_train'],
        lapsvm_agg['accuracy_test'],
        alpha=0.15, color='gray', label='Brecha (posible overfitting)',
    )
    ax.set_xscale('log')
    ax.set_xlabel('gamma_I (escala log)')
    ax.set_ylabel('Accuracy promedio')
    ax.set_title('LapSVM — Sensibilidad a gamma_I\n(promedio sobre n_neighbors y C)')
    ax.legend(fontsize=9)
    ax.set_ylim(0, 1.05)
    _xticks(ax, lapsvm_agg['gamma_I'].tolist())

    plt.tight_layout()
    out = os.path.join(OUT_DIR, 'curvas_validacion.png')
    plt.savefig(out, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Guardado: {out}")


def _xticks(ax, values):
    ax.set_xticks(values)
    ax.get_xaxis().set_major_formatter(plt.ScalarFormatter())
    ax.ticklabel_format(axis='x', style='plain')


if __name__ == '__main__':
    print("Cargando sensibilidad_grid.csv...")
    tsvm, lapsvm = load_data()
    print(f"  TSVM: {len(tsvm)} combinaciones | LapSVM: {len(lapsvm)} combinaciones")

    print("\nGenerando heatmaps...")
    plot_heatmaps(tsvm, lapsvm)

    print("Generando curvas de validacion...")
    plot_curvas_validacion(tsvm, lapsvm)

    print("\nTodos los graficos guardados en:", OUT_DIR)
