import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

def generar_graficos_eda():
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    DATA_PATH = os.path.join(BASE_DIR, 'data', 'raw', 'adult.csv')
    GRAFICOS_DIR = os.path.join(BASE_DIR, 'visualizaciones', 'graficos', 'eda')
    os.makedirs(GRAFICOS_DIR, exist_ok=True)

    sns.set_theme(style='whitegrid', palette='muted')

    print("Cargando dataset para EDA...")
    df = pd.read_csv(DATA_PATH)

    NUMERIC_COLS = ['age', 'fnlwgt', 'education_num', 'capital_gain', 'capital_loss', 'hours_per_week']

    # 1. Balance de clases
    print("Generando grafico de balance de clases...")
    vc = df['income'].value_counts()
    fig, ax = plt.subplots(figsize=(6, 4))
    colors = ['#4C72B0', '#DD8452']
    vc.plot(kind='bar', ax=ax, color=colors, edgecolor='white', width=0.6)
    ax.set_title('Balance de Clases — Income', fontsize=14)
    ax.set_xlabel('Clase')
    ax.set_ylabel('Frecuencia')
    ax.tick_params(axis='x', rotation=0)
    for bar, val in zip(ax.patches, vc):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 100,
                f'{val:,}', ha='center', va='bottom', fontsize=11)
    plt.tight_layout()
    plt.savefig(os.path.join(GRAFICOS_DIR, 'balance_clases.png'), dpi=300)
    plt.close()

    # 2. Distribuciones numericas
    print("Generando histogramas...")
    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    axes = axes.flatten()
    for i, col in enumerate(NUMERIC_COLS):
        axes[i].hist(df[col].dropna(), bins=35, color='#4C72B0', edgecolor='white', alpha=0.85)
        axes[i].set_title(col)
        axes[i].set_xlabel(col)
        axes[i].set_ylabel('Frecuencia')
    plt.suptitle('Distribuciones — Variables Numericas', fontsize=15, y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(GRAFICOS_DIR, 'histogramas.png'), dpi=300, bbox_inches='tight')
    plt.close()

    # 3. Heatmap de correlacion
    print("Generando mapa de calor...")
    corr = df[NUMERIC_COLS].corr()
    fig, ax = plt.subplots(figsize=(8, 6))
    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(corr, annot=True, fmt='.2f', cmap='coolwarm', ax=ax,
                mask=mask, square=True, linewidths=0.5, vmin=-1, vmax=1)
    ax.set_title('Heatmap de Correlacion — Variables Numericas', fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(GRAFICOS_DIR, 'heatmap_correlacion.png'), dpi=300)
    plt.close()

    # 4. Boxplots por clase
    print("Generando boxplots...")
    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    axes = axes.flatten()
    for i, col in enumerate(NUMERIC_COLS):
        data_pos = df.loc[df['income'] == '>50K', col].dropna()
        data_neg = df.loc[df['income'] == '<=50K', col].dropna()
        axes[i].boxplot([data_neg, data_pos], labels=['<=50K', '>50K'],
                        patch_artist=True, boxprops=dict(facecolor='#4C72B0', alpha=0.6))
        axes[i].set_title(col)
        axes[i].set_xlabel('Income')
        axes[i].set_ylabel(col)
    plt.suptitle('Boxplots por Clase — Separabilidad', fontsize=15, y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(GRAFICOS_DIR, 'boxplots.png'), dpi=300, bbox_inches='tight')
    plt.close()

    print("Graficos EDA generados exitosamente en visualizaciones/graficos/eda/")

if __name__ == "__main__":
    generar_graficos_eda()
