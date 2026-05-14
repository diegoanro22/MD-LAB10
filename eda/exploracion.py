"""
EDA — Adult Census Income Dataset
Genera graficos en visualizaciones/graficos/eda/
"""
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import os

BASE_DIR    = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH   = os.path.join(BASE_DIR, 'data', 'raw', 'adult.csv')
GRAFICOS_DIR = os.path.join(BASE_DIR, 'visualizaciones', 'graficos', 'eda')
os.makedirs(GRAFICOS_DIR, exist_ok=True)

sns.set_theme(style='whitegrid', palette='muted')

# ---------------------------------------------------------------------------
print("Cargando dataset...")
df = pd.read_csv(DATA_PATH)
print(f"Shape: {df.shape}")

NUMERIC_COLS = ['age', 'fnlwgt', 'education_num', 'capital_gain',
                'capital_loss', 'hours_per_week']
CATEG_COLS   = ['workclass', 'education', 'marital_status', 'occupation',
                'relationship', 'race', 'sex', 'native_country']

# ---------------------------------------------------------------------------
print("\n--- Tipos de datos ---")
print(df.dtypes)

print("\n--- Estadisticas descriptivas ---")
print(df.describe(include='all'))

print("\n--- Valores nulos ---")
print(df.isnull().sum())

print("\n--- Balance de clases ---")
vc = df['income'].value_counts()
print(vc)
print(vc / len(df) * 100)

# ---------------------------------------------------------------------------
# 1. Balance de clases
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
print("\n[OK] balance_clases.png")

# ---------------------------------------------------------------------------
# 2. Distribuciones numericas
fig, axes = plt.subplots(2, 3, figsize=(16, 9))
axes = axes.flatten()
for i, col in enumerate(NUMERIC_COLS):
    axes[i].hist(df[col].dropna(), bins=35, color='#4C72B0',
                 edgecolor='white', alpha=0.85)
    axes[i].set_title(col)
    axes[i].set_xlabel(col)
    axes[i].set_ylabel('Frecuencia')
plt.suptitle('Distribuciones — Variables Numericas', fontsize=15, y=1.02)
plt.tight_layout()
plt.savefig(os.path.join(GRAFICOS_DIR, 'distribuciones_numericas.png'),
            dpi=300, bbox_inches='tight')
plt.close()
print("[OK] distribuciones_numericas.png")

# ---------------------------------------------------------------------------
# 3. Heatmap de correlacion
corr = df[NUMERIC_COLS].corr()
fig, ax = plt.subplots(figsize=(8, 6))
mask = np.triu(np.ones_like(corr, dtype=bool))
sns.heatmap(corr, annot=True, fmt='.2f', cmap='coolwarm', ax=ax,
            mask=mask, square=True, linewidths=0.5, vmin=-1, vmax=1)
ax.set_title('Heatmap de Correlacion — Variables Numericas', fontsize=14)
plt.tight_layout()
plt.savefig(os.path.join(GRAFICOS_DIR, 'heatmap_correlacion.png'), dpi=300)
plt.close()
print("[OK] heatmap_correlacion.png")

# ---------------------------------------------------------------------------
# 4. Boxplots por clase (separabilidad)
fig, axes = plt.subplots(2, 3, figsize=(16, 9))
axes = axes.flatten()
for i, col in enumerate(NUMERIC_COLS):
    data_pos = df.loc[df['income'] == '>50K', col].dropna()
    data_neg = df.loc[df['income'] == '<=50K', col].dropna()
    axes[i].boxplot([data_neg, data_pos], labels=['<=50K', '>50K'],
                    patch_artist=True,
                    boxprops=dict(facecolor='#4C72B0', alpha=0.6))
    axes[i].set_title(col)
    axes[i].set_xlabel('Income')
    axes[i].set_ylabel(col)
plt.suptitle('Boxplots por Clase — Separabilidad', fontsize=15, y=1.02)
plt.tight_layout()
plt.savefig(os.path.join(GRAFICOS_DIR, 'boxplots_por_clase.png'),
            dpi=300, bbox_inches='tight')
plt.close()
print("[OK] boxplots_por_clase.png")

# ---------------------------------------------------------------------------
# 5. Variables categoricas (top valores)
fig, axes = plt.subplots(2, 4, figsize=(22, 10))
axes = axes.flatten()
for i, col in enumerate(CATEG_COLS):
    top = df[col].value_counts().head(6)
    top.plot(kind='bar', ax=axes[i], color='#4C72B0', edgecolor='white')
    axes[i].set_title(col)
    axes[i].tick_params(axis='x', rotation=40)
plt.suptitle('Distribucion Variables Categoricas (Top 6)', fontsize=15, y=1.01)
plt.tight_layout()
plt.savefig(os.path.join(GRAFICOS_DIR, 'categoricas.png'),
            dpi=300, bbox_inches='tight')
plt.close()
print("[OK] categoricas.png")

# ---------------------------------------------------------------------------
# 6. Correlacion con target (numericas por clase)
fig, ax = plt.subplots(figsize=(9, 5))
df_corr = df[NUMERIC_COLS].copy()
df_corr['income_bin'] = (df['income'] == '>50K').astype(int)
corr_target = df_corr.corr()['income_bin'].drop('income_bin').sort_values()
corr_target.plot(kind='barh', ax=ax, color=['#DD8452' if x < 0 else '#4C72B0'
                                             for x in corr_target])
ax.set_title('Correlacion de Variables Numericas con Target', fontsize=14)
ax.set_xlabel('Correlacion de Pearson')
ax.axvline(0, color='black', linewidth=0.8)
plt.tight_layout()
plt.savefig(os.path.join(GRAFICOS_DIR, 'correlacion_target.png'), dpi=300)
plt.close()
print("[OK] correlacion_target.png")

print(f"\n✅ EDA completo. Graficos en: {GRAFICOS_DIR}")
