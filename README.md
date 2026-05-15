# Laboratorio 10 — Aprendizaje Semi-Supervisado
**CC3074 Minería de Datos — Grupo 04 | Universidad del Valle de Guatemala**

## Descripción

Este laboratorio implementa y compara tres modelos de clasificación sobre un escenario semi-supervisado: un SVM supervisado como baseline, un Transductive SVM (TSVM) y un Laplacian SVM (LapSVM). Los dos últimos aprovechan datos no etiquetados durante el entrenamiento para mejorar el margen de decisión. Los experimentos se realizan sobre el dataset Adult Census Income con tres niveles de etiquetado: 5%, 10% y 20% de los datos de entrenamiento.

## Dataset

**Adult Census Income** — UCI Machine Learning Repository. Contiene 48,842 registros con 15 variables socioeconómicas (edad, ocupación, nivel educativo, entre otras) y una variable objetivo binaria que indica si el ingreso anual de una persona supera los 50,000 USD.

## Requisitos

Python 3.8 o superior. Para instalar las dependencias ejecutar:

```bash
pip install numpy pandas scikit-learn scipy matplotlib seaborn
```

## Estructura del repositorio

```
.
├── data/
│   ├── raw/                  <- dataset original
│   └── processed/            <- datos limpios, escalados y divididos en train/test
├── eda/
│   ├── exploracion.py
│   └── preprocesamiento.py
├── modelos/
│   ├── utils.py
│   ├── baseline_svm.py
│   └── semisupervisados/
│       ├── tsvm.py
│       └── lapsvm.py
├── experimentacion/
│   ├── comparacion_etiquetas.py
│   └── sensibilidad.py
├── visualizaciones/
│   ├── graficos_eda.py
│   ├── graficos_modelos.py
│   ├── graficos_sensibilidad.py
│   └── graficos/
│       ├── eda/
│       ├── modelos/
│       └── sensibilidad/
└── resultados/
    ├── baseline_metrics.csv
    ├── tsvm_metrics.csv
    ├── lapsvm_metrics.csv
    ├── sensibilidad_grid.csv
    └── comparacion_final.csv
```

## Cómo correr

Los scripts deben ejecutarse en el siguiente orden desde la raíz del repositorio:

```bash
# Fase 1 — Preprocesamiento
python eda/preprocesamiento.py        # limpia el dataset, aplica encoding, escala con StandardScaler y genera los CSVs en data/processed/
python eda/exploracion.py             # genera estadísticas descriptivas y análisis exploratorio del dataset

# Fase 2 — Modelos (pueden correrse en paralelo)
python modelos/baseline_svm.py                    # entrena SVC supervisado en los escenarios 5%, 10% y 20%
python modelos/semisupervisados/tsvm.py           # entrena TSVM semi-supervisado en los mismos escenarios
python modelos/semisupervisados/lapsvm.py         # entrena LapSVM semi-supervisado en los mismos escenarios

# Fase 3 — Experimentación (requiere que los tres modelos hayan corrido)
python experimentacion/comparacion_etiquetas.py   # consolida las métricas de los tres modelos en una tabla comparativa
python experimentacion/sensibilidad.py            # realiza grid search de hiperparámetros para TSVM y LapSVM

# Fase 4 — Visualizaciones
python visualizaciones/graficos_eda.py            # genera gráficos del análisis exploratorio
python visualizaciones/graficos_modelos.py        # genera curvas de desempeño, matrices de confusión y comparación de predicciones
python visualizaciones/graficos_sensibilidad.py   # genera heatmaps y curvas de validación del análisis de sensibilidad
```

## Resultados

Los CSVs con métricas se guardan en `resultados/` y todos los gráficos se exportan en `visualizaciones/graficos/`, organizados por subcarpeta según la etapa del análisis.

