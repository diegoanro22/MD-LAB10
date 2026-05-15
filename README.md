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

