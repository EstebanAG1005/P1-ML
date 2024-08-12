# -*- coding: utf-8 -*-
"""EDA.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/16tR0I-9gtedlsiM3uSBzijPSK1UNB6Qb

## Setup
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# Cargar los datos
train_data = pd.read_csv('../data/train.csv')
test_data = pd.read_csv('../data/test.csv')

# Análisis Exploratorio de Datos (EDA)
# Descripción de las variables y valores nulos
train_description = train_data.describe(include='all')
train_nulls = train_data.isnull().sum()

"""## EDA"""

train_data.head()

train_data.info()

# Visualización de valores nulos
plt.figure(figsize=(10, 8))
sns.heatmap(train_data.isnull(), cbar=False, cmap='viridis')
plt.title('Mapa de Calor de Valores Nulos en el Conjunto de Entrenamiento')
plt.show()

# Resumen estadístico
train_data.describe(include='all')

# Tipos de datos y valores nulos
train_info = train_data.info()
train_nulls = train_data.isnull().sum()
print(train_nulls)

# Histograma para variables numéricas
num_cols = train_data.select_dtypes(include=[np.number]).columns
train_data[num_cols].hist(figsize=(15, 10), bins=30, layout=(len(num_cols) // 3 + 1, 3))
plt.suptitle('Distribución de Variables Numéricas')
plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.show()

# Conteo de valores para variables categóricas
cat_cols = train_data.select_dtypes(include=['object']).columns
for col in cat_cols:
    plt.figure(figsize=(10, 5))
    sns.countplot(x=col, data=train_data)
    plt.title(f'Distribución de la variable categórica: {col}')
    plt.xticks(rotation=45)
    plt.show()

# Seleccionar solo las columnas numéricas para la matriz de correlación
num_cols = train_data.select_dtypes(include=[np.number]).columns
corr_matrix = train_data[num_cols].corr()

# Visualizar el mapa de calor de la correlación
plt.figure(figsize=(12, 10))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Mapa de Calor de la Correlación entre Variables Numéricas')
plt.show()