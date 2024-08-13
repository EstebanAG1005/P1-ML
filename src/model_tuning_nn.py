# -*- coding: utf-8 -*-
"""Model_Tuning_NN.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1xtyVaBaDe09-qaSniBa0o2YqroCc3rKn
"""

# notebooks/Model_Tuning_NN.ipynb

# Instalación de dependencias necesarias
!pip install pandas numpy scikit-learn tensorflow

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
import joblib
import matplotlib.pyplot as plt
import os

# Crear directorios para guardar modelos si no existen
os.makedirs('../models', exist_ok=True)
os.makedirs('../reports', exist_ok=True)

# Cargar los datos
train_data = pd.read_csv('../data/train.csv')

# Preprocesamiento de datos: Imputación de valores nulos en columnas numéricas
numeric_cols = train_data.select_dtypes(include=[np.number]).columns
train_data_imputed = train_data.copy()
train_data_imputed[numeric_cols] = train_data[numeric_cols].fillna(train_data[numeric_cols].mean())

# Selección de características y variable objetivo
features = ['OverallQual', 'GrLivArea', 'GarageCars', 'GarageArea', 'TotalBsmtSF']
target = 'SalePrice'

X = train_data_imputed[features]
y = train_data_imputed[target]

# Estandarización de características
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# División de datos en conjuntos de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Definición del modelo de red neuronal con regularización
model = Sequential([
    Dense(128, activation='relu', input_shape=(X_train.shape[1],)),
    Dropout(0.3),
    Dense(64, activation='relu'),
    Dropout(0.3),
    Dense(32, activation='relu'),
    Dense(1)
])

# Compilación del modelo
model.compile(optimizer='adam', loss='mse')

# Callback para detener el entrenamiento si no hay mejora
early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

# Entrenamiento del modelo
history = model.fit(X_train, y_train, epochs=200, validation_data=(X_test, y_test), callbacks=[early_stopping])

# Guardar el modelo entrenado con regularización
model.save('../models/nn_model_tuned.h5')

# Predicciones y evaluación
y_pred = model.predict(X_test).flatten()
rmse_tuned = np.sqrt(mean_squared_error(y_test, y_pred))
print(f'RMSE después de ajuste: {rmse_tuned}')

# Guardar los resultados en un archivo
with open('../reports/tuning_report_nn.txt', 'w') as f:
    f.write(f'RMSE después de ajuste: {rmse_tuned}\n')

# Gráfica de la pérdida durante el entrenamiento
plt.figure(figsize=(10, 6))
plt.plot(history.history['loss'], label='Pérdida de entrenamiento')
plt.plot(history.history['val_loss'], label='Pérdida de validación')
plt.title('Pérdida durante el entrenamiento (con ajuste)')
plt.xlabel('Época')
plt.ylabel('Pérdida')
plt.legend()
plt.grid(True)
plt.savefig('../reports/loss_curve_tuned.png')
plt.show()

# Gráfica de predicciones vs valores reales
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred)
plt.xlabel('Valores Reales')
plt.ylabel('Predicciones')
plt.title('Predicciones vs Valores Reales (con ajuste)')
plt.grid(True)
plt.savefig('../reports/pred_vs_real_tuned.png')
plt.show()