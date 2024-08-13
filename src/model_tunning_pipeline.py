import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.wrappers.scikit_learn import KerasRegressor
from tensorflow.keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt
import os

# Función para crear el modelo de Keras
def create_model():
    model = Sequential([
        Dense(128, activation='relu', input_shape=(5,)),  # Ajusta según las características
        Dropout(0.3),
        Dense(64, activation='relu'),
        Dropout(0.3),
        Dense(32, activation='relu'),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mse')
    return model

# Crear el pipeline
pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='mean')),  # Imputación de valores nulos
    ('scaler', StandardScaler()),  # Estandarización de características
    ('model', KerasRegressor(build_fn=create_model, epochs=200, batch_size=32, verbose=0))  # Modelo
])

# Crear directorios para guardar modelos si no existen
os.makedirs('models', exist_ok=True)
os.makedirs('reports', exist_ok=True)

# Cargar los datos
train_data = pd.read_csv('data/train.csv')

# Selección de características y variable objetivo
features = ['OverallQual', 'GrLivArea', 'GarageCars', 'GarageArea', 'TotalBsmtSF']
target = 'SalePrice'

X = train_data[features]
y = train_data[target]

# División de datos en conjuntos de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Entrenar el pipeline con early stopping
early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
pipeline.named_steps['model'].fit(X_train, y_train, validation_data=(X_test, y_test), callbacks=[early_stopping])

# Guardar el modelo entrenado con regularización
pipeline.named_steps['model'].model.save('models/nn_model_tuned.h5')

# Predicciones y evaluación
y_pred = pipeline.predict(X_test)
rmse_tuned = np.sqrt(mean_squared_error(y_test, y_pred))
print(f'RMSE después de ajuste: {rmse_tuned}')

# Guardar los resultados en un archivo
with open('reports/tuning_report_nn.txt', 'w') as f:
    f.write(f'RMSE después de ajuste: {rmse_tuned}\n')

# Gráfica de la pérdida durante el entrenamiento
history = pipeline.named_steps['model'].model.history
plt.figure(figsize=(10, 6))
plt.plot(history.history['loss'], label='Pérdida de entrenamiento')
plt.plot(history.history['val_loss'], label='Pérdida de validación')
plt.title('Pérdida durante el entrenamiento (con ajuste)')
plt.xlabel('Época')
plt.ylabel('Pérdida')
plt.legend()
plt.grid(True)
plt.savefig('reports/loss_curve_tuned.png')
plt.show()

# Gráfica de predicciones vs valores reales
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred)
plt.xlabel('Valores Reales')
plt.ylabel('Predicciones')
plt.title('Predicciones vs Valores Reales (con ajuste)')
plt.grid(True)
plt.savefig('reports/pred_vs_real_tuned.png')
plt.show()
