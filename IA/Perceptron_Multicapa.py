# 1. Importación de librerías
# numpy: permite trabajar con arreglos y operaciones matemáticas.
# tensorflow y keras: sirven para construir y entrenar la red neuronal.
# sklearn: genera datos, divide el conjunto y mide el rendimiento del modelo.
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (accuracy_score, confusion_matrix,
                              classification_report, roc_auc_score)

# 2. Generación del dataset
# Se crea un conjunto de datos sintético para clasificación binaria.
# 1000 ejemplos, 12 variables de entrada, 2 clases posibles.
X, y = make_classification(n_samples=1000, n_features=12, n_informative=8,
                            n_redundant=2, n_classes=2, random_state=42)

# Separación en entrenamiento y prueba.
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

# 3. Normalización de los datos
# Convierte cada característica a media 0 y desviación estándar 1.
# Esto ayuda a que la red aprenda mejor y más rápido.
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 4. Definición de la arquitectura del Perceptrón Multicapa
# Una red con dos capas ocultas y una capa de salida.
# La capa oculta usa ReLU, y la salida usa Sigmoid porque es clasificación binaria.
modelo = keras.Sequential([
    layers.Input(shape=(12,)),
    layers.Dense(16, activation='relu'),   # capa oculta 1
    layers.Dense(8, activation='relu'),    # capa oculta 2
    layers.Dense(1, activation='sigmoid')  # salida binaria
])

# 5. Compilación del modelo
# Adam: optimizador; binary_crossentropy: pérdida para clasificación binaria;
# accuracy: métrica de evaluación.
modelo.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.001),
    loss='binary_crossentropy',
    metrics=['accuracy']
)

modelo.summary()

# 6. Entrenamiento mediante retropropagación
# El modelo ajusta sus pesos para minimizar la pérdida.
historial = modelo.fit(
    X_train_scaled, y_train,
    validation_split=0.2,
    epochs=100,
    batch_size=32,
    verbose=1
)

# 7. Evaluación del modelo sobre el conjunto de prueba
# Se convierte la probabilidad en una etiqueta final (0 o 1).
y_proba = modelo.predict(X_test_scaled).ravel()
y_pred = (y_proba >= 0.5).astype(int)

# 8. Métricas de evaluación
print("Accuracy:", accuracy_score(y_test, y_pred))
print("ROC AUC:", roc_auc_score(y_test, y_proba))
print("\nMatriz de confusión:\n", confusion_matrix(y_test, y_pred))
print("\nReporte de clasificación:\n", classification_report(y_test, y_pred))