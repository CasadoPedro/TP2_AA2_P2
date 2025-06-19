import numpy as np
import pickle
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.callbacks import EarlyStopping

# --- Cargar Q-table entrenada ---
QTABLE_PATH = "flappy_birds_q_table_final.pkl"  # Cambia el path si es necesario
with open(QTABLE_PATH, "rb") as f:
    q_table = pickle.load(f)

# --- Preparar datos para entrenamiento ---
# Convertir la Q-table en X (estados) e y (valores Q para cada acción)
X = []  # Estados discretos
y = []  # Q-values para cada acción
for state, q_values in q_table.items():
    X.append(state)
    y.append(q_values)
X = np.array(X)
y = np.array(y)
print(f"Datos cargados: {len(X)} estados, {len(y)} valores Q")
print(f"Forma de X: {X.shape}, Forma de y: {y.shape}")
print(f"Primer estado: {X[0]}, Primer valor Q: {y[0]}")
print(np.min(y), np.max(y))

# --- Definir la red neuronal ---
model = keras.Sequential(
    [
        layers.Input(shape=(X.shape[1],)),  # Tamaño del estado
        # Normalización de entrada
        # layers.BatchNormalization(),
        # Capas ocultas
        layers.Dense(32, activation="relu"),
        layers.Dropout(0.2),
        layers.Dense(32, activation="relu"),
        layers.Dense(y.shape[1]),  # cantidad de acciones posibles
    ]
)

model.compile(optimizer="adam", loss="mse", metrics=["accuracy"])
# Usamos EarlyStopping para evitar sobreajuste
early_stop = EarlyStopping(monitor="val_loss", patience=15, restore_best_weights=True)

# --- Entrenar la red neuronal ---
history = model.fit(X, y, epochs=150, batch_size=64, validation_split=0.2, verbose=1, callbacks=[early_stop])
# --- Mostrar resultados del entrenamiento ---
plt.plot(history.history["loss"], label="Loss")
plt.plot(history.history["val_loss"], label="Val Loss")
plt.xlabel("Época")
plt.ylabel("Error cuadrático medio")
plt.legend()
plt.title("Curva de entrenamiento del modelo Q-NN")
# plt.show()

# --- Guardar el modelo entrenado ---

model.save("flappy_q_nn_model.keras")
print("Modelo guardado como TensorFlow SavedModel en flappy_q_nn_model/")

# --- Notas para los alumnos ---
# - Puedes modificar la arquitectura de la red y los hiperparámetros.
# - Puedes usar la red entrenada para aproximar la Q-table y luego usarla en un agente tipo DQN.
# - Si tu estado es una tupla de enteros, no hace falta normalizar, pero puedes probarlo.
# - Si tienes dudas sobre cómo usar el modelo para predecir acciones, consulta la documentación de Keras.
