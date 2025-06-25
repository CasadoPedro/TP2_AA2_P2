import numpy as np
import pickle
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau


# --- Cargar Q-table entrenada ---
QTABLE_PATH = "flappy_birds_q_table_final.pkl"
MODEL_PATH = "flappy_q_nn_model.keras"
with open(QTABLE_PATH, "rb") as f:
    q_table = pickle.load(f)

# --- Preparar datos para entrenamiento ---
# Convertir la Q-table en X (estados) e y (valores Q para cada acción)
# --- Preparar datos ---
X, y = zip(*q_table.items())
X = np.array(X, dtype=np.float32)
y = np.array(y, dtype=np.float32)
print(f"Dataset: {len(X)} estados | X shape: {X.shape}, y shape: {y.shape}")
print(f"Rango de Q-values: {np.min(y):.3f} a {np.max(y):.3f}")

# --- Definir la red neuronal ---
model = keras.Sequential(
    [
        layers.Input(shape=(X.shape[1],)),  # Tamaño del estado
        # Capas ocultas
        layers.Dense(32, activation="relu"),
        layers.Dropout(0.3),
        layers.Dense(64, activation="relu"),
        layers.Dense(y.shape[1]),  # cantidad de acciones posibles
    ]
)

model.compile(optimizer="adam", loss="mse", metrics=["mae"])
# Usamos EarlyStopping para evitar sobreajuste
early_stop = EarlyStopping(monitor="val_loss", patience=20, restore_best_weights=True)
# Reduce el learning rate si la validación no mejora
reduce_lr = ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=10, min_lr=1e-5, verbose=1)
# --- Entrenar la red neuronal ---
history = model.fit(
    X, y, epochs=150, batch_size=64, validation_split=0.2, shuffle=True, verbose=1, callbacks=[early_stop, reduce_lr]
)
# --- Visualización ---
plt.figure(figsize=(10, 4))

plt.subplot(1, 2, 1)
plt.plot(history.history["loss"], label="MSE - Entrenamiento")
plt.plot(history.history["val_loss"], label="MSE - Validación")
plt.title("Pérdida (MSE)")
plt.xlabel("Épocas")
plt.ylabel("Error cuadrático medio")
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history["mae"], label="MAE - Entrenamiento")
plt.plot(history.history["val_mae"], label="MAE - Validación")
plt.title("Error absoluto medio (MAE)")
plt.xlabel("Épocas")
plt.ylabel("MAE")
plt.legend()

plt.tight_layout()
plt.savefig("training_metrics.png")  # Guardamos la imagen de las métricas

# --- Guardar el modelo entrenado ---

model.save(MODEL_PATH)
print(f"Modelo guardado como TensorFlow SavedModel en: {MODEL_PATH}")
