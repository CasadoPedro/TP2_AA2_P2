from agentes.base import Agent
import numpy as np
import tensorflow as tf

class NNAgent(Agent):
    """
    Agente que utiliza una red neuronal entrenada para aproximar la Q-table.
    La red debe estar guardada como TensorFlow SavedModel.
    """
    def __init__(self, actions, game=None, model_path='flappy_q_nn_model'):
        super().__init__(actions, game)
        # Cargar el modelo entrenado
        self.model = tf.keras.models.load_model(model_path)

     
    def act(self, state):
        """
        Usa la red neuronal entrenada para elegir la mejor acción dada un estado.
        """
        state_array = np.array(state).reshape(1, -1)  # Formato compatible con la red
        q_values = self.model.predict(state_array, verbose=0)  # Predicción
        action_idx = np.argmax(q_values)
        return self.actions[action_idx]


