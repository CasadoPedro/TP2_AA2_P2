from agentes.base import Agent
import numpy as np
import tensorflow as tf


class NNAgent(Agent):
    """
    Agente que utiliza una red neuronal entrenada para aproximar la Q-table.
    La red debe estar guardada como TensorFlow SavedModel.
    """

    def __init__(self, actions, game=None, model_path="flappy_q_nn_model.keras"):
        super().__init__(actions, game)
        # Cargar el modelo entrenado
        self.model = tf.keras.models.load_model(model_path)
        self.num_bins = {
            "relative_center_y": 20,  # Diferencia vertical entre jugador y centro de la apertura en las tuberías
            "next_pipe_distance": 20,  # Distancia a la siguiente tubería
        }

    def discretize_state(self, state):
        """
        Discretiza el estado continuo en un estado discreto (tupla).
        """
        # state['player_y'] es la posición en y del jugador
        player_center_y = state["player_y"]

        # 1. Posición relativa del centro del jugador respecto al centro del gap en las tuberías
        relative_gap_center_y = player_center_y - (state["next_pipe_top_y"] + state["next_pipe_bottom_y"]) / 2

        max_relative_distance = self.game.height / 2
        half_bins = self.num_bins_2["relative_center_y"] / 2

        # Clip distancia para que no se salga de rango
        clipped_relative = np.clip(relative_gap_center_y, -max_relative_distance, max_relative_distance)
        # Escalar a bin con signo
        bin_index = int(np.round((clipped_relative / max_relative_distance) * half_bins))
        # Asegurarse de que esté en el rango [-half_bins, half_bins]
        relative_gap_y_bin = int(np.clip(bin_index, -half_bins, half_bins))

        # 2. Posición relativa del centro del jugador respecto al gap en la segunda tubería
        relative_gap_center_y_2 = (
            player_center_y - (state["next_next_pipe_top_y"] + state["next_next_pipe_bottom_y"]) / 2
        )
        clipped_relative_2 = np.clip(relative_gap_center_y_2, -max_relative_distance, max_relative_distance)
        bin_index_2 = int(np.round((clipped_relative_2 / max_relative_distance) * half_bins))
        relative_gap_y_bin_2 = int(np.clip(bin_index_2, -half_bins, half_bins))

        # 3. Distancia a la siguiente tubería
        next_pipe_distance = state["next_pipe_dist_to_player"]
        next_pipe_distance_bin = int(
            np.clip(
                next_pipe_distance / self.game.width * self.num_bins_2["next_pipe_distance"],
                0,
                self.num_bins_2["next_pipe_distance"] - 1,
            )
        )

        # 4. Velocidad del jugador en Y
        # state['player_vel'] es la velocidad del jugador
        player_velocity = state["player_vel"]

        clipped_velocity = max(-10.0, min(10.0, player_velocity))

        # Escalamos a rango [0, 8] y redondeamos
        bin_index = int(round(((clipped_velocity + 10) / 20) * 8))

        # Convertimos a rango [-4, +4]
        player_vy_bin = bin_index - 4

        # 5. Altura del jugador
        player_height_bin = int(
            np.clip(
                (player_center_y / self.game.height) * self.num_bins_2["relative_center_y"],
                0,
                self.num_bins_2["relative_center_y"] - 1,
            )
        )
        return np.array(
            [
                relative_gap_y_bin,
                relative_gap_y_bin_2,
                player_vy_bin,
                next_pipe_distance_bin,
                player_height_bin,
            ],
            dtype=np.float32,
        )

    def act(self, state):
        """
        Usa la red neuronal entrenada para elegir la mejor acción dada un estado.
        """
        # Discretizamos el estado
        state_array = self.discretize_state(state).reshape(1, -1)  # Aseguramos que sea 2D para la predicción
        # Obtenemos los valores Q para el estado actual
        q_values = self.model(state_array, training=False).numpy()[0]
        # Elegimos la acción con el valor Q más alto
        action_idx = np.argmax(q_values)
        return self.actions[action_idx]
