from agentes.base import Agent
import numpy as np
from collections import defaultdict
import pickle
import random


class QAgent(Agent):
    """
    Agente de Q-Learning.
    """

    def __init__(
        self,
        actions,
        game=None,
        learning_rate=0.1,
        discount_factor=0.99,
        epsilon=1.0,
        epsilon_decay=0.995,
        min_epsilon=0.01,
        load_q_table_path="flappy_birds_q_table.pkl",
    ):
        super().__init__(actions, game)
        self.lr = learning_rate
        self.gamma = discount_factor
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.min_epsilon = min_epsilon
        if load_q_table_path:
            try:
                with open(load_q_table_path, "rb") as f:
                    q_dict = pickle.load(f)
                self.q_table = defaultdict(lambda: np.zeros(len(self.actions)), q_dict)
                print(f"Q-table cargada desde {load_q_table_path}")
            except FileNotFoundError:
                print(f"Archivo Q-table no encontrado en {load_q_table_path}. Se inicia una nueva Q-table vacía.")
                self.q_table = defaultdict(lambda: np.zeros(len(self.actions)))
        else:
            self.q_table = defaultdict(lambda: np.zeros(len(self.actions)))
        # Definimos los bins para la discretización del estado
        self.num_bins = {
            "relative_center_y": 30,  # Diferencia vertical entre jugador y centro de la apertura en las tuberias
            "next_pipe_distance": 30,  # Distancia a la siguiente tubería
        }

    def discretize_state(self, state):
        """
        Discretiza el estado continuo en un estado discreto (tupla).
        """
        # state_dict['player_y'] es el centro de la paleta del jugador (de FlappyBird.getGameState)
        player_center_y = state["player_y"]

        # 1. Posición relativa del centro del jugador respecto al centro del gap en las tuberías
        relative_gap_center_y = player_center_y - (state["next_pipe_top_y"] + state["next_pipe_bottom_y"]) / 2

        max_relative_distance = self.game.height / 2  # o /3 para mayor resolución
        half_bins = self.num_bins["relative_center_y"] / 2
        # Clip distancia para que no se salga de rango
        clipped_relative = np.clip(relative_gap_center_y, -max_relative_distance, max_relative_distance)
        # Escalar a bin con signo
        bin_index = int(np.round((clipped_relative / max_relative_distance) * half_bins))
        # Asegurarse de que esté en el rango [-half_bins, half_bins]
        relative_gap_y_bin = int(np.clip(bin_index, -half_bins, half_bins))

        # 2. Distancia a la siguiente tubería
        next_pipe_distance = state["next_pipe_dist_to_player"]
        next_pipe_distance_bin = int(
            np.clip(
                next_pipe_distance / self.game.width * self.num_bins["next_pipe_distance"],
                0,
                self.num_bins["next_pipe_distance"] - 1,
            )
        )

        # 3. Velocidad del jugador en Y
        # state_dict['player_vel'] es la velocidad del jugador (de FlappyBird.getGameState)
        player_velocity = state["player_vel"]

        clipped_velocity = max(-10.0, min(10.0, player_velocity))

        # Escalamos a rango [0, 8] y redondeamos
        bin_index = int(round(((clipped_velocity + 10) / 20) * 8))

        # Convertimos a rango [-4, +4]
        player_vy_sign_bin = bin_index - 4

        return (relative_gap_y_bin, player_vy_sign_bin, next_pipe_distance_bin)

    def act(self, state):
        """
        Elige una acción usando epsilon-greedy sobre la Q-table.
        """
        discrete_state = self.discretize_state(state)
        if random.random() < self.epsilon:
            return random.choice(self.actions)
        else:
            q_values = self.q_table[discrete_state]
            return self.actions[np.argmax(q_values)]

    def update(self, state, action, reward, next_state, done):
        """
        Actualiza la Q-table usando la regla de Q-learning.
        """
        discrete_state = self.discretize_state(state)
        discrete_next_state = self.discretize_state(next_state)
        action_idx = self.actions.index(action)
        # Inicializar si el estado no está en la Q-table
        if discrete_state not in self.q_table:
            self.q_table[discrete_state] = np.zeros(len(self.actions))
        if discrete_next_state not in self.q_table:
            self.q_table[discrete_next_state] = np.zeros(len(self.actions))
        current_q = self.q_table[discrete_state][action_idx]
        max_future_q = 0
        if not done:
            max_future_q = np.max(self.q_table[discrete_next_state])
        new_q = current_q + self.lr * (reward + self.gamma * max_future_q - current_q)
        self.q_table[discrete_state][action_idx] = new_q

    def decay_epsilon(self):
        """
        Disminuye epsilon para reducir la exploración con el tiempo.
        """
        self.epsilon = max(self.min_epsilon, self.epsilon * self.epsilon_decay)

    def save_q_table(self, path):
        """
        Guarda la Q-table en un archivo usando pickle.
        """
        import pickle

        with open(path, "wb") as f:
            pickle.dump(dict(self.q_table), f)
        print(f"Q-table guardada en {path}")

    def load_q_table(self, path):
        """
        Carga la Q-table desde un archivo usando pickle.
        """
        import pickle

        try:
            with open(path, "rb") as f:
                q_dict = pickle.load(f)
            self.q_table = defaultdict(lambda: np.zeros(len(self.actions)), q_dict)
            print(f"Q-table cargada desde {path}")
        except FileNotFoundError:
            print(f"Archivo Q-table no encontrado en {path}. Se inicia una nueva Q-table vacía.")
            self.q_table = defaultdict(lambda: np.zeros(len(self.actions)))
