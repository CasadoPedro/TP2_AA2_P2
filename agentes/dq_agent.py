from agentes.base import Agent
import numpy as np
from collections import defaultdict
import pickle
import random


class QAgent(Agent):
    """
    Agente de Q-Learning.
    Completar la discretización del estado y la función de acción.
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
                print(
                    f"Archivo Q-table no encontrado en {load_q_table_path}. Se inicia una nueva Q-table vacía."
                )
                self.q_table = defaultdict(lambda: np.zeros(len(self.actions)))
        else:
            self.q_table = defaultdict(lambda: np.zeros(len(self.actions)))
        # Parámetros de discretización (AJUSTAR ESTOS)
        self.num_bins = {
            "player_y": 15,
            "relative_center_y": 10,  # Diferencia vertical entre jugador y centro de la apertura en las tuberias
            "player_velocity_sign": 3,  # 1 (up), 0 (still), -1 (down)
            "player_velocity_y_sign": 5,  # Más granularidad para la velocidad Y
            "next_pipe_distance": 5,  # Distancia a la siguiente tubería
        }
        self.player_v_threshold_slow = 2.0
        self.player_v_threshold_fast = 5.0
        self.player_v_threshold_still = 0.5

    def discretize_state(self, state):
        """
        Discretiza el estado continuo en un estado discreto (tupla).
        COMPLETAR: Implementar la discretización adecuada para el entorno.
        """
        # state_dict['player_y'] es el centro de la paleta del jugador (de FlappyBird.getGameState)
        player_center_y = state["player_y"]
        # state_dict['player_vel'] es la velocidad del jugador (de FlappyBird.getGameState)
        player_velocity = state["player_vel"]
        # 1. Posición relativa del centro del jugador respecto al centro del gap en las tuberías
        relative_gap_center_y = (
            state["next_pipe_top_y"] - state["next_pipe_bottom_y"] - player_center_y
        )
        # Rango de relative_ball_y es aprox. [-game_height, game_height]
        # Escalamos esto a [0, 1] para los bins, considerando un rango efectivo de
        # [-game_height/2, game_height/2] centrado en 0.
        # (valor + rango_max_positivo) / rango_total
        # Si relative_ball_y es -H/2 => (-H/2 + H/2) / H = 0
        # Si relative_ball_y es +H/2 => (H/2 + H/2) / H = 1
        scaled_relative_gap_center_y = (
            relative_gap_center_y + self.game.height / 2
        ) / self.game.height
        relative_gap_y_bin = int(
            np.clip(
                scaled_relative_gap_center_y * self.num_bins["relative_center_y"],
                0,
                self.num_bins["relative_center_y"] - 1,
            )
        )
        # 2. Posicion relativa del centro del jugador respecto al centro del gap en segunda tubería cercana
        relative_gap_center_y = (
            state["next_next_pipe_top_y"]
            - state["next_next_pipe_bottom_y"]
            - player_center_y
        )
        # Rango de relative_ball_y es aprox. [-game_height, game_height]
        # Escalamos esto a [0, 1] para los bins, considerando un rango efectivo de
        # [-game_height/2, game_height/2] centrado en 0.
        # (valor + rango_max_positivo) / rango_total
        # Si relative_ball_y es -H/2 => (-H/2 + H/2) / H = 0
        # Si relative_ball_y es +H/2 => (H/2 + H/2) / H = 1
        scaled_relative_gap_center_y = (
            relative_gap_center_y + self.game.height / 2
        ) / self.game.height
        relative_gap_y_bin = int(
            np.clip(
                scaled_relative_gap_center_y * self.num_bins["relative_center_y"],
                0,
                self.num_bins["relative_center_y"] - 1,
            )
        )
        # 3. Distancia a la siguiente tubería
        next_pipe_distance = state["next_pipe_dist_to_player"]
        next_pipe_distance_bin = int(
            np.clip(
                next_pipe_distance
                / self.game.width
                * self.num_bins["next_pipe_distance"],
                0,
                self.num_bins["next_pipe_distance"] - 1,
            )
        )
        # 4. Distancia a la segunda tubería cercana
        next_next_pipe_distance = state["next_next_pipe_dist_to_player"]
        next_next_pipe_distance_bin = int(
            np.clip(
                next_next_pipe_distance
                / self.game.width
                * self.num_bins["next_pipe_distance"],
                0,
                self.num_bins["next_pipe_distance"] - 1,
            )
        )

        # 2. Signo de la velocidad del jugador
        if player_velocity < -self.player_v_threshold_still:
            player_velocity_sign_bin = 1  # Moviéndose arriba
        elif player_velocity > self.player_v_threshold_still:
            player_velocity_sign_bin = -1  # Moviéndose abajo
        else:
            player_velocity_sign_bin = 0  # Quieto o casi quieto

        # 3. Posición Y del jugador normalizada y discretizada
        player_y_normalized = player_center_y / self.game.height
        player_y_bin = int(
            np.clip(
                player_y_normalized * self.num_bins["player_y"],
                0,
                self.num_bins["player_y"] - 1,
            )
        )

        # 4. Dirección Y del jugador (más granular)
        if player_velocity < -self.player_v_threshold_fast:  # Arriba muy rápido
            player_vy_sign_bin = 0
        elif player_velocity < -self.player_v_threshold_slow:  # Arriba rápido
            player_vy_sign_bin = 1
        elif (
            player_velocity <= self.player_v_threshold_slow
        ):  # Lento o quieta (incluye 0)
            player_vy_sign_bin = 2
        elif player_velocity <= self.player_v_threshold_fast:  # Abajo rápido
            player_vy_sign_bin = 3
        else:  # Abajo muy rápido
            player_vy_sign_bin = 4
        # num_bins['player_velocity_y_sign'] debe ser 5

        return (
            relative_gap_y_bin,
            player_velocity_sign_bin,
            player_y_bin,
            player_vy_sign_bin,
            next_pipe_distance_bin,
            next_next_pipe_distance_bin,
        )

    def act(self, state):
        """
        Elige una acción usando epsilon-greedy sobre la Q-table.
        COMPLETAR: Implementar la política epsilon-greedy.
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
            print(
                f"Archivo Q-table no encontrado en {path}. Se inicia una nueva Q-table vacía."
            )
            self.q_table = defaultdict(lambda: np.zeros(len(self.actions)))
