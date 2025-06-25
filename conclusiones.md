# Conclusiones

## Ingeniería de características (Discretización del estado del juego)

Para entrenar un agente de Q-Learning que juegue Flappy Bird, fue necesario transformar el estado continuo del juego en una representación discreta adecuada para una tabla Q. Esta ingeniería de características fue clave para el rendimiento del agente.

### Discretización clásica

Se diseñó una función `discretize_state` que transforma el estado continuo en una tupla discreta de tres variables:

1. **Diferencia vertical entre el centro del jugador y el centro del gap**  
   Esta variable (`relative_gap_center_y`) condensa tres variables originales (posición del jugador, parte superior e inferior de la tubería) en una sola diferencia relativa. Se discretizó de forma simétrica en 30 bins centrados en 0. Esto permite distinguir si el jugador está más arriba o abajo del centro del gap, y favorece interpretabilidad y aprendizaje.

2. **Velocidad vertical del jugador**  
   Se discretizó en 9 bins, también centrados en 0 (de -4 a +4). Esta variable demostró ser crítica para una buena toma de decisiones, especialmente en momentos cercanos a obstáculos.

3. **Distancia horizontal a la siguiente tubería**  
   Se discretizó en 30 bins. Permite anticipar si se debe comenzar a subir o bajar en base a la proximidad del obstáculo.

Este diseño se enfocó en **usar pocas variables, pero con alta resolución**, para mantener un espacio de estados compacto pero expresivo.

### Discretización extendida

Con el objetivo de entrenar un agente basado en red neuronal que aproximara la Q-table, se implementó una segunda función `discretize_state_2`, que genera una representación más rica del estado:

1. Diferencia vertical con el centro del primer gap  
2. Diferencia vertical con el centro del segundo gap  
3. Velocidad vertical  
4. Distancia horizontal al siguiente obstáculo  
5. Altura absoluta del jugador

Cada variable fue discretizada en 20 bins (salvo la velocidad, que se discretizó en 9). Aunque este esquema es más complejo, generó una Q-table más densa y variada, lo que resultó crucial para el entrenamiento exitoso de la red neuronal.

## Comparación de resultados

Se entrenaron y evaluaron dos tipos de agentes, cada uno con ambas discretizaciones:

- **Agente Q-Learning clásico**, que usa la tabla Q directamente
- **Agente con red neuronal (DQN)**, que aprende a imitar una Q-table generada previamente

### Evaluación de cada combinación

| Q-table | Agente Q-Learning | Agente DQN         |
|--------|--------------------|---------------------|
| `discretize_state`  (pocas variables, alta resolución) | ✅ Excelente rendimiento | ❌ NN no logra aproximarla bien |
| `discretize_state_2` (más variables, menos resolución) | ⚠️ Peor que la anterior | ✅ Satisfactorio |

### Observaciones:

- El **agente Q-Learning clásico** funcionó mucho mejor con la discretización más simple. Tener pocas variables pero muchos bins fue clave para lograr políticas efectivas y estables.

- En cambio, la **red neuronal no pudo aproximar adecuadamente** la Q-table generada con pocas variables, debido a que el dataset era demasiado pequeño (~3000 registros). Al entrenarse sobre la Q-table generada con la segunda discretización, el rendimiento del agente DQN mejoró considerablemente.

- Esto confirma que no solo importa qué variables se discretizan y cómo, sino también **cuántos ejemplos únicos se generan**, especialmente cuando se entrena un modelo supervisado.

### Métricas y evaluación

Ambos agentes fueron evaluados con `test_agent.py`, que mide la recompensa por episodio. Como se esperaba:

- El mejor rendimiento general fue del **agente Q-learning con `discretize_state`**
- El **agente DQN con `discretize_state_2`** logró resultados aceptables, gracias a una Q-table más densa y variada

### Conclusión general

La ingeniería de características fue determinante para el éxito del agente. La clave fue encontrar un **equilibrio entre cantidad de variables y resolución de discretización**, en función del tipo de agente. Para Q-learning clásico, fue mejor usar **pocas variables y alta precisión**. Para redes neuronales, fue necesario ampliar la representación del estado para que la red pueda generalizar adecuadamente sobre suficientes ejemplos.

