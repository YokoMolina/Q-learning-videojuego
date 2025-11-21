# Q-learning-videojuego

Este proyecto integra visión por computadora con aprendizaje por refuerzo para controlar un agente en un videojuego, utilizando distintas estrategias de Q-learning.

## Visión por Computadora  
El agente observa el entorno del juego a través de la entrada visual (fotogramas / capturas de pantalla), procesa esa información para extraer el estado relevante, y luego toma decisiones de acción mediante Q-learning. De este modo, la percepción visual está directamente integrada al flujo de control del agente.

## Modelos utilizados  
En el proyecto se han implementado los siguientes modelos/algoritmos:
- Algoritmo clásico de Q‑Learning (tabla Q) sobre estados discretizados extraídos de la imagen.  
- Algoritmo de Deep Q‑Learning (DQN) donde la entrada es procesada por una red neuronal (por ejemplo, una red convolucional) para estimar la función Q.  
- (Opcional) Algoritmo híbrido con visión + procesamiento adicional (pre-procesamiento de imágenes, extracción de características) para optimizar el aprendizaje.

## Flujo general del proyecto  
1. Captura de imagen del juego → preprocesamiento (por ejemplo escala de grises, reducción de resolución, recorte).  
2. Extracción del estado visual relevante (por ejemplo segmentos, detección de objetos, posición del agente/enemigo).  
3. Entrada del estado al modelo de Q-learning (red neuronal) → cálculo de valores Q(s, a) → elección de acción mediante estrategia ε-greedy.  
4. Ejecución de la acción en el entorno del videojuego → obtención de recompensa y nuevo estado.  
5. Actualización de la tabla Q o entrenamiento de la red (en el caso DQN).  
6. Repetición de episodios hasta convergencia del comportamiento.

## Cómo ejecutar  
1. Clona el repositorio:  
   ```bash
   git clone https://github.com/YokoMolina/Q-learning-videojuego.git
