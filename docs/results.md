# Resultados P1 Ball Pursuit

## MDP

S de 20 estados. Distancia en 4 zonas con corte en 0.8, 3 y 8 m. Angulo en 5 sectores con cortes en 15 y 60 grados. A de 4 acciones DASH 100, DASH 50, TURN mas 35 y TURN menos 35. P inducida por avance de 1.0 o 0.5 m y giros de 35 grados mas la dinamica del simulador. R igual a delta d_b menos 0.2 mas 100 si d_b menor o igual a 0.8. Gamma 0.99. Q de 20 por 4 con 80 valores. V es maximo de Q y pi es argmax de Q.

La distancia inicial medida en 500 muestras da minimo 11.25, maximo 19.03 y media 15.27, dentro del rango oficial de 5 a 40. Maximo de 40 pasos por episodio segun el criterio de exito. Se corrigio la recompensa del starter a la oficial y el horizonte de 80 a 40.

## Estrategias de exploracion

Mismo entorno, recompensa, gamma, episodios y semilla 42. Constante con epsilon 0.1. Decreciente con epsilon igual a maximo entre 0.05 y 1.0 por 0.998 elevado al episodio. N igual a 3500 por rama.

## Hiperparametros

Gamma 0.99. Episodios 3500. Max pasos 40. Epsilon constante 0.1. Decreciente inicio 1.0, minimo 0.05 y decaimiento 0.998. Semilla 42.

## Numeros finales

Promedio de los ultimos 100 episodios de entrenamiento. Decreciente con G0 84.76, exito 1.000 y pasos 27.52. Constante con G0 menos 8.81, exito 0.000 y pasos 40.00. Evaluacion greedy de 500 episodios con semilla 99. Decreciente con exito 1.000 y pasos 26.76. Constante con exito 0.000 y pasos 40.00.

## Lectura de curvas

G0 separa rapido a las dos ramas. Decreciente sube y se estabiliza arriba de 80. Constante queda plana cerca de menos 9. La tasa de exito con media movil 100 llega a 1.0 en decreciente y a 0.0 en constante. Los pasos bajan a 27 en decreciente y quedan en 40 en constante. La diferencia se explica porque el decaimiento visita la captura al inicio con epsilon alto y luego explota, mientras epsilon 0.1 fijo se atasca sin visitar la secuencia de captura.

## Politica aprendida

De frente la accion es DASH 100 en casi todas las distancias. Con el balon a la derecha aprende GIRAR DER y a la izquierda GIRAR IZQ. Atras tambien gira para realinearse. V crece cerca del balon por el bono de 100 y baja lejos por el costo por paso.

## Archivos

`results/learning_curves.png`, `results/success_rate.png`, `results/steps_per_episode.png`, `results/policy_value.png`, `results/trajectory.png` y `results/metrics.csv`.
