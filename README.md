# Proyecto RL P1 Ball Pursuit RoboCup 2D

Curso DS5345 Aprendizaje por Refuerzo. Tarea acotada de persecucion e intercepcion de balon con Monte Carlo Control tabular.

## Tarea Ball Pursuit

El agente inicia en posicion arbitraria cerca de x menos 15 y el balon cerca del centro a distancia estocastica entre 5 y 40 m. Debe orientarse y avanzar hasta capturar con d_b menor o igual a 0.8 m en menos de 40 pasos.

Estado discreto de 20 estados con 4 zonas de distancia y 5 sectores de angulo. Acciones DASH 100, DASH 50, TURN mas 35 y TURN menos 35. Recompensa oficial delta d_b menos 0.2 mas 100 si hay captura. Gamma 0.99. Tabla Q de 20 por 4.

## Algoritmo

Monte Carlo Control On Policy First Visit con politica epsilon greedy. Se generan episodios completos, se calcula G con gamma, se promedia la primera visita de cada par estado accion y la politica greedy es argmax de Q. Comparacion justa con mismo entorno, recompensa, gamma y 3500 episodios. Rama A epsilon constante 0.1. Rama B epsilon decreciente 1.0 por 0.998 elevado al episodio con minimo 0.05.

## Inicio con Docker

```bash
cd RL-Project
docker compose up -d --build
docker compose ps
```

Jupyter Lab queda en el puerto 8888 sin token. En Lightning se expone el puerto 8888 y se abre la URL del estudio. El servidor rcssserver usa UDP 6000. No se necesita Docker Desktop ni pantalla. Todo es con matplotlib dentro del cuaderno.

## Cuaderno a ejecutar

`workspace/agente_cero_mc_control_persecucion.ipynb`. Abrir en Jupyter, menu Kernel, Restart Kernel and Run All. Guarda figuras y CSV en `results/`.

## Verificar conexion

```bash
docker compose exec rl-agent python /workspace/test_random_agent.py
```

Debe mostrar distancias del balon y acciones DASH o TURN por 50 pasos. Tambien sirve el tutorial `workspace/tutorial_primer_paso.ipynb`.

## Resultados

Carpeta `results/` con `learning_curves.png`, `success_rate.png`, `steps_per_episode.png`, `policy_value.png`, `trajectory.png` y `metrics.csv` con columnas episodio, estrategia, retorno, exito, pasos y epsilon. Resumen en `docs/results.md`.

## Resumen observado

Semilla 42 y 3500 episodios por rama. Decreciente logra G0 cerca de 85, exito 100 por ciento en los ultimos 100 y 27.5 pasos. Constante 0.1 queda en G0 cerca de menos 9, exito 0 por ciento y 40 pasos. La evaluacion greedy de 500 episodios da 100 por ciento y 26.8 pasos para decreciente y 0 por ciento para constante. El criterio mayor a 90 por ciento en menos de 40 pasos se cumple con la politica decreciente. La politica gira hacia el balon y avanza con DASH 100 de frente.
