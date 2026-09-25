# Guion de Leonardo Candio — slides 1 a 7 (≈ 6 minutos)

Lo que se dice en voz alta y no aparece en las slides. Tiempo por slide.

## Slide 1 · Portada (30 s)

Somos el grupo 1 del curso. El trabajo está en github.com/artrivas/RL-Project, y todo lo que van a ver
sale de corridas guardadas en el repositorio, no de figuras hechas a mano. La pregunta del proyecto fue
simple: ¿cuánto tarda un jugador en llegar a un balón quieto y cómo enseñamos eso con aprendizaje por
refuerzo tabular?

## Slide 2 · La tarea (50 s)

Contexto que no está en la slide: en RoboCup 2D el servidor manda un ciclo cada 100 ms, así que los 40
pasos del enunciado son 4 segundos reales de juego. Ojo con un detalle del enunciado que parece menor y
lo cambia todo: fija que d_0 esté entre 5 y 40 metros, pero no dice cómo se distribuye ese d_0. Nosotros
entrenamos con d_0 uniforme en todo el rango, que es la lectura más dura. El entorno del kit es a propósito
simple: DASH 100 mueve exactamente 1 metro, TURN 35 gira exactamente 35 grados, sin inercia ni ruido. Es
un sistema cinemático, no dinámico. Eso nos deja estudiar el algoritmo sin que la física ensucie el
análisis, y después medimos qué pasa cuando la física sí importa.

## Slide 3 · MDP (50 s)

Dos ideas que no están escritas. Primera: por qué 99 estados y no la posición continua, porque P1 pide
tabular. Segunda: el aliasing. La transición real es determinista; la estocástica aparece recién al
agrupar en bins. Ejemplo concreto para decir: si el balón está a 30 grados y giro 35, queda al frente; si
está a 80 grados y giro 35, sigue de lado. Los dos casos caen en el mismo sector angosto, entonces la
tabla no puede distinguirlos. Eso es aliasing, y es el precio de la discretización, no un error del
algoritmo. Aclarar que 99 son los estados no terminales y que el estado terminal de captura no lleva
valores Q.

## Slide 4 · Recompensa (50 s)

Lo que conviene explicar con la fórmula a la vista: el término de aproximación es telescópico. Si
sumamos d_t − d_{t+1} a lo largo del episodio, todo el interior se cancela y queda d_0 − d_T. Por eso
con gamma igual a 1 maximizar el retorno es casi lo mismo que minimizar pasos: d_T tras una captura es
siempre 0.8 metros o menos. El −0.2 por paso es el costo de tiempo y el bono de 100 es la captura. La
tarjeta de la derecha es la justificación de gamma 0.99: con 0.9 el bono de los últimos pasos vale 1.6,
es decir, el agente sería casi indiferente a capturar en el paso 39. Con 0.99 todavía vale 67.6. Decir
también que en el truncamiento hacemos bootstrap desde s′ porque el tiempo no está en el estado; si no,
la tabla aprendería valores distintos para lo que es esencialmente lo mismo.

## Slide 5 · Discretización (45 s)

Esta es la decisión de diseño que más nos costó. La tolerancia angular viene de la geometría: para capturar
a 0.8 metros con el balón a distancia d, el error de rumbo permitido es arcsin(0.8/d). A 10 metros son 4.6
grados, a 30 metros 1.5. Los sectores finos cerca del frente no son decoración. El borde en 17.5 grados es
la mitad del giro: es el error máximo que queda después de girar. Y cada zona de distancia equivale a 3 a
5 pasos del presupuesto de 40, o sea que la rejilla mide tiempo restante. Mencionar que el kit usaba
cortes en 0.8, 3 y 8 metros con sectores de 15 y 60 grados, mucho más gruesos.

## Slide 6 · Algoritmo (55 s)

La comparación es justa porque ambas ramas terminan en el mismo epsilon: 0.1. Lo único que cambia es el
calendario. El decaimiento empieza en 1.0, o sea explora al azar al principio, y llega a 0.1 al 60 por
ciento del presupuesto, con d = 0.999952 por episodio. Resaltar que los tres algoritmos que probamos
(Q-Learning, SARSA y Monte Carlo first-visit) empataron dentro del ruido entre semillas, así que la
diferencia no está en el algoritmo y por eso reportamos Q-Learning. Si preguntan por el alpha: 0.1 fijo
porque el estado es tabular y cada visita a un bin es una muestra ruidosa; no tuneamos alpha.

## Slide 7 · Diseño experimental (60 s)

Explicar el protocolo como un contrato de medición: se entrena en [5, 40] metros, se evalúa greedy cada
4 000 episodios sobre los mismos 500 inicios fijos, sin aprendizaje, para que las curvas sean comparables
entre corridas. La fidelidad al kit es un control: reimplementamos su notebook y su entorno y reproducimos
su resultado, con mucha varianza entre semillas (89.2 con desviación 17.2), por eso jamás reportamos una
sola semilla. El óptimo exacto es el punto fuerte del análisis: como el kit es determinista, calculamos
con búsqueda A* cuántos pasos mínimo necesita cada inicio, con heurística admisible. Ese número, 87.9,
no es una estimación de un algoritmo, es el techo demostrado del entorno. Decirlo así porque cambia cómo
se lee todo lo que sigue.

## Preguntas que me pueden tocar

- ¿Por qué Q-Learning y no DQN? P1 pide tabular y los tres algoritmos empataron dentro del ruido.
- ¿Por qué gamma 0.99? El bono de captura con 0.9 se evapora al paso 39 (1.6 contra 67.6).
- ¿Por qué solo observación total y no un POMDP? Porque el kit entrega distancia y rumbo verdaderos;
  la visión parcial aparece recién en el servidor y se trata en la parte de Dimael.
- ¿Por qué 40 pasos y no más? Es el presupuesto del enunciado; cambiarlo cambiaría el criterio.

Se lo paso a Dimael con: "ahora los resultados y el análisis".