# Guion de Dimael Rivas — slides 8 a 14 (≈ 6 minutos)

Lo que se dice en voz alta y no aparece en las slides. Tiempo por slide.

## Slide 8 · Ablación (60 s)

Abrir con la lectura de la tabla, no de la curva: las dos ramas terminan cerca (85.7 contra 84.3), la
diferencia real es el calendario. El decreciente llega al 70 por ciento en 12 000 episodios en todas las
semillas; el constante, en 16 000. Al 80 por ciento la brecha se abre: 12 000 a 16 000 contra 16 000 a
32 000. O sea, el decaimiento no gana por el valor final, gana por llegar antes y con menos varianza
(desviación 0.3 contra 1.3). En el servidor real esos miles de episodios de diferencia son horas de
computo. Y la fila en cursiva es el mensaje central: ninguna política, de ningún algoritmo, pasa de 87.9
en ese rango. La política queda en el 97.5 por ciento de ese techo.

## Slide 9 · Entrenamiento (35 s)

No leer los números de la curva, ya están escritos. Explicar sí por qué las curvas de entrenamiento
quedan por debajo de la evaluación greedy: siguen explorando con epsilon, entonces toman acciones
subóptimas a propósito. La media móvil es de 500 episodios. Si preguntan por los picos: son la varianza
de inicios lejanos, que solo se resuelven con captura.

## Slide 10 · Política (50 s)

La política aprendida cabe en una frase y conviene decirla despavilada: avanza con DASH 100 mientras el
balón esté entre −35 y +17.5 grados, y fuera de esa ventana gira hacia él. Detalle técnico que no está en
la slide: la ventana es asimétrica y más ancha que un giro completo de 35 grados. Por eso no oscila; tras
girar 35, el balón queda dentro de la ventana y el agente vuelve a avanzar. Una rejilla sin borde en 35
grados no puede siquiera representar esa ventana, y esa es la explicación compatible con los datos de por
qué R3 le va mejor. Decirlo como hipótesis medida, no como demostración, porque no aislamos el componente.

## Slide 11 · Trayectorias (35 s)

Sirve para mostrar el aprendizaje sin curvas: las líneas claras de los primeros episodios son erráticas y
casi rectas al final. La moraleja técnica es que no hay un estado de "corrección fina": la corrección
final la hace la geometría. El agente se acerca con un error chico, el rumbo del balón crece al avanzar,
cruza el umbral de 17.5 grados y recién ahí gira. La tabla no modela esa maniobra, la induce.

## Slide 12 · Por qué no mejores resultados (60 s)

Esta es la slide que responde la pregunta incómoda del informe. Tres causas, todas medidas. Primera: con
1 metro por paso, un balón a 35 o 40 metros se come casi todo el horizonte solo para llegar; el óptimo
captura apenas 18.7 por ciento de esos inicios. El 90 por ciento en [5, 40] es inalcanzable para cualquier
política, es límite de la tarea como la interpretamos, no del aprendizaje. Segunda: la brecha con el
óptimo está concentrada en 30 a 40 metros; hasta 30 capturamos el 100 por ciento igual que el óptimo.
Tercera: la física. Y con las barras cerrar el punto de la discretización: más estados no es gratis, R4
con 150 estados rindió menos que R3 porque el presupuesto de datos no alcanza para llenar la tabla. Notar
que ni siquiera supimos qué componente exacto de R3 explica la mejora; R1, que solo agregaba un sector de
±5 grados, no ayudó.

## Slide 13 · Transferencia (60 s)

Contar la cadena de razonamiento. La política del kit en vivo da 26 por ciento con el reset. Eso parece
un desastre, pero nuestro propio simulador con la física del servidor da 26.0 y 21.0, lo mismo: la caída
se debe a la física, no a la conexión ni al cliente. Por eso entrenamos de nuevo con esa física, mirando
solo lo que ve el jugador y con un bit de velocidad en el estado, y esa política llega a 73.5 por ciento
en vivo. La prueba de 500 episodios seguidos sin errores de socket descarta el problema de
comunicación. Si preguntan por qué no entrenar siempre en el servidor: cada corrida en vivo toma horas y
el kit corre en milisegundos.

Para la demostración abrir `workspace/demo_qlearning_rcssserver.ipynb` por el puerto 8888. La tabla es
Q-Learning con la física del servidor, semilla 4 de la evaluación en vivo del informe; no es la tabla Monte
Carlo del cuaderno del kit. Elegir el balón lateral (8, 9), pulsar Iniciar demo y luego recorrer los pasos
con Anterior y Siguiente. Una sola ejecución no sustituye el promedio de 200 episodios del informe.

## Slide 14 · Conclusiones (40 s)

Cerrar con las tres cifras y una sola frase por cada una. Después las otras tres tareas del catálogo están
formuladas pero no implementadas: dejamos el MDP con tablas de 216 a 1 296 valores, manejables con los
mismos algoritmos de este informe. P2 en dos ideas: aproximadores continuos para evitar la tabla fina, y
entrenar directamente con la física del servidor. No abrir temas nuevos en las preguntas; si algo se sale
del informe, remitir a la sección de sensibilidad del PDF.

## Preguntas que me pueden tocar

- ¿Por qué el resultado en vivo es tan bajo? Física del servidor: el avance crece gradualmente de 0.6 a
  1 metro, el giro se reduce con la velocidad, hay ruido y visión parcial. Coincide con el simulador.
- ¿Es truco medir con los mismos inicios fijos? No: son fijos solo para evaluar, el entrenamiento usa
  inicios nuevos, y todas las condiciones usan los mismos 500.
- ¿Por qué no refinan más la tabla? R4 (150 estados) ya cayó a 81 por ciento: faltan datos, no estados.
- ¿Sirve este trabajo para P2? Sí, la parte del servidor ya está probada; P2 agrega DQN/PPO.

Cierre en voz alta: "las tres cifras clave son 100 por ciento con el reset del kit, 85.7 por ciento en
rango completo contra un techo de 87.9, y 73.5 por ciento ya en el servidor real".
