# Guion de exposición P1 — Ball Pursuit (8 minutos)

Ideas para decir en voz alta. Nada de esto va en las slides.

## 1. Portada (30 s)

Somos el grupo 1. El repositorio está público y todo lo que muestro sale de corridas guardadas, no de figuras dibujadas a mano.

## 2. La tarea (1 min)

El balón no se mueve, así que el problema es puro control de navegación con observación parcial del rumbo. Cada decisión equivale a un ciclo del servidor, 100 ms. El presupuesto de 40 pasos son 4 segundos reales. Mencionar que el enunciado fija el rango de distancia pero no su distribución, porque ese detalle decide todo el resultado.

## 3. Hallazgo central (1 min)

El 87.9 por ciento sale de una búsqueda A con física exacta, no de aprendizaje. Es un techo demostrado, no una estimación. Contar que con el reset del kit estamos a medio paso del óptimo, 19.1 contra 18.6. Si preguntan por el 90, la respuesta es que en rango completo es inalcanzable para cualquier política.

## 4. Exploración (1 min)

Ambas ramas terminan con el mismo nivel final de ruido, así que la comparación es justa. El decaimiento no gana por el valor final sino por el calendario, mucho ruido al inicio y casi nada al final. El ahorro es de miles de episodios, que en el servidor real son horas.

## 5. Entrenamiento (30 s)

Las curvas de entrenamiento llevan exploración encima, por eso van por debajo de la evaluación greedy. La media móvil es de 500 episodios. No leer los números, ya están en la figura.

## 6. Política (1 min)

La ventana de avance es asimétrica por el signo del giro, menos 35 contra más 17.5. Las rejillas gruesas sin borde en 35 grados no pueden representarla, y esa es la mejora de R3. La corrección fina no la hace un estado especial sino la geometría, al acercarse el rumbo crece y cruza el umbral.

## 7. Discretización (30 s)

R4 con 150 estados rindió menos porque el presupuesto de 80 mil episodios no alcanza para llenar la tabla. Más estados no es gratis.

## 8. Servidor real (1 min)

La caída de 100 a 26 por ciento es física, no conexión, porque nuestro simulador con física del servidor da lo mismo. Cero comandos perdidos en 200 episodios y prueba de 500 seguidos sin errores de socket. Si preguntan por qué no entrenar siempre en el servidor, responder tiempo, cada corrida en vivo toma horas.

## 9. Cierre (30 s)

Cerrar con las tres cifras y la propuesta de P2. No abrir temas nuevos en las preguntas, derivar a la sección de sensibilidad del informe.

## Preguntas probables

Por qué Q-Learning y no DQN. Porque P1 pide tabular y los tres algoritmos empataron dentro del ruido. Por qué gamma 0.99. Porque con 0.9 el bono de captura se evapora al paso 39. Por qué no R4. Por falta de datos, no por teoría.
