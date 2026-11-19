# Preguntas para la defensa

**¿Por qué H0 supera a los agentes?**  
En este SKU, simulador y presupuesto, la heurística es una referencia fuerte. Los resultados no identifican una causa única. Aproximación, exploración, función de recompensa y dificultad de optimización son hipótesis, no conclusiones causales.

**¿Cuál es el mejor agente?**  
Q-learning tiene el mayor retorno medio de test entre los agentes entrenados. PPO fue elegido por validación y presenta el mayor fill rate y menor dispersión del retorno. Debe especificarse la métrica y conservar la decisión tomada antes de test.

**¿Se alcanzó la reducción de quiebres del 30%?**  
No se alcanzó la meta conjunta en ninguna de las 24 corridas. Se informa el resultado negativo y se distingue la meta del desempeño observado.

**¿La elasticidad fue estimada con M5?**  
No. Se fijó en 1.5 como supuesto del simulador. La estimación requerida sigue siendo una limitación; no puede sustituirse retrospectivamente sin cambiar el experimento.

**¿Cómo puede PPO mantener acciones constantes y pedidos variables?**  
La acción selecciona un multiplicador de precio y un objetivo de cobertura. La cantidad pedida se calcula a partir del pronóstico, inventario y tránsito. Un índice constante no implica una cantidad constante. Tampoco permite afirmar cambios diarios aprendidos del multiplicador.

**¿Qué preserva PBRS?**  
Con el mismo descuento y potencial terminal cero, la suma descontada del shaping es menos el potencial inicial. Para un mismo estado inicial, añade una constante independiente de la trayectoria, preservando el orden de retornos descontados. No garantiza aprender más rápido ni obtener mejor retorno finito.

**¿Se demostró reward hacking en M5?**  
Se construyó un contraejemplo sintético donde el bono ingenuo cambia la preferencia hacia acaparar y PBRS la conserva. No se estableció que N0 reproduzca esa conducta en las trayectorias de M5.

**¿Es Dyna un world model completo?**  
Es híbrido: aprende demanda y usa ecuaciones conocidas para inventario, reparto y recompensa. Genera experiencia sintética para actualizar DDQN. No es Dreamer ni un modelo latente completo del entorno.

**¿Por qué menor MAE no mejora el retorno?**  
MAE no mide la utilidad de las decisiones. Además, RMSE empeora, los errores se evalúan en distribuciones inducidas por cada política y el aprendizaje de valor tiene sus propias dificultades. No hay evidencia suficiente para atribuir el resultado a una única causa.

**¿Tres semillas bastan para afirmar superioridad?**  
Permiten describir variación de optimización, con alcance limitado. No equivalen a tres muestras temporales independientes. Se reportan medias, dispersión y contrastes emparejados sin afirmar significancia no demostrada.

**¿Son beneficios y quiebres históricos reales?**  
No. Son proxies calculados en el simulador. M5 aporta ventas, precios y calendario; inventarios, costes y respuesta contrafactual al precio requieren supuestos.

**¿Puede reproducirse todo con este ZIP?**  
Se pueden revisar las trazas, tablas, figuras y demo. Para repetir el entrenamiento hacen falta el código fijado, dependencias y datos M5. Los checkpoints están fuera de este ZIP. No se presenta la ejecución completa de Docker como validada.
