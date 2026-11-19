# Informe de la lista de ejercicios 1

El archivo `informe_lista_1.tex` es autónomo y contiene las respuestas de la lista de DS5345, 2026-II.

- Ejercicio 1: exactamente 40 fuentes con enlaces, agrupadas por tema.
- Ejercicios 2 al 14: desarrollo matemático y explicaciones en español.
- Se evitó el punto y coma en todo el texto y en el código LaTeX.
- El ejercicio 4 distingue el retorno total sin descuento del criterio de recompensa promedio, porque la figura no tiene estados terminales.

Para compilar, ejecuta dos veces desde esta carpeta:

```text
pdflatex -interaction=nonstopmode -halt-on-error informe_lista_1.tex
pdflatex -interaction=nonstopmode -halt-on-error informe_lista_1.tex
```

También puedes subir únicamente el archivo `.tex` a Overleaf. No necesita imágenes, archivos de bibliografía ni datos externos para compilar.
