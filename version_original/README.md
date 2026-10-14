# Versión 1 (original)

Script original del proyecto, conservado tal como se publicó para que se pueda comparar con la versión 2.

Lo que la versión 2 corrige:
1. El escalado no cambiaba el resultado por `var_smoothing`, no por *underflow*.
2. Usaba notas del 2º semestre, cuando muchos desertores ya se habían ido.
3. Trataba códigos (carrera, ocupación de los padres) como números continuos.
4. En las conclusiones, "84,8 % de precisión" era en realidad la **exactitud** (la precisión era 77,1 %).

La matriz de confusión original está en `matriz_confusion_desercion.png`.

Ver el notebook en `../notebooks/desercion_estudiantil_naive_bayes_v2.ipynb`.
