# Predicción de Deserción Estudiantil (Naive Bayes)

Modelo de clasificación que predice si un estudiante **desertará o no** de la educación superior, a partir de variables conocidas al momento de la matrícula y del rendimiento del primer y segundo semestre.

Proyecto basado en la metodología del **Google Advanced Data Analytics Professional Certificate** (Coursera), replicando el enfoque usado para predecir churn bancario con Naive Bayes, aplicado a un caso de **retención estudiantil en educación superior**.

## Dataset

**Predict Students' Dropout and Academic Success**
Fuente: https://www.kaggle.com/datasets/thedevastator/higher-education-predictors-of-student-retention
4,424 estudiantes, 36 variables demográficas, socioeconómicas y académicas de una institución de educación superior portuguesa.

> Realinho, V., Machado, J., Baptista, L., & Martins, M.V. (2021). *Predict students' dropout and academic success* [Data set]. Zenodo. https://doi.org/10.5281/zenodo.5777340

## Metodología

### 1. Variable objetivo

El dataset original tiene 3 categorías (`Dropout`, `Enrolled`, `Graduate`). Se reformuló como problema **binario** — ¿el estudiante desertará (1) o no (0)? — análogo al caso de churn bancario, y más accionable para un equipo de retención.

Balance de clases: **32% desertó / 68% no desertó** — desbalance moderado, no extremo.

### 2. Ingeniería de features y variables dependientes

Se construyeron dos tasas de aprobación:

```
approval_rate_1er_sem = aprobadas_1er_sem / inscritas_1er_sem
approval_rate_2do_sem = aprobadas_2do_sem / inscritas_2do_sem
```

Como Naive Bayes asume independencia condicional entre variables, se eliminaron las columnas originales (`aprobadas`, `inscritas`) usadas para construir estas tasas, evitando dependencia directa entre features — mismo criterio aplicado en el taller original al eliminar `Tenure` y `Age` tras crear `Loyalty`.

### 3. Modelo

`GaussianNB` de scikit-learn, con `train_test_split` estratificado (`stratify=y`), comparando resultados **sin escalar** vs. con `MinMaxScaler`.

## Resultados

| Métrica | Sin escalar | Con MinMaxScaler |
|---|---|---|
| Accuracy | 0.848 | 0.848 |
| Precision | 0.771 | 0.771 |
| Recall | 0.749 | 0.749 |
| F1 Score | 0.760 | 0.760 |

**Hallazgo relevante:** a diferencia del ejemplo bancario del curso original (donde escalar era indispensable porque una variable como `Balance` alcanzaba 250,000+, generando *underflow* numérico en `GaussianNB`), en este dataset **ninguna variable tiene una escala tan extrema**, por lo que escalar no cambia el resultado. Esto demuestra que escalar es buena práctica preventiva, pero conviene diagnosticar antes de asumir que es la causa de un bajo desempeño.

### Matriz de confusión

| | Predicho: No Deserción | Predicho: Deserción |
|---|---|---|
| **Real: No Deserción** | 672 | 79 |
| **Real: Deserción** | 89 | 266 |

## Conclusiones

1. El modelo alcanza **84.8% de accuracy** y **74.9% de recall** para detectar deserción — un desempeño sólido para un modelo simple y rápido de entrenar.
2. El recall de ~75% significa que el modelo detecta **3 de cada 4** estudiantes que efectivamente desertarán, dejando ~25% de falsos negativos: casos que un equipo de retención no alcanzaría a priorizar solo con este modelo.
3. **Aplicación práctica:** un modelo como este podría integrarse a un sistema de alerta temprana que priorice el acompañamiento de estudiantes con alta probabilidad de deserción — un enfoque directamente relacionado con procesos de retención académica (matrículas, certificaciones, seguimiento estudiantil) en educación superior.

## Cómo ejecutar

```bash
pip install pandas numpy scikit-learn matplotlib
python desercion_estudiantil_naive_bayes.py
```

(Descargar el CSV de Kaggle y ajustar `DATA_PATH` en el script.)

## Stack

Python · pandas · scikit-learn · matplotlib
