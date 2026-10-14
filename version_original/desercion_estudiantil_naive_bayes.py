"""
Prediccion de Desercion Estudiantil con Naive Bayes
=====================================================
Autor: Cristobal Rojas Troncoso
Basado en la metodologia del Google Advanced Data Analytics Professional
Certificate (Naive Bayes para prediccion de churn bancario), aplicada a
un caso de desercion estudiantil en educacion superior.

Dataset: Predict Students' Dropout and Academic Success (UCI / Kaggle)
Fuente: https://www.kaggle.com/datasets/thedevastator/higher-education-predictors-of-student-retention
Realinho, V., Machado, J., Baptista, L., & Martins, M.V. (2021).
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, ConfusionMatrixDisplay
)

# ---------------------------------------------------------------------------
# 1. Carga de datos
# ---------------------------------------------------------------------------
DATA_PATH = "dataset.csv"  # ajustar segun ubicacion local
df = pd.read_csv(DATA_PATH)

print(f"Dataset cargado: {df.shape[0]:,} estudiantes, {df.shape[1]} columnas")
print("\nDistribucion original de Target (3 categorias):")
print(df['Target'].value_counts())

# ---------------------------------------------------------------------------
# 2. Variable objetivo: Desercion (binaria)
# ---------------------------------------------------------------------------
# Se reformula el problema original de 3 clases (Dropout / Enrolled / Graduate)
# como una prediccion binaria de desercion, analoga al caso de churn bancario:
# ¿el estudiante desertara (1) o no (0)? Esto es mas accionable para un equipo
# de retencion academica.
df['Dropout_flag'] = (df['Target'] == 'Dropout').astype(int)

print("\nBalance de clases (Dropout_flag):")
print(df['Dropout_flag'].value_counts(normalize=True) * 100)
# 32% desercion / 68% no desercion -> desbalance moderado, no extremo

# ---------------------------------------------------------------------------
# 3. Ingenieria de features y eliminacion de variables dependientes
# ---------------------------------------------------------------------------
# Igual que en el lab original (donde 'Loyalty' se construyo como
# Tenure/Age y luego se eliminaron Tenure y Age para no violar el supuesto
# de independencia condicional de Naive Bayes), aqui construimos tasas de
# aprobacion y eliminamos las columnas crudas que las componen.
def safe_ratio(approved, enrolled):
    return np.where(enrolled > 0, approved / enrolled, 0)

df['approval_rate_1st'] = safe_ratio(
    df['Curricular units 1st sem (approved)'],
    df['Curricular units 1st sem (enrolled)']
)
df['approval_rate_2nd'] = safe_ratio(
    df['Curricular units 2nd sem (approved)'],
    df['Curricular units 2nd sem (enrolled)']
)

cols_to_drop = [
    'Curricular units 1st sem (approved)', 'Curricular units 1st sem (enrolled)',
    'Curricular units 2nd sem (approved)', 'Curricular units 2nd sem (enrolled)',
    'Target'
]
df_model = df.drop(columns=cols_to_drop)

# ---------------------------------------------------------------------------
# 4. Split de datos (con estratificacion, igual que en el lab)
# ---------------------------------------------------------------------------
y = df_model['Dropout_flag']
X = df_model.drop(columns=['Dropout_flag'])

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, stratify=y, random_state=42
)

# ---------------------------------------------------------------------------
# 5. Modelo 1: Naive Bayes sin escalar
# ---------------------------------------------------------------------------
gnb = GaussianNB()
gnb.fit(X_train, y_train)
preds = gnb.predict(X_test)

print("\n=== Naive Bayes (sin escalar) ===")
print('Accuracy:', '%.3f' % accuracy_score(y_test, preds))
print('Precision:', '%.3f' % precision_score(y_test, preds))
print('Recall:', '%.3f' % recall_score(y_test, preds))
print('F1 Score:', '%.3f' % f1_score(y_test, preds))

# ---------------------------------------------------------------------------
# 6. Modelo 2: Naive Bayes con MinMaxScaler
# ---------------------------------------------------------------------------
scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

gnb_scaled = GaussianNB()
gnb_scaled.fit(X_train_scaled, y_train)
preds_scaled = gnb_scaled.predict(X_test_scaled)

print("\n=== Naive Bayes (con MinMaxScaler) ===")
print('Accuracy:', '%.3f' % accuracy_score(y_test, preds_scaled))
print('Precision:', '%.3f' % precision_score(y_test, preds_scaled))
print('Recall:', '%.3f' % recall_score(y_test, preds_scaled))
print('F1 Score:', '%.3f' % f1_score(y_test, preds_scaled))

# Nota: a diferencia del dataset del lab original (donde una variable
# como 'Balance' llegaba a 250,000+ y provocaba underflow numerico en
# GaussianNB, forzando al modelo a predecir siempre la clase mayoritaria),
# en este dataset ninguna variable tiene una escala tan extrema, por lo
# que escalar no cambia el resultado. Esto ilustra que escalar es buena
# practica preventiva, pero conviene diagnosticar antes de asumir que es
# la causa de un mal desempeño.

# ---------------------------------------------------------------------------
# 7. Matriz de confusion
# ---------------------------------------------------------------------------
cm = confusion_matrix(y_test, preds_scaled, labels=[0, 1])
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['No Desercion', 'Desercion'])
disp.plot(values_format='', cmap='Blues')
plt.title('Matriz de confusion - Prediccion de Desercion Estudiantil')
plt.tight_layout()
plt.savefig('matriz_confusion_desercion.png', dpi=150)
plt.show()

# ---------------------------------------------------------------------------
# 8. Conclusiones
# ---------------------------------------------------------------------------
print("""
CONCLUSIONES
============
1. El modelo Naive Bayes alcanza 84.8% de accuracy, 77.1% de precision y
   74.9% de recall para detectar desercion, un desempeño solido para un
   modelo simple y rapido de entrenar.
2. El recall de ~75% significa que el modelo detecta 3 de cada 4
   estudiantes que efectivamente desertaran, dejando ~25% de falsos
   negativos -> casos que un equipo de retencion no alcanzaria a priorizar
   solo con este modelo.
3. Escalar los datos no cambio el resultado en este caso, a diferencia
   del ejemplo bancario del curso -> el escalado importa mas cuando hay
   variables con escalas extremadamente dispares.
4. Aplicacion practica: un modelo como este podria integrarse a un
   sistema de alerta temprana (tutoring team) que priorice el
   acompañamiento de estudiantes con alta probabilidad de desercion,
   similar al enfoque de retencion aplicado en procesos academicos de
   Iplacex (matriculas, certificaciones, gestion documental).
""")
