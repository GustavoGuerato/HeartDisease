# 8  3. Evaluation
# 9  4. Features
# 10 5. Modelling
# 11 6. Experimentation
# 12
# 13 ## 1. Problem Definition
# 14
# 15 In a statement,
# 16
# Given clinical parameters about a patient, can we predict whether or not
# they have heart
# disease?
# 17
# 18 ## 2. Data
# 19
# 20 The original data came from the Cleavland data from the UCI
# Machine Learning Repository.
# https://archive.ics.uci.edu/ml/datasets/heart+Disease
# 21
# 22 There is also a version of it available on Kaggle.
# https://www.kaggle.com/ronitf/heart-
# disease-uci
# 23
# 24 ## 3. Evaluation
# 25
# 26 > If we can reach 95% accuracy at predicting whether or not a patient has
# heart disease
# during the proof of concept, we'll pursue the project.

# 27 ## 4. Features
# Data Dictionary
# age: Age of the patient (in years)
# sex: Sex of the patient (0 = female, 1 = male)
# cp (chest pain type): Type of chest pain
#     0: No chest pain
#     1: Typical angina
#     2: Atypical angina
#     3: Non-anginal pain
#     4: Asymptomatic
# trestbps (resting blood pressure): Resting blood pressure (mm Hg)
# chol (serum cholesterol): Serum cholesterol level (mg/dL)
# fbs (fasting blood sugar): Fasting blood sugar level
#     1 = >120 mg/dL, 0 = ≤120 mg/dL
# restecg (resting electrocardiographic results):
#     Resting electrocardiogram results
#     0: Normal
#     1: ST-T wave abnormality
#     2: Probable left ventricular hypertrophy
# thalach (maximum heart rate achieved): Maximum heart rate achieved
# exang (exercise induced angina): Exercise induced angina (1 = yes, 0 = no)
# oldpeak: Depression of the ST segment induced by exercise relative to rest
# slope: Slope of the ST segment during peak exercise
#     0: Upsloping
#     1: Flat
#     2: Downsloping
# ca (number of major vessels colored by fluoroscopy):
#     Number of major vessels colored by fluoroscopy (0 to 3)
# thal: Thalassemia type
#     1: Normal
#     2: Fixed defect
#     3: Reversable defect
# target: Indicates if the person has heart disease
#         (1 = disease, 0 = no disease)

# We will Use the following libraries to data analysis and manipulation

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.metrics import RocCurveDisplay

# Carregar dados
df = pd.read_csv("heart-disease.csv")
print(df.shape)
print(df.head())
print(df["target"].value_counts())
df["target"].value_counts().plot(kind="bar", color=["salmon", "lightblue"])
plt.show()
print(df.info())

# Verificar valores nulos
print(df.isna().sum())

# Visualizar as primeiras linhas
print(df.describe())
print(df.sex.value_counts())
print(pd.crosstab(df.target, df.sex))

# Visualizar gráfico de distribuição de sexo e presença de doença cardíaca
pd.crosstab(df.target, df.sex).plot(
    kind="bar", figsize=(10, 6), color=["salmon", "lightblue"]
)
plt.title("Heart Disease Frequency for sex")
plt.xlabel("0 = No Disease, 1 = Disease")
plt.ylabel("Amount")
plt.legend(["Female", "Male"])
plt.xticks(rotation=0)
plt.show()

# Análise de idade vs taxa máxima de batimento cardíaco
plt.figure(figsize=(10, 6))
plt.scatter(df.age[df.target == 1], df.thalach[df.target == 1], c="salmon")
plt.scatter(df.age[df.target == 0], df.thalach[df.target == 0], c="lightblue")
plt.title("Heart Disease in function of Age and Max Heart Rate")
plt.xlabel("Age")
plt.ylabel("Max Heart Rate")
plt.legend(["Disease", "No Disease"])
plt.show()

# Visualizar a distribuição de idade
df.age.plot.hist()
plt.show()

# Análise de tipo de dor no peito
print(pd.crosstab(df.cp, df.target))
pd.crosstab(df.cp, df.target).plot(
    kind="bar", figsize=(10, 6), color=["salmon", "lightblue"]
)
plt.title("Heart Disease Frequency Per Chest Pain Type")
plt.xlabel("Chest Pain Type")
plt.ylabel("Amount")
plt.legend(["No Disease", "Disease"])
plt.xticks(rotation=0)
plt.show()

# Correlação entre variáveis
print(df.corr())
corr_matrix = df.corr()
fix, ax = plt.subplots(figsize=(15, 10))
ax = sns.heatmap(corr_matrix, annot=True, linewidths=0.5, fmt=".2f", cmap="YlGnBu")
bottom, top = ax.get_ylim()
ax.set_ylim(bottom + 0.5, top - 0.5)
plt.show()

# Definir as variáveis de entrada (X) e saída (y)
X = df.drop("target", axis=1)
y = df["target"]

# Dividir os dados em treino e teste
np.random.seed(42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Modelos a serem testados
models = {
    "Logistic Regression": LogisticRegression(),
    "KNN": KNeighborsClassifier(),
    "Random Forest": RandomForestClassifier(),
}


# Função para treinar e avaliar os modelos
def fit_and_score(models, X_train, X_test, y_train, y_test):
    np.random.seed(42)
    models_scores = {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        models_scores[name] = model.score(X_test, y_test)
    return models_scores


# Avaliar os modelos
models_scores = fit_and_score(
    models=models, X_train=X_train, X_test=X_test, y_train=y_train, y_test=y_test
)

# Mostrar a comparação entre os modelos
model_compare = pd.DataFrame(models_scores, index=["accuracy"])
model_compare.T.plot.bar()
plt.show()

# KNN - Testando diferentes valores de K
train_scores = []
test_scores = []
neighbors = range(1, 21)

knn = KNeighborsClassifier()

for i in neighbors:
    knn.set_params(n_neighbors=i)
    knn.fit(X_train, y_train)

    train_scores.append(knn.score(X_train, y_train))
    test_scores.append(knn.score(X_test, y_test))

plt.plot(neighbors, train_scores, label="Train score")
plt.plot(neighbors, test_scores, label="Test score")
plt.xticks(np.arange(1, 21, 1))
plt.xlabel("Number of neighbors")
plt.ylabel("Model score")
plt.legend()
plt.show()

print(f"maximum KNN score on the test data: {max(test_scores)*100:.2f}%")

# Realizando a busca por hiperparâmetros para Logística e Random Forest
log_reg_grid = {"C": np.logspace(-4, 4, 20), "solver": ["liblinear"]}
rf_grid = {
    "n_estimators": np.arange(10, 1000, 50),
    "max_depth": [None, 3, 5, 10],
    "min_samples_split": np.arange(2, 20, 2),
}

# RandomizedSearchCV para Logística
np.random.seed(42)
rs_log_reg = RandomizedSearchCV(
    LogisticRegression(),
    param_distributions=log_reg_grid,
    cv=5,
    n_iter=20,
    verbose=True,
)
rs_log_reg.fit(X_train, y_train)
print(rs_log_reg.best_params_)
rs_log_reg.score(X_test, y_test)

# RandomizedSearchCV para Random Forest
rs_rf = RandomizedSearchCV(
    RandomForestClassifier(), param_distributions=rf_grid, cv=5, n_iter=20, verbose=True
)
rs_rf.fit(X_train, y_train)
print(rs_rf.best_params_)
rs_rf.score(X_test, y_test)

# GridSearchCV para Logística
log_reg_grid = {"C": np.logspace(-4, 4, 30), "solver": ["liblinear"]}
gs_log_reg = GridSearchCV(
    LogisticRegression(), param_grid=log_reg_grid, cv=5, verbose=True
)
gs_log_reg.fit(X_train, y_train)
print(gs_log_reg.best_params_)
gs_log_reg.score(X_test, y_test)

y_preds = gs_log_reg.predict(X_test)

# Gerar a curva ROC
RocCurveDisplay.from_estimator(gs_log_reg.best_estimator_, X_test, y_test)
plt.show()

# Mostrar a matriz de confusão
print(confusion_matrix(y_test, y_preds))

# Função para plotar a matriz de confusão
sns.set(font_scale=1.5)


def plot_conf_mat(y_test, y_preds):
    fig, ax = plt.subplots(figsize=(3, 3))
    ax = sns.heatmap(confusion_matrix(y_test, y_preds), annot=True, cbar=False)
    plt.xlabel("True label")
    plt.ylabel("Predicted label")


# Plotar a matriz de confusão
plot_conf_mat(y_test, y_preds)
plt.show()

print(classification_report(y_test, y_preds))

clf = LogisticRegression(C=0.20433597178569418, solver="liblinear")

cv_acc = cross_val_score(clf, X, y, cv=5, scoring="accuracy")
cv_acc = np.mean(cv_acc)

cv_precision = cross_val_score(clf, X, y, cv=5, scoring="precision")
cv_precision = np.mean(cv_precision)

cv_recall = cross_val_score(clf, X, y, cv=5, scoring="recall")
cv_recall = np.mean(cv_recall)

cv_f1 = cross_val_score(clf, X, y, cv=5, scoring="f1")
cv_f1 = np.mean(cv_f1)

cv_metrics = pd.DataFrame(
    {
        "Accuracy": cv_acc,
        "Precision": cv_precision,
        "Recall": cv_recall,
        "F1": cv_f1,
    },
    index=[0],
)
cv_metrics.T.plot.bar(title="Cross-validated classification metrics", legend=False)
# Fit an instance of LogisticRegression
clf = LogisticRegression(C=0.20433597178569418, solver="liblinear")

clf.fit(X_train, y_train)
# Check coef_
clf.coef_
df.head()

# Match coef's of features to columns
feature_dict = dict(zip(df.columns, list(clf.coef_[0])))
feature_dict

# Visualize feature importance
feature_df = pd.DataFrame(feature_dict, index=[0])
feature_df.T.plot.bar(title="Feature Importance", legend=False)

pd.crosstab(df["sex"], df["target"])

pd.crosstab(df["slope"], df["target"])
