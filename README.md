# Heart Disease Prediction Using Machine Learning

Este projeto tem como objetivo prever se um paciente tem doença cardíaca com base em vários parâmetros clínicos, utilizando modelos de aprendizado de máquina.

## 1. Problem Definition

Dado os parâmetros clínicos sobre um paciente, o objetivo é prever se ele tem ou não doença cardíaca.

## 2. Data

O conjunto de dados utilizado neste projeto é o Cleveland Heart Disease Dataset, disponível no repositório UCI Machine Learning.

- [UCI Heart Disease Dataset](https://archive.ics.uci.edu/ml/datasets/heart+Disease)
- [Kaggle Heart Disease Dataset](https://www.kaggle.com/ronitf/heart-disease-uci)

### Data Dictionary:

- **age**: Idade do paciente (em anos)
- **sex**: Sexo do paciente (0 = feminino, 1 = masculino)
- **cp (tipo de dor no peito)**: Tipo de dor no peito:
  - 0: Sem dor no peito
  - 1: Angina típica
  - 2: Angina atípica
  - 3: Dor não-anginal
  - 4: Assintomático
- **trestbps (pressão arterial em repouso)**: Pressão arterial em repouso (mm Hg)
- **chol (colesterol sérico)**: Nível de colesterol sérico (mg/dL)
- **fbs (glicemia em jejum)**: Nível de glicemia em jejum (1 = >120 mg/dL, 0 = ≤120 mg/dL)
- **restecg (resultados do eletrocardiograma em repouso)**: Resultados do eletrocardiograma em repouso
  - 0: Normal
  - 1: Anormalidade da onda ST-T
  - 2: Hipertrofia ventricular esquerda provável
- **thalach (taxa máxima de batimento cardíaco atingida)**: Taxa máxima de batimento cardíaco atingida
- **exang (angina induzida por exercício)**: Angina induzida por exercício (1 = sim, 0 = não)
- **oldpeak**: Depressão do segmento ST induzida por exercício em comparação com o repouso
- **slope**: Inclinação do segmento ST durante o exercício de pico
  - 0: Ascendente
  - 1: Plano
  - 2: Descendente
- **ca (número de vasos principais coloridos por fluoroscopia)**: Número de vasos principais coloridos por fluoroscopia (0 a 3)
- **thal**: Tipo de talassemia
  - 1: Normal
  - 2: Defeito fixo
  - 3: Defeito reversível
- **target**: Indica se a pessoa tem doença cardíaca (1 = doença, 0 = sem doença)

## 3. Evaluation

O objetivo do projeto é atingir **95% de precisão** na previsão de doença cardíaca durante a fase de prova de conceito.

## 4. Features

Vários parâmetros clínicos são usados para prever a presença de doença cardíaca, como idade, sexo, níveis de colesterol, taxa máxima de batimento cardíaco e tipo de dor no peito.

### Exploratory Data Analysis:

- **Distribuição de Gênero**: Analisou-se a frequência de doença cardíaca entre homens e mulheres.
- **Idade vs Taxa Máxima de Batimento Cardíaco**: Exploração da relação entre a idade do paciente e a taxa máxima de batimento cardíaco, colorida pelo status de doença.
- **Tipo de Dor no Peito**: Analisada a frequência de doença cardíaca para cada tipo de dor no peito.
- **Matriz de Correlação**: Visualização das correlações entre as variáveis para entender quais fatores estão mais associados à doença cardíaca.

## 5. Modeling

### Modelos Usados:

- **Regressão Logística**
- **K-Nearest Neighbors (KNN)**
- **Random Forest**

### Passos:

1. **Pré-processamento dos Dados**: Os dados foram divididos em variáveis de entrada (X) e saída (y).
2. **Divisão em Treinamento e Teste**: Os dados foram divididos em conjuntos de treinamento (80%) e teste (20%).
3. **Avaliação de Modelos**: A precisão de cada modelo foi avaliada usando um conjunto de teste.
4. **Ajuste de Hiperparâmetros**:
   - **RandomizedSearchCV** foi usado para ajustar os hiperparâmetros para a Regressão Logística e Random Forest.
   - **GridSearchCV** foi utilizado para afinar a Regressão Logística.

### Resultados:

- A Regressão Logística com ajuste de hiperparâmetros alcançou a maior precisão e foi escolhida como o modelo final.
- O desempenho do modelo foi validado utilizando **validação cruzada** e métricas de avaliação como precisão, recall, F1 score e matriz de confusão.

### Métricas Importantes:

- **Acurácia**: Mede a porcentagem de previsões corretas.
- **Precisão**: Mede a proporção de previsões positivas que foram corretas.
- **Recall**: Mede a proporção de positivos reais que foram corretamente identificados.
- **F1 Score**: A média harmônica entre precisão e recall.

### Curva ROC:

- A Curva de Característica de Operação do Receptor (ROC) foi usada para avaliar o desempenho do modelo, com a Regressão Logística mostrando os melhores resultados.

## 6. Experimentation

Vários experimentos foram realizados para analisar como diferentes valores de **K** no modelo KNN afetaram a precisão. O melhor valor de **K** foi encontrado em 9.

## 7. Results

Após o ajuste de hiperparâmetros e a otimização do modelo, o modelo de Regressão Logística obteve uma precisão superior a 90% na previsão de doença cardíaca, o que é um bom resultado para esta tarefa.

### Modelo Final:
O modelo final é um classificador de Regressão Logística treinado no conjunto de dados completo, com hiperparâmetros otimizados.

### Importância das Características:
Foi gerado um gráfico de barras para mostrar a importância de cada característica, com variáveis como **colesterol**, **taxa máxima de batimento cardíaco** e **idade** sendo as mais influentes na previsão de doença cardíaca.

## 8. Conclusion

Este projeto previu com sucesso a doença cardíaca com base em dados clínicos, alcançando alta precisão com o modelo de Regressão Logística. Trabalhos futuros podem focar em melhorar o desempenho do modelo incluindo dados adicionais ou utilizando técnicas mais avançadas, como métodos de ensemble.

## Requirements

- `numpy`
- `pandas`
- `matplotlib`
- `seaborn`
- `scikit-learn`

### Installation:

```bash
pip install numpy pandas matplotlib seaborn scikit-learn
