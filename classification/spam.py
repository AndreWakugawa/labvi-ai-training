import pandas as pd
import kagglehub
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.ensemble import RandomForestClassifier
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Download do dataset do Kaggle
path = kagglehub.dataset_download("somesh24/spambase")

# Encontrar o arquivo CSV dentro do diretório baixado
csv_file = [f for f in os.listdir(path) if f.endswith(".csv")][0]


df = pd.read_csv(os.path.join(path, csv_file), header=None, skiprows=1) # Skip da primeira linha já que não tem header

# Renomear a última coluna para 'spam'
df.rename(columns={57: 'spam'}, inplace=True)

# Converter todas as colunas para numéricas. A coluna 'spam' deve ser int.
for col in df.columns:
    if col == 'spam':
        df[col] = df[col].astype(int)
    else:
        df[col] = pd.to_numeric(df[col], errors='coerce')

# Exibir as primeiras linhas e informações do dataset
print(df.head())
print(df.info())

# Verificar valores ausentes
print(df.isnull().sum().sum())

# Exibir estatísticas descritivas para colunas numéricas
print(df.describe())

# Preparação dos dados para Machine Learning
# Separar features (X) e target (y)
X = df.drop("spam", axis=1)
y = df["spam"]

# Dividir os dados em conjuntos de treino, validação e teste
# 20% para teste, e o restante para treino e validação
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.25, random_state=42, stratify=y_train) # 0.25 * 0.8 = 0.2 do dataset original para validação

print(f'\nTamanho do conjunto de treino: {len(X_train)} registros')
print(f'Tamanho do conjunto de validação: {len(X_val)} registros')
print(f'Tamanho do conjunto de teste: {len(X_test)} registros')

# Escalar features numéricas usando MinMaxScaler (para 0-1)
scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(X_test)

# Treinamento e Otimização do Modelo (RandomForestClassifier com GridSearchCV)
param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [None, 5, 10],
    'min_samples_leaf': [1, 2],
    'min_samples_split': [2, 5]
}

grid_search = GridSearchCV(RandomForestClassifier(random_state=42), param_grid, cv=3, scoring='accuracy', n_jobs=-1)
grid_search.fit(X_train_scaled, y_train)

best_model = grid_search.best_estimator_

print(f'\nMelhores parâmetros encontrados: {grid_search.best_params_}')
print(f'Melhor acurácia no conjunto de treino (com validação cruzada): {grid_search.best_score_:.2f}')

# Avaliação do Modelo no conjunto de teste
y_pred = best_model.predict(X_test_scaled)

print('\nRelatório de Classificação no conjunto de TESTE:')
print(classification_report(y_test, y_pred))

# Matriz de Confusão
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Não Spam', 'Spam'], yticklabels=['Não Spam', 'Spam'])
plt.xlabel('Previsão')
plt.ylabel('Verdadeiro')
plt.title('Matriz de Confusão')
plt.savefig('confusion_matrix.png')
# plt.show() # Não usar plt.show() em ambiente headless
