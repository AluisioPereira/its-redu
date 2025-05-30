# -*- coding: utf-8 -*-
"""artigo-classificadores.ipynb


Original Aluisio Pereira

#### Importes
"""

!pip install sklearn_lvq
!pip3 install memory_profiler
!pip3 install optuna
!pip install sklvq
!pip install yellowbrick
!pip install imblearn

# Commented out IPython magic to ensure Python compatibility.
import numpy
import numpy as np
import pandas as pd
import sklvq
from sklvq import GLVQ
import itertools
import matplotlib.pyplot as plt
# %matplotlib inline
import time
import os
import sys
import requests
from sklearn import datasets
from sklearn.utils import Bunch
from memory_profiler import memory_usage
import optuna
from sklearn.metrics import accuracy_score
from sklearn import metrics

from numpy import mean
from numpy import std
from sklearn.datasets import make_classification
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import StackingClassifier
from sklearn.ensemble import AdaBoostClassifier
# Importando as bibliotecas necessárias
from sklearn.ensemble import VotingClassifier
from matplotlib import pyplot

from sklearn.datasets import make_classification
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB

# modelos
# knn
from sklearn.neighbors import KNeighborsClassifier
import matplotlib as mpl

# Máquinas de vetor de suporte
from sklearn import svm
from sklearn.svm import SVC
from sklearn import datasets

# para gerar os gráficos da árvore de decisão
from sklearn.tree import DecisionTreeClassifier, export_text, export_graphviz, plot_tree
try:
    from StringIO import StringIO ## para Python 2
except ImportError:
    from io import StringIO ## para Python 3
from IPython.display import Image
import pydotplus

# RNA
plt.style.use('seaborn-talk')
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import GridSearchCV, train_test_split, cross_val_score, StratifiedKFold
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import RepeatedKFold


# metricas e plots

from numpy.ma.extras import average
from sklearn.metrics import f1_score
from sklearn.model_selection import RepeatedKFold

from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split

from yellowbrick.datasets import load_concrete
from yellowbrick.regressor import ResidualsPlot

from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from matplotlib import pyplot as plt
import seaborn as sns
import seaborn as sn

"""# 1 - Análise Exploratória dos Dados
Realizou-se a importação dos dados e analisou-se inicialmente, verificando a distribuição, as classes existentes, frequência de ocorrência, correlação, e informações quantidade, para entender o conjunto de dados.
"""

# importando a base de dados - Interaçãoes dos Estudantes
!wget "https://github.com/AluisioPereira/artigo---ensemble/blob/main/padrao_interacao_ef_em.csv" -O "DADOS.csv" -

db = pd.read_csv("DADOS.csv",  sep=",", encoding = "ISO-8859-1", low_memory=False)

db.head()

df = pd.DataFrame(db)
df.head()

data = df.drop(['id-aluno', 'id-turma', 'serie', 'ensino', 'login-count', 'ano-cadastro-usuario', 'dias-ultimo-login', 'cluster'], axis=1, inplace=False)
data.head()

# separando os atributos em categóricos e contínuos
continuous = data.describe().columns
categorical = data.drop(list(continuous), axis=1).columns
print(continuous)
print(categorical)

# Commented out IPython magic to ensure Python compatibility.
'''
Adotou-se uma abordagem de regressão linear para verificar e completar dados faltantes.
Em que, variáveis ausentes serão preenchidas por valores previstos a partir do modelo (a partir dos valores mais fortemente correlacionados)
'''
import matplotlib.pyplot as plt
# %matplotlib inline

continuous_columns_missing_values = []
for column in continuous:
  if data[column].isnull().sum() > 0:
    continuous_columns_missing_values.append(column)
print(continuous_columns_missing_values)

most_correlated_columns = {}
candidates = [
  x for x in continuous if x not in continuous_columns_missing_values
]
for column in continuous_columns_missing_values:
  most_correlated_columns[column] = max(
      candidates, key=lambda x: abs(data[x].corr(data[column]))
  )

'''
Dados categóricos - para adequar os dados a algoritmos de Machine Learning se fez necessário converter variáveis qualitativas em quantitativas para permitir operações numéricas.
Sendo assim, as variáveis categóricas da base de dados foram convertidas em inteiros através do LabelEncoder da biblioteca sklear.
'''
categorical_columns_missing_values = [
  p[0] for p in dict(data[categorical].isna().sum() > 0).items() if p[1]
]
complete_data = data.dropna()
print(categorical_columns_missing_values)

# Convertendo dados categóricos para números inteiros usando o LabelEncoder da biblioteca Sklearn
from sklearn.preprocessing import LabelEncoder
from collections import defaultdict

label_dict = defaultdict(LabelEncoder)
complete_data = complete_data.apply(
    lambda x: label_dict[x.name].fit_transform(x)
    if x.name in list(categorical)
    else x
)

#reconstruindo sem os dados categoricos
labels = data['padrao']
data.drop('padrao', axis=1, inplace=True)
X = data.apply(
    lambda x: label_dict[x.name].fit_transform(x)
    if x.name in categorical
    else x
)
print(X)

# Codificando os rótulos
Y = label_dict['padrao'].fit_transform(labels)
target_names=['M. I.', 'I. E.', 'I. R.']
print(Y)

X.boxplot(vert=False)
plt.show()

data.boxplot(vert=False)
plt.show()

data.corr()

correlacao = X.corr() #var01 a var11
plot = sn.heatmap(correlacao, annot = True, fmt=".2f", linewidths=.6, mask= np.triu(correlacao))

X.describe()

correlacao = data.corr() #var01 a var10
plot = sn.heatmap(correlacao, annot = True, fmt=".2f", linewidths=.6, mask= np.triu(correlacao))

"""# 2 - Pré-processamento dos Dados
Pré-processou os dados para verificar a existência de dados faltantes, tipos de dados, realizarem o tratamento e transformar os dados para melhor ajuste aos modelos de aprendizagem de máquina.
"""

data.isna().sum()

data.describe()

data.info()

data.head()

"""# 3- Divisão da Base

Utilizou-se do k-fold, foi dividido a base em Treino e Teste.
"""

rkf = RepeatedKFold(n_splits=5, n_repeats=10)

xtrain, xtest, ytrain, ytest = train_test_split(X, Y, test_size=0.3, stratify=Y)

model = Ridge()
visualizer = ResidualsPlot(model, hist=False, qqplot=True)
visualizer.fit(xtrain, ytrain)
visualizer.score(xtest, ytest)
visualizer.show()

"""# 4 - Modelagem
A modelagem foi dividida em duas etapas:



1.   Avaliação prévia do Modelo de AM (KNN, SVM, Árvore de Decisão e/ou Redes Neurais Artificiais, escolher um dentre esses, acho que o melhor seria o de RNA) para verificar a necessidade de implementação de comitês para o dataset proposto.
2. Avaliação do comportamento do Modelo após a aplicação dos comitês bagging, boosting, stacking e voting para o dataset proposto.

##4.1 - Avaliação prévia

###  K-NN ([k-vizinhos mais próximos](https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html))
"""

knn = KNeighborsClassifier(n_neighbors=5)

predictions = []

for k in range(1, 13, 2):
  knn = KNeighborsClassifier(n_neighbors=k)
  knn.fit(xtrain, ytrain)

  x_pred = knn.predict(xtrain)
  print("\nk-NN para k = ",k)
  print("TREINO: \nAcurácia = ", metrics.accuracy_score(ytrain, x_pred))
  print(classification_report(ytrain, x_pred))

  y_pred = knn.predict(xtest)
  predictions.append(y_pred)
  print("TESTE \nAcurácia: ", metrics.accuracy_score(ytest, y_pred))
  print(classification_report(ytest, y_pred))

"""### SVM ([Máquinas de Vetores de Suporte](https://scikit-learn.org/stable/modules/svm.html))




"""

tuned_parameters = [
    {"kernel": ["linear"], "C": [1, 10, 100, 1000]},
    {"kernel": ["poly"], "C": [1, 10, 100, 1000]},
    {"kernel": ["rbf"], "gamma": [1e-3, 1e-4], "C": [1, 10, 100, 1000]},
    {"kernel": ["sigmoid"], "C": [1, 10, 100, 1000]},
]

clf = GridSearchCV(SVC(), tuned_parameters, cv=rkf)
clf.fit(xtrain, ytrain)
melhor_modelo = clf.best_params_
print("\n\n\n\nTREINO: \n")
means = clf.cv_results_["mean_test_score"]
stds = clf.cv_results_["std_test_score"]
for mean, std, params in zip(means, stds, clf.cv_results_["params"]):
    print("Média: %0.3f Score: (+/-%0.03f) com: %r" % (mean, std * 2, params))


print("\n\n\n\nTESTE - Relatório melhor estimador: ", melhor_modelo)
y_true, y_pred = ytest, clf.predict(xtest)
print(classification_report(y_true, y_pred, target_names=target_names))

# Polinomial kernel SVM
svc_pol = SVC(kernel='linear', C=100)

#TREINO
svc_pol.fit(xtrain, ytrain)

x_pred = svc_pol.predict(xtrain)
print("\nSVM - melhor modelo: ", melhor_modelo)
print("\nTREINO: \nAcurácia = ", metrics.accuracy_score(ytrain, x_pred))
print(classification_report(ytrain, x_pred))


#matriz de confusão - TREINO
ConfusionMatrixDisplay.from_estimator(svc_pol, xtrain, ytrain, display_labels=target_names)
plt.show()

# TESTE
y_pred = svc_pol.predict(xtest)
predictions.append(y_pred)
print("\nSVM - melhor modelo: ", melhor_modelo)
print("\nTESTE \nAcurácia: ", metrics.accuracy_score(ytest, y_pred))
print(classification_report(ytest, y_pred))


#matriz de confusão - TESTE
ConfusionMatrixDisplay.from_estimator(svc_pol, xtest, ytest, display_labels=target_names)
plt.show()

"""### DT ([Árvore de Decisão](https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeClassifier.html))"""

decision_tree = DecisionTreeClassifier()

param_grid = [{'criterion': ["gini","entropy"], 'max_depth': [2, 3, 5, 7, 10], 'min_samples_leaf': [2, 5, 10, 15, 20], 'min_samples_split' : [2,3,4,5,10,15]}]

scoring = {'Accuracy':'accuracy','F1':'f1_macro','Recall':'recall_macro','Precision':'precision_macro'}
grid_search = GridSearchCV(estimator=decision_tree, param_grid=param_grid, n_jobs=-1, cv=rkf, scoring=scoring, error_score=0, refit=False)

grid_result = grid_search.fit(xtrain, ytrain)

means_acc = grid_result.cv_results_['mean_test_Accuracy']
means_f1 = grid_result.cv_results_['mean_test_F1']
means_pre = grid_result.cv_results_['mean_test_Precision']
means_rec = grid_result.cv_results_['mean_test_Recall']
params = grid_result.cv_results_['params']

import csv
f = open('arvore.csv', 'w', newline='', encoding='utf-8')
w = csv.writer(f)
w.writerow(['accuracy', 'f1_macro', 'recall_macro', 'precision_macro', 'sum'])
for acc, f1, rec, pre, param in zip(means_acc, means_f1, means_rec, means_pre, params):
  if acc != 0:
    df2 = pd.DataFrame(params)
    w.writerow([acc, f1, rec, pre, (acc+f1+rec+pre)])

df = pd.read_csv("arvore.csv",  sep=",")
conc = pd.concat([df, df2], axis=1, join='inner')
print(conc)

gride_tree = GridSearchCV(decision_tree, param_grid, cv=rkf)

x_hat = gride_tree.fit(xtrain, ytrain)
estimador_tree, escore_tree = gride_tree.best_estimator_, gride_tree.best_score_
print("\n\nMelhor estimador: ",estimador_tree,"\nMelhor score: ", escore_tree)

"""TREINO"""

decision_tree = DecisionTreeClassifier(criterion='gini',max_depth=5, min_samples_leaf=2, min_samples_split=4)

#TREINO
decision_tree.fit(xtrain, ytrain)

x_pred = decision_tree.predict(xtrain)
print("\n Árvore de decisão - melhor modelo: ", estimador_tree)
print("\nTREINO: \nAcurácia = ", metrics.accuracy_score(ytrain, x_pred))
print(classification_report(ytrain, x_pred))

#matriz de confusão - TREINO
ConfusionMatrixDisplay.from_estimator(decision_tree, xtrain, ytrain, display_labels=target_names)
plt.show()

"""TESTE"""

# TESTE
y_pred = decision_tree.predict(xtest)
predictions.append(y_pred)
print("\n Árvore de decisão - melhor modelo: ", estimador_tree)
print("\nTESTE \nAcurácia: ", metrics.accuracy_score(ytest, y_pred))
print(classification_report(ytest, y_pred))


#matriz de confusão - TESTE
ConfusionMatrixDisplay.from_estimator(decision_tree, xtest, ytest, display_labels=target_names)
plt.show()

"""### ANN ([Redes Neurais Artificiais](https://scikit-learn.org/stable/modules/neural_networks_supervised.html))

Utilizou-se do `MLPClassifier` (classificador perceptron multicamadas) do `sklearn` como modelo
"""

param_dict = {
    'alpha': [0.0001, 0.001, 0.01, 0.1, 1.0],
    'tol': [0.001, 0.01, 0.1],
    'activation': ['identity', 'logistic', 'tanh', 'relu'],
}

net = MLPClassifier(activation='relu', hidden_layer_sizes=(20,),  alpha=0.01, tol=0.001, max_iter=500, solver='sgd')
grid = GridSearchCV(net, param_dict, cv=rkf)

#TREINO
grid.fit(xtrain, ytrain)

x_pred = grid.predict(xtrain)
estimador_rna, escore_rna = grid.best_estimator_, grid.best_score_
print("\n Rede neural - melhor modelo: ", estimador_rna)
print("Acurácia = ", metrics.accuracy_score(ytrain, x_pred))


#matriz de confusão - TREINO
ConfusionMatrixDisplay.from_estimator(grid, xtrain, ytrain, display_labels=target_names)
plt.show()

x_melhor = grid.best_estimator_.predict(xtrain)
y_hat = grid.best_estimator_.predict(xtest)
print("\n\nTESTE - \n\nRelatório melhor estimador: ", estimador_rna)
print("Acurácia = ", metrics.accuracy_score(ytest, y_hat))
print(classification_report(ytest, y_hat, target_names=target_names, zero_division=0))

mat = confusion_matrix(ytest, y_hat)
fig, ax = plt.subplots()
cax = ax.matshow(mat, cmap='summer')
ticks = np.arange(0, len(target_names))
ax.set_xticks(ticks)
ax.set_yticks(ticks)
ax.set_xticklabels(target_names, rotation=45, ha='right')
ax.set_yticklabels(target_names, rotation=45, ha='right')
ax.set_ylabel('True label')
ax.set_xlabel('Predicted label')
ax.xaxis.set_ticks_position('bottom')

for i in range(mat.shape[0]):
    for j in range(mat.shape[1]):
        ax.text(j, i, mat[i, j], ha='center', va='center')

knn = KNeighborsClassifier(n_neighbors=3)
svm = SVC(kernel='linear', C=100)
dt = DecisionTreeClassifier(criterion='gini',max_depth=5, min_samples_leaf=2, min_samples_split=4)
rna = MLPClassifier(activation='identity', alpha=0.01, hidden_layer_sizes=(20,), max_iter=500, solver='sgd', tol=0.001)

#matriz de confusão - TESTE
  ConfusionMatrixDisplay.from_estimator(svc_pol, xtest, ytest, display_labels=target_names)
  plt.show()

# Realizar a validação cruzada entre algoritmos individuais
for clf, label in zip([knn, svm, dt, rna], ['KNN', 'SVM', 'DT', 'RNA']):
    scores = cross_val_score(clf, X, Y, cv=10, scoring='accuracy')
    print("Accuracy: %0.2f (+/- %0.2f) [%s]" % (scores.mean(), scores.std(), label))

# Gerar o gráfico de barras com as acurácias dos modelos
plt.figure(figsize=(8, 6))
plt.bar(['KNN', 'SVM', 'DT', 'RNA'], [0.83, 0.89, 0.84, 0.88])
plt.ylim(0.8, 0.95)
plt.title('Acurácia dos modelos')
plt.xlabel('Modelo')
plt.ylabel('Acurácia')
plt.show()

# Realizar a validação cruzada entre algoritmos individuais
for clf, label in zip([knn, svm, dt, rna], ['KNN', 'SVM', 'DT', 'RNA']):
    scores = cross_val_score(clf, X, Y, cv=10, scoring='f1_macro')
    print("Accuracy: %0.2f (+/- %0.2f) [%s]" % (scores.mean(), scores.std(), label))

# Gerar o gráfico de barras com as acurácias dos modelos
plt.figure(figsize=(8, 6))
plt.bar(['KNN', 'SVM', 'DT', 'RNA'], [0.83, 0.89, 0.84, 0.88])
plt.ylim(0.8, 0.95)
plt.title('Macro f1-score dos modelos')
plt.xlabel('Modelo')
plt.ylabel('Macro f1-score')
plt.show()

# Realizar a validação cruzada entre algoritmos individuais
for clf, label in zip([knn, svm, dt, rna], ['KNN', 'SVM', 'DT', 'RNA']):
    scores = cross_val_score(clf, X, Y, cv=10, scoring='f1_micro')
    print("Accuracy: %0.2f (+/- %0.2f) [%s]" % (scores.mean(), scores.std(), label))

# Gerar o gráfico de barras com as acurácias dos modelos
plt.figure(figsize=(8, 6))
plt.bar(['KNN', 'SVM', 'DT', 'RNA'], [0.83, 0.89, 0.84, 0.88])
plt.ylim(0.8, 0.95)
plt.title('Micro f1-score dos modelos')
plt.xlabel('Modelo')
plt.ylabel('Micro f1-score')
plt.show()

"""##4.2 - Avaliação dos comitês nos Modelos de AM

###[Bagging](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.BaggingClassifier.html)

Bagging tem um bom desempenho em geral e fornece a base para todo um campo de algoritmos de árvore de decisão, como os populares algoritmos de floresta aleatória e conjuntos de árvores extras, bem como os menos conhecidos algoritmos de colagem, subespaços aleatórios e patches aleatórios

que muitas vezes considera aprendizes fracos homogêneos, aprende-os independentemente uns dos outros em paralelo e os combina seguindo algum tipo de processo de média determinística
"""

# obter uma lista de modelos para avaliar
def get_models():
  models = dict()
  models['KNN'] = KNeighborsClassifier(n_neighbors=3)
  models['SVM'] = SVC(kernel='linear', C=100)
  models['DT'] = DecisionTreeClassifier(criterion='gini',max_depth=5, min_samples_leaf=2, min_samples_split=4)
  models['RNA'] = MLPClassifier(activation='identity', alpha=0.01, hidden_layer_sizes=(20,), max_iter=500, solver='sgd', tol=0.001)

  return models

models = get_models()
results, names = list(), list()
for name, model in models.items():
  bagging = BaggingClassifier(model)
  bagging.fit(xtrain, ytrain)
  y_pred = bagging.predict(xtest)
  score = metrics.accuracy_score(ytest, y_pred)
  print('\n\n>  %s %.4f' % (name, score))
  print (classification_report(ytest, y_pred, target_names=target_names, zero_division=0))

# Realizar a validação cruzada com o ensemble Bagging e os modelos individuais
for clf, label in zip([knn, svm, dt, rna, bagging], ['KNN', 'SVM', 'DT', 'RNA', 'Ensemble Bagging']):
    scores = cross_val_score(clf, X, Y, cv=10, scoring='accuracy')
    print("Accuracy: %0.2f (+/- %0.2f) [%s]" % (scores.mean(), scores.std(), label))

# Gerar o gráfico de barras com as acurácias dos modelos
plt.figure(figsize=(8, 6))
plt.bar(['KNN', 'SVM', 'DT', 'RNA', 'Ensemble Bagging'], [0.83, 0.89, 0.84, 0.88, 0.91])
plt.ylim(0.8, 0.95)
plt.title('Acurácia dos modelos')
plt.xlabel('Modelo')
plt.ylabel('Acurácia')
plt.show()

"""###[Boosting](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.AdaBoostClassifier.html)

que muitas vezes considera aprendizes fracos homogêneos, os aprende sequencialmente de forma muito adaptativa (um modelo base depende dos anteriores) e os combina seguindo uma estratégia determinística

"""

# obter uma lista de modelos para avaliar
def get_models():
  models = dict()
  models['SVM'] = SVC(kernel='linear', C=100)
  models['DT'] = DecisionTreeClassifier(criterion='gini',max_depth=5, min_samples_leaf=2, min_samples_split=4)

  return models

models = get_models()
results, names = list(), list()
for name, model in models.items():
  boosting = AdaBoostClassifier(model, algorithm='SAMME')
  boosting.fit(xtrain, ytrain)
  y_pred = boosting.predict(xtest)
  score = metrics.accuracy_score(ytest, y_pred)
  print('\n\n>  %s %.4f' % (name, score))
  print (classification_report(ytest, y_pred, target_names=target_names, zero_division=0))

boosting = AdaBoostClassifier(SVC(kernel='linear', C=100), algorithm='SAMME')
  boosting.fit(xtrain, ytrain)
  y_pred = boosting.predict(xtest)
  score = metrics.accuracy_score(ytest, y_pred)
  print('> SVM %.4f' % (score))
  print (classification_report(ytest, y_pred, target_names=target_names, zero_division=0))

boosting = AdaBoostClassifier(DecisionTreeClassifier(criterion='gini',max_depth=5, min_samples_leaf=2, min_samples_split=4))
  boosting.fit(xtrain, ytrain)
  y_pred = boosting.predict(xtest)
  score = metrics.accuracy_score(ytest, y_pred)
  print('> DT %.4f' % (score))
  print (classification_report(ytest, y_pred, target_names=target_names, zero_division=0))

# Realizar a validação cruzada com o ensemble Boosting e os modelos individuais
for clf, label in zip([svm, dt, boosting], ['SVM', 'DT', 'Ensemble Boosting']):
    scores = cross_val_score(clf, X, Y, cv=10, scoring='accuracy')
    print("Accuracy: %0.2f (+/- %0.2f) [%s]" % (scores.mean(), scores.std(), label))

# Gerar o gráfico de barras com as acurácias dos modelos
plt.figure(figsize=(8, 6))
plt.bar(['SVM', 'DT', 'Ensemble Boosting'], [0.89, 0.84, 0.88])
plt.ylim(0.8, 0.95)
plt.title('Acurácia dos modelos')
plt.xlabel('Modelo')
plt.ylabel('Acurácia')
plt.show()

"""###[Stacking](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.StackingClassifier.html)

que geralmente considera aprendizes fracos heterogêneos, aprende-os em paralelo e os combina treinando um meta-modelo para produzir uma previsão com base nas diferentes previsões de modelos fracos
"""

knn = KNeighborsClassifier(n_neighbors=3)
svm = SVC(kernel='linear', C=100)
dt = DecisionTreeClassifier(criterion='gini',max_depth=5, min_samples_leaf=2, min_samples_split=4)
rna = MLPClassifier(activation='identity', alpha=0.01, hidden_layer_sizes=(20,), max_iter=500, solver='sgd', tol=0.001)

estimators = [
            ('knn',knn),
            ('svm',svm),
            ('dt',dt),
            ('rna',rna),
            ]

# Definindo o modelo ensemble com a estratégia de Stacking
ensemble_stacking = StackingClassifier(estimators=estimators, cv=5)

# Treinando o modelo ensemble
ensemble_stacking.fit(xtrain,ytrain)

# Fazendo as previsões no conjunto de testes
y_pred = ensemble_stacking.predict(xtest)
print("> Stacking metrícas: ", metrics.accuracy_score(ytest, y_pred))
print(classification_report(ytest, y_pred, target_names=target_names, zero_division=0))

stacking_knn = StackingClassifier(estimators=estimators, final_estimator=knn, cv=5)
stacking_svm = StackingClassifier(estimators=estimators, final_estimator=svm, cv=5)
stacking_dt = StackingClassifier(estimators=estimators, final_estimator=dt, cv=5)
stacking_rna = StackingClassifier(estimators=estimators, final_estimator=rna, cv=5)

#knn
stacking_knn.fit(xtrain,ytrain)
y_pred = stacking_knn.predict(xtest)
print("> KNN ", metrics.accuracy_score(ytest, y_pred))
print(classification_report(ytest, y_pred, target_names=target_names, zero_division=0))

#svm
stacking_svm.fit(xtrain,ytrain)
y_pred = stacking_svm.predict(xtest)
print("> SVM ", metrics.accuracy_score(ytest, y_pred))
print(classification_report(ytest, y_pred, target_names=target_names, zero_division=0))

#Árvore de decisão
stacking_dt.fit(xtrain,ytrain)
y_pred = stacking_dt.predict(xtest)
print("> DT ", metrics.accuracy_score(ytest, y_pred))
print(classification_report(ytest, y_pred, target_names=target_names, zero_division=0))

#Redes neurais
stacking_rna.fit(xtrain,ytrain)
y_pred = stacking_rna.predict(xtest)
print("> RNA ", metrics.accuracy_score(ytest, y_pred))
print (classification_report(ytest, y_pred, target_names=target_names, zero_division=0))

# Realizar a validação cruzada com o ensemble Stacking e os modelos individuais
for clf, label in zip([knn, svm, dt, rna, ensemble_stacking], ['KNN', 'SVM', 'DT', 'RNA', 'Ensemble Stacking']):
    scores = cross_val_score(clf, X, Y, cv=10, scoring='accuracy')
    print("Accuracy: %0.2f (+/- %0.2f) [%s]" % (scores.mean(), scores.std(), label))

# Gerar o gráfico de barras com as acurácias dos modelos
plt.figure(figsize=(8, 6))
plt.bar(['KNN', 'SVM', 'DT', 'RNA', 'Ensemble Stacking'], [0.83, 0.89, 0.84, 0.88, 0.91])
plt.ylim(0.8, 0.95)
plt.title('Acurácia dos modelos')
plt.xlabel('Modelo')
plt.ylabel('Acurácia')
plt.show()

"""###[Voting](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.VotingClassifier.html)

Na estratégia de Voting, cada modelo é treinado independentemente e, em seguida, suas previsões são combinadas por meio de uma votação. Existem duas formas de votação: a votação dura (hard voting) e a votação suave (soft voting). Na votação dura, a classe prevista pela maioria dos modelos é selecionada como a previsão final, enquanto na votação suave, é feita uma média das probabilidades previstas por cada modelo e a classe com a maior probabilidade é selecionada como a previsão final.

"""

knn = KNeighborsClassifier(n_neighbors=3)
svm = SVC(kernel='linear', C=100)
dt = DecisionTreeClassifier(criterion='gini',max_depth=5, min_samples_leaf=2, min_samples_split=4)
rna = MLPClassifier(activation='identity', alpha=0.01, hidden_layer_sizes=(20,), max_iter=500, solver='sgd', tol=0.001)

estimators = [
            ('knn',knn),
            ('svm',svm),
            ('dt',dt),
            ('rna',rna),
            ]

# Definindo o modelo ensemble com a estratégia de Voting
ensemble_voting = VotingClassifier(estimators=estimators, voting='hard')

# Treinando o modelo ensemble
ensemble_voting.fit(xtrain,ytrain)

# Fazendo as previsões no conjunto de testes
y_pred = ensemble_voting.predict(xtest)
print("> Voting metrícas: ", metrics.accuracy_score(ytest, y_pred))
print(classification_report(ytest, y_pred, target_names=target_names, zero_division=0))

# Avaliando a performance do modelo ensemble
accuracy = ensemble_voting.score(xtest, ytest)
print('Acurácia: {:.2f}%'.format(accuracy*100))

voiting_knn = VotingClassifier(estimators=estimators, voting='hard')
voiting_svm = VotingClassifier(estimators=estimators, voting='hard')
voiting_dt = VotingClassifier(estimators=estimators, voting='hard')
voiting_rna = VotingClassifier(estimators=estimators, voting='hard')

#knn
voiting_knn.fit(xtrain,ytrain)
y_pred = voiting_knn.predict(xtest)
print("> KNN ", metrics.accuracy_score(ytest, y_pred))
print(classification_report(ytest, y_pred, target_names=target_names, zero_division=0))

#svm
voiting_svm.fit(xtrain,ytrain)
y_pred = voiting_svm.predict(xtest)
print("> SVM ", metrics.accuracy_score(ytest, y_pred))
print(classification_report(ytest, y_pred, target_names=target_names, zero_division=0))

#Árvore de decisão
voiting_dt.fit(xtrain,ytrain)
y_pred = voiting_dt.predict(xtest)
print("> DT ", metrics.accuracy_score(ytest, y_pred))
print(classification_report(ytest, y_pred, target_names=target_names, zero_division=0))

#Redes neurais
voiting_rna.fit(xtrain,ytrain)
y_pred = voiting_rna.predict(xtest)
print("> RNA ", metrics.accuracy_score(ytest, y_pred))
print (classification_report(ytest, y_pred, target_names=target_names, zero_division=0))

[
            ('knn',knn),
            ('svm',svm),
            ('dt',dt),
            ('rna',rna),
            ]

# Realizar a validação cruzada com o ensemble Voting e os modelos individuais
for clf, label in zip([knn, svm, dt, rna, ensemble_voting], ['KNN', 'SVM', 'DT', 'RNA', 'Ensemble Voting']):
    scores = cross_val_score(clf, X, Y, cv=10, scoring='accuracy')
    print("Accuracy: %0.2f (+/- %0.2f) [%s]" % (scores.mean(), scores.std(), label))

# Gerar o gráfico de barras com as acurácias dos modelos
plt.figure(figsize=(8, 6))
plt.bar(['KNN', 'SVM', 'DT', 'RNA', 'Ensemble Voting'], [0.83, 0.89, 0.84, 0.88, 0.91])
plt.ylim(0.8, 0.95)
plt.title('Acurácia dos modelos')
plt.xlabel('Modelo')
plt.ylabel('Acurácia')
plt.show()
