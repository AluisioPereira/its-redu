# -*- coding: utf-8 -*-
"""Artigo 11 - ANTECIPAÇÃO DE DEMANDAS DE TUTORIA A PARTIR DAS DIFICULDADES DOS ESTUDANTES NO APRENDIZADO ON-LINE

Original file: Aluisio Pereira

# Bibliotecas
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import LabelEncoder

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Embedding, LSTM

import pandas as pd
import numpy as np
import re
import string
import nltk
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
from keras.preprocessing.text import Tokenizer
#from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Dense, Embedding, LSTM

"""# Coletas das Mensagens"""

# importando a base de dados - Interaçãoes dos Estudantes
!wget "DADOS DE MENSAGENS DO WhatsApp" -O "DADOS.csv" -

# Carregando o dataset
data = pd.read_csv("DADOS.csv",  sep=",", encoding = "ISO-8859-1", low_memory=False)

"""# Análise das Mensagens"""

# carregando o dataset
# df = pd.read_csv('mensagens.csv')
df = data

# removendo pontuação, stopwords, caracteres especiais e emoticons
nltk.download('stopwords')
stop_words = set(stopwords.words('portuguese'))
def clean_text(text):
    text = text.lower()
    text = re.sub('[%s]' % re.escape(string.punctuation), '', text)
    text = re.sub('[^\w\s]', '', text)  # remoção de caracteres especiais
    text = re.sub('[^\x00-\x7F]+', '', text)  # remoção de emoticons
    text = re.sub(r'(.)\1+', r'\1\1', text)  # remoção de pontuações excessivas
    text = ' '.join([word for word in text.split() if word not in stop_words])
    return text

df['mensagens_estudantes'] = df['mensagens_estudantes'].apply(clean_text)

# tokenização das mensagens
tokenizer = Tokenizer(num_words=5000, split=' ')
tokenizer.fit_on_texts(df['mensagens_estudantes'].values)
X = tokenizer.texts_to_sequences(df['mensagens_estudantes'].values)
X = pad_sequences(X)

"""# Definição do Modelo de PLN

## Modelo de Rede Neural Covolucional (CNN)
"""

# definindo o modelo Rede Neural Convolucional (CNN)
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, Conv1D, GlobalMaxPooling1D, Dense

model = Sequential()
model.add(Embedding(5000, 128, input_length=X.shape[1]))
model.add(Conv1D(filters=64, kernel_size=3, activation='relu'))
model.add(GlobalMaxPooling1D())
model.add(Dense(128, activation='relu'))
model.add(Dense(3, activation='sigmoid'))

# Dense ajustar conforme a quantidade de classes
# ValueError - Dense (número de classes, ex.: 4)
model.add(Dense(3, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# dividindo em dados de treinamento e teste
Y = pd.get_dummies(df['classe']).values
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=80)

# treinando o modelo
model.fit(X_train, Y_train, epochs=10, batch_size=80)

# avaliando o modelo
print('Acurácia do modelo - para estilo de tutoria:')
print(model.evaluate(X_test, Y_test)[1])

import matplotlib.pyplot as plt

# Treinamento do modelo
history = model.fit(X_train, Y_train, epochs=10, batch_size=80, validation_data=(X_test, Y_test))

# Obter histórico de perda e acurácia
train_loss = history.history['loss']
val_loss = history.history['val_loss']
train_acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

# Plotar gráfico de perda
plt.plot(train_loss, label='Perda de Treinamento')
plt.plot(val_loss, label='Perda de Validação')
plt.title('Gráfico de Perda')
plt.xlabel('Épocas')
plt.ylabel('Perda')
plt.legend()
plt.show()

# Plotar gráfico de acurácia
plt.plot(train_acc, label='Acurácia de Treinamento')
plt.plot(val_acc, label='Acurácia de Validação')
plt.title('Gráfico de Acurácia')
plt.xlabel('Épocas')
plt.ylabel('Acurácia')
plt.legend()
plt.show()

from sklearn.metrics import confusion_matrix
import seaborn as sns

# Fazer as previsões no conjunto de teste
Y_pred = model.predict(X_test)
Y_pred = (Y_pred > 0.5)

# Converter as previsões em rótulos de classe
Y_pred_classes = np.argmax(Y_pred, axis=1)
Y_test_classes = np.argmax(Y_test, axis=1)

# Calcular a matriz de confusão
confusion_mtx = confusion_matrix(Y_test_classes, Y_pred_classes)

# Definir os rótulos das classes
class_labels = ['tecnicas', 'outras', 'pessoais']

# Plotar a matriz de confusão
plt.figure(figsize=(8, 6))
sns.heatmap(confusion_mtx, annot=True, fmt="d", cmap='Blues', xticklabels=class_labels, yticklabels=class_labels)
plt.title('Matriz de Confusão')
plt.xlabel('Previsão')
plt.ylabel('Real')
plt.show()

"""## Modelo de Rede Neural Recorrente (RNN)"""

# definindo o modelo Rede Neural Recorrente (RNN)
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense

model = Sequential()
model.add(Embedding(5000, 128, input_length=X.shape[1]))
model.add(LSTM(128, dropout=0.2, recurrent_dropout=0.2))
model.add(Dense(3, activation='sigmoid'))

# Dense ajustar conforme a quantidade de classes
# ValueError - Dense (número de classes, ex.: 7)
model.add(Dense(3, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# dividindo em dados de treinamento e teste
Y = pd.get_dummies(df['classe']).values
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=80)

# treinando o modelo
model.fit(X_train, Y_train, epochs=10, batch_size=80)

# avaliando o modelo
print('Acurácia do modelo - para estilo de tutoria:')
print(model.evaluate(X_test, Y_test)[1])

import matplotlib.pyplot as plt

# Treinamento do modelo
history = model.fit(X_train, Y_train, epochs=10, batch_size=80, validation_data=(X_test, Y_test))

# Obter histórico de perda e acurácia
train_loss = history.history['loss']
val_loss = history.history['val_loss']
train_acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

# Plotar gráfico de perda
plt.plot(train_loss, label='Perda de Treinamento')
plt.plot(val_loss, label='Perda de Validação')
plt.title('Gráfico de Perda')
plt.xlabel('Épocas')
plt.ylabel('Perda')
plt.legend()
plt.show()

# Plotar gráfico de acurácia
plt.plot(train_acc, label='Acurácia de Treinamento')
plt.plot(val_acc, label='Acurácia de Validação')
plt.title('Gráfico de Acurácia')
plt.xlabel('Épocas')
plt.ylabel('Acurácia')
plt.legend()
plt.show()

from sklearn.metrics import confusion_matrix
import seaborn as sns

# Fazer as previsões no conjunto de teste
Y_pred = model.predict(X_test)
Y_pred = (Y_pred > 0.5)

# Converter as previsões em rótulos de classe
Y_pred_classes = np.argmax(Y_pred, axis=1)
Y_test_classes = np.argmax(Y_test, axis=1)

# Calcular a matriz de confusão
confusion_mtx = confusion_matrix(Y_test_classes, Y_pred_classes)

# Definir os rótulos das classes
class_labels = ['tecnicas', 'outras', 'pessoais']

# Plotar a matriz de confusão
plt.figure(figsize=(8, 6))
sns.heatmap(confusion_mtx, annot=True, fmt="d", cmap='Blues', xticklabels=class_labels, yticklabels=class_labels)
plt.title('Matriz de Confusão')
plt.xlabel('Previsão')
plt.ylabel('Real')
plt.show()

"""## Modelo de Multi-Layer Perceptron (MLP)"""

# definindo o modelo com Multi-Layer Perceptron
model = Sequential()
model.add(Dense(128, input_dim=X.shape[1], activation='relu'))
model.add(Dense(128, activation='relu'))
model.add(Dense(3, activation='sigmoid'))

# Dense ajustar conforme a quantidade de classes
# ValueError - Dense (número de classes, ex.: 7)
model.add(Dense(3, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# dividindo em dados de treinamento e teste
Y = pd.get_dummies(df['classe']).values
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=80)

# treinando o modelo
model.fit(X_train, Y_train, epochs=10, batch_size=80)

# avaliando o modelo
print('Acurácia do modelo - para estilo de tutoria:')
print(model.evaluate(X_test, Y_test)[1])

import matplotlib.pyplot as plt

# Treinamento do modelo
history = model.fit(X_train, Y_train, epochs=10, batch_size=80, validation_data=(X_test, Y_test))

# Obter histórico de perda e acurácia
train_loss = history.history['loss']
val_loss = history.history['val_loss']
train_acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

# Plotar gráfico de perda
plt.plot(train_loss, label='Perda de Treinamento')
plt.plot(val_loss, label='Perda de Validação')
plt.title('Gráfico de Perda')
plt.xlabel('Épocas')
plt.ylabel('Perda')
plt.legend()
plt.show()

# Plotar gráfico de acurácia
plt.plot(train_acc, label='Acurácia de Treinamento')
plt.plot(val_acc, label='Acurácia de Validação')
plt.title('Gráfico de Acurácia')
plt.xlabel('Épocas')
plt.ylabel('Acurácia')
plt.legend()
plt.show()


from sklearn.metrics import confusion_matrix
import seaborn as sns

# Fazer as previsões no conjunto de teste
Y_pred = model.predict(X_test)
Y_pred = (Y_pred > 0.5)

# Converter as previsões em rótulos de classe
Y_pred_classes = np.argmax(Y_pred, axis=1)
Y_test_classes = np.argmax(Y_test, axis=1)

# Calcular a matriz de confusão
confusion_mtx = confusion_matrix(Y_test_classes, Y_pred_classes)

# Definir os rótulos das classes
class_labels = ['tecnicas', 'outras', 'pessoais']

# Plotar a matriz de confusão
plt.figure(figsize=(8, 6))
sns.heatmap(confusion_mtx, annot=True, fmt="d", cmap='Blues', xticklabels=class_labels, yticklabels=class_labels)
plt.title('Matriz de Confusão')
plt.xlabel('Previsão')
plt.ylabel('Real')
plt.show()

"""

## Modelo de Long Short-Term (LSTM)"""

# definindo o modelo - Sequencial
model = Sequential()
model.add(Embedding(5000, 128, input_length=X.shape[1]))
model.add(LSTM(128, dropout=0.2, recurrent_dropout=0.2))

# Dense ajustar conforme a quantidade de classes
# ValueError - Dense (número de classes, ex.: 7)
model.add(Dense(3, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# dividindo em dados de treinamento e teste
Y = pd.get_dummies(df['classe']).values
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=80)

# treinando o modelo
model.fit(X_train, Y_train, epochs=10, batch_size=80)

# avaliando o modelo
print('Acurácia do modelo - para estilo de tutoria:')
print(model.evaluate(X_test, Y_test)[1])

import matplotlib.pyplot as plt

# Treinamento do modelo
history = model.fit(X_train, Y_train, epochs=10, batch_size=80, validation_data=(X_test, Y_test))

# Obter histórico de perda e acurácia
train_loss = history.history['loss']
val_loss = history.history['val_loss']
train_acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

# Plotar gráfico de perda
plt.plot(train_loss, label='Perda de Treinamento')
plt.plot(val_loss, label='Perda de Validação')
plt.title('Gráfico de Perda')
plt.xlabel('Épocas')
plt.ylabel('Perda')
plt.legend()
plt.show()

# Plotar gráfico de acurácia
plt.plot(train_acc, label='Acurácia de Treinamento')
plt.plot(val_acc, label='Acurácia de Validação')
plt.title('Gráfico de Acurácia')
plt.xlabel('Épocas')
plt.ylabel('Acurácia')
plt.legend()
plt.show()



from sklearn.metrics import confusion_matrix
import seaborn as sns

# Fazer as previsões no conjunto de teste
Y_pred = model.predict(X_test)
Y_pred = (Y_pred > 0.5)

# Converter as previsões em rótulos de classe
Y_pred_classes = np.argmax(Y_pred, axis=1)
Y_test_classes = np.argmax(Y_test, axis=1)

# Calcular a matriz de confusão
confusion_mtx = confusion_matrix(Y_test_classes, Y_pred_classes)

# Definir os rótulos das classes
class_labels = ['tecnicas', 'outras', 'pessoais']

# Plotar a matriz de confusão
plt.figure(figsize=(8, 6))
sns.heatmap(confusion_mtx, annot=True, fmt="d", cmap='Blues', xticklabels=class_labels, yticklabels=class_labels)
plt.title('Matriz de Confusão')
plt.xlabel('Previsão')
plt.ylabel('Real')
plt.show()

"""### variando os parâmetros"""

import matplotlib.pyplot as plt
def graficos():
    # Treinamento do modelo
    history = model.fit(X_train, Y_train, epochs=10, batch_size=80, validation_data=(X_test, Y_test))

    # Obter histórico de perda e acurácia
    train_loss = history.history['loss']
    val_loss = history.history['val_loss']
    train_acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']

    # Plotar gráfico de perda
    plt.plot(train_loss, label='Perda de Treinamento')
    plt.plot(val_loss, label='Perda de Validação')
    plt.title('Gráfico de Perda')
    plt.xlabel('Épocas')
    plt.ylabel('Perda')
    plt.legend()
    plt.show()

    # Plotar gráfico de acurácia
    plt.plot(train_acc, label='Acurácia de Treinamento')
    plt.plot(val_acc, label='Acurácia de Validação')
    plt.title('Gráfico de Acurácia')
    plt.xlabel('Épocas')
    plt.ylabel('Acurácia')
    plt.legend()
    plt.show()



    from sklearn.metrics import confusion_matrix
    import seaborn as sns

    # Fazer as previsões no conjunto de teste
    Y_pred = model.predict(X_test)
    Y_pred = (Y_pred > 0.5)

    # Converter as previsões em rótulos de classe
    Y_pred_classes = np.argmax(Y_pred, axis=1)
    Y_test_classes = np.argmax(Y_test, axis=1)

    # Calcular a matriz de confusão
    confusion_mtx = confusion_matrix(Y_test_classes, Y_pred_classes)

    # Definir os rótulos das classes
    class_labels = ['tecnicas', 'outras', 'pessoais']

    # Plotar a matriz de confusão
    plt.figure(figsize=(8, 6))
    sns.heatmap(confusion_mtx, annot=True, fmt="d", cmap='Blues', xticklabels=class_labels, yticklabels=class_labels)
    plt.title('Matriz de Confusão')
    plt.xlabel('Previsão')
    plt.ylabel('Real')
    plt.show()

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense
from tensorflow.keras.optimizers import Adam

# Definir X e Y adequadamente
Y = pd.get_dummies(df['classe']).values
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=80)

# Variar os parâmetros do modelo de PLN
embedding_sizes = [64, 128, 256]
lstm_units = [64, 128, 256]
dropouts = [0.2, 0.3, 0.4]
optimizers = ['adam', 'rmsprop', 'sgd']

best_accuracy = 0.0
best_params = {}

# Loop para variar os parâmetros e encontrar a melhor configuração
for embedding_size in embedding_sizes:
    for lstm_unit in lstm_units:
        for dropout in dropouts:
            for optimizer in optimizers:
                # Definir o modelo sequencial
                model = Sequential()
                model.add(Embedding(5000, embedding_size, input_length=X.shape[1]))
                model.add(LSTM(lstm_unit, dropout=dropout, recurrent_dropout=dropout))
                model.add(Dense(3, activation='sigmoid'))

                # Compilar o modelo com o otimizador especificado
                model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])

                # Dividir em dados de treinamento e teste
                X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=80)

                # Treinar o modelo
                model.fit(X_train, Y_train, epochs=10, batch_size=80)

                # Avaliar o modelo
                accuracy = model.evaluate(X_test, Y_test)[1]
                print(f"Acurácia do modelo: {accuracy}")

                # Verificar se a acurácia atual é a melhor
                if accuracy > best_accuracy:
                    best_accuracy = accuracy
                    best_params = {
                        'embedding_size': embedding_size,
                        'lstm_units': lstm_unit,
                        'dropout': dropout,
                        'optimizer': optimizer
                    }
        graficos()
# Exibir os melhores parâmetros encontrados
print("Melhor configuração:")
print(best_params)
print("Melhor acurácia:")
print(best_accuracy)

"""# melhor modelo

## configuração I
"""

# definindo o modelo - Sequencial

model = Sequential()
embedding_size=128
lstm_units=128
dropout= 0.2
optimizer='adam'

model.add(Embedding(5000, embedding_size, input_length=X.shape[1]))
model.add(LSTM(lstm_unit, dropout=dropout, recurrent_dropout=dropout))
model.add(Dense(3, activation='sigmoid'))


model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# dividindo em dados de treinamento e teste
Y = pd.get_dummies(df['classe']).values
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=80)

# treinando o modelo
model.fit(X_train, Y_train, epochs=10, batch_size=80)

# avaliando o modelo
print('Acurácia do modelo - para estilo de tutoria:')
print(model.evaluate(X_test, Y_test)[1])

import matplotlib.pyplot as plt

# Treinamento do modelo
history = model.fit(X_train, Y_train, epochs=10, batch_size=80, validation_data=(X_test, Y_test))

# Obter histórico de perda e acurácia
train_loss = history.history['loss']
val_loss = history.history['val_loss']
train_acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

# Plotar gráfico de perda
plt.plot(train_loss, label='Perda de Treinamento')
plt.plot(val_loss, label='Perda de Validação')
plt.title('Gráfico de Perda')
plt.xlabel('Épocas')
plt.ylabel('Perda')
plt.legend()
plt.show()

# Plotar gráfico de acurácia
plt.plot(train_acc, label='Acurácia de Treinamento')
plt.plot(val_acc, label='Acurácia de Validação')
plt.title('Gráfico de Acurácia')
plt.xlabel('Épocas')
plt.ylabel('Acurácia')
plt.legend()
plt.show()



from sklearn.metrics import confusion_matrix
import seaborn as sns

# Fazer as previsões no conjunto de teste
Y_pred = model.predict(X_test)
Y_pred = (Y_pred > 0.5)

# Converter as previsões em rótulos de classe
Y_pred_classes = np.argmax(Y_pred, axis=1)
Y_test_classes = np.argmax(Y_test, axis=1)

# Calcular a matriz de confusão
confusion_mtx = confusion_matrix(Y_test_classes, Y_pred_classes)

# Definir os rótulos das classes
class_labels = ['tecnicas', 'outras', 'pessoais']

# Plotar a matriz de confusão
plt.figure(figsize=(8, 6))
sns.heatmap(confusion_mtx, annot=True, fmt="d", cmap='Blues', xticklabels=class_labels, yticklabels=class_labels)
plt.title('Matriz de Confusão')
plt.xlabel('Previsão')
plt.ylabel('Real')
plt.show()

"""## configuração II"""

# definindo o modelo - Sequencial

model = Sequential()
embedding_size=256
lstm_units=256
dropout= 0.3
optimizer='adam'

model.add(Embedding(5000, embedding_size, input_length=X.shape[1]))
model.add(LSTM(lstm_unit, dropout=dropout, recurrent_dropout=dropout))
model.add(Dense(3, activation='sigmoid'))


model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# dividindo em dados de treinamento e teste
Y = pd.get_dummies(df['classe']).values
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=80)

# treinando o modelo
model.fit(X_train, Y_train, epochs=10, batch_size=80)

# avaliando o modelo
print('Acurácia do modelo - para estilo de tutoria:')
print(model.evaluate(X_test, Y_test)[1])

import matplotlib.pyplot as plt

# Treinamento do modelo
history = model.fit(X_train, Y_train, epochs=10, batch_size=80, validation_data=(X_test, Y_test))

# Obter histórico de perda e acurácia
train_loss = history.history['loss']
val_loss = history.history['val_loss']
train_acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

# Plotar gráfico de perda
plt.plot(train_loss, label='Perda de Treinamento')
plt.plot(val_loss, label='Perda de Validação')
plt.title('Gráfico de Perda')
plt.xlabel('Épocas')
plt.ylabel('Perda')
plt.legend()
plt.show()

# Plotar gráfico de acurácia
plt.plot(train_acc, label='Acurácia de Treinamento')
plt.plot(val_acc, label='Acurácia de Validação')
plt.title('Gráfico de Acurácia')
plt.xlabel('Épocas')
plt.ylabel('Acurácia')
plt.legend()
plt.show()



from sklearn.metrics import confusion_matrix
import seaborn as sns

# Fazer as previsões no conjunto de teste
Y_pred = model.predict(X_test)
Y_pred = (Y_pred > 0.5)

# Converter as previsões em rótulos de classe
Y_pred_classes = np.argmax(Y_pred, axis=1)
Y_test_classes = np.argmax(Y_test, axis=1)

# Calcular a matriz de confusão
confusion_mtx = confusion_matrix(Y_test_classes, Y_pred_classes)

# Definir os rótulos das classes
class_labels = ['tecnicas', 'outras', 'pessoais']

# Plotar a matriz de confusão
plt.figure(figsize=(8, 6))
sns.heatmap(confusion_mtx, annot=True, fmt="d", cmap='Blues', xticklabels=class_labels, yticklabels=class_labels)
plt.title('Matriz de Confusão')
plt.xlabel('Previsão')
plt.ylabel('Real')
plt.show()

"""# outros

"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense
from tensorflow.keras.optimizers import Adam

# Variando os parâmetros do modelo de PLN
embedding_sizes = [64, 128, 256]
lstm_units = [64, 128, 256]
dropouts = [0.2, 0.3, 0.4]
optimizers = ['adam', 'rmsprop', 'sgd']

best_accuracy = 0.0
best_params = {}

# Loop para variar os parâmetros e encontrar a melhor configuração
for embedding_size in embedding_sizes:
    for lstm_unit in lstm_units:
        for dropout in dropouts:
            for optimizer in optimizers:
                # Definindo o modelo sequencial
                model = Sequential()
                model.add(Embedding(5000, embedding_size, input_length=X.shape[1]))
                model.add(LSTM(lstm_unit, dropout=dropout, recurrent_dropout=dropout))
                model.add(Dense(3, activation='sigmoid'))

                # Compilando o modelo com o otimizador especificado
                model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])

                # Dividindo em dados de treinamento e teste
                X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=80)

                # Treinando o modelo
                model.fit(X_train, Y_train, epochs=10, batch_size=80)

                # Avaliando o modelo
                accuracy = model.evaluate(X_test, Y_test)[1]
                print(f"Acurácia do modelo: {accuracy}")

                # Verificando se a acurácia atual é a melhor
                if accuracy > best_accuracy:
                    best_accuracy = accuracy
                    best_params = {
                        'embedding_size': embedding_size,
                        'lstm_units': lstm_unit,
                        'dropout': dropout,
                        'optimizer': optimizer
                    }

# Exibindo os melhores parâmetros encontrados
print("Melhor configuração:")
print(best_params)
print("Melhor acurácia:")
print(best_accuracy)

import numpy as np
import matplotlib.pyplot as plt

def show_intrinsic_attention(model, X_test, vectorizer):
    # Obter camada LSTM
    lstm_layer = model.get_layer('lstm')

    # Obter pesos do LSTM
    weights = lstm_layer.get_weights()[0]  # Shape: (embedding_dim, lstm_units)

    # Obter vetor de entrada da camada Embedding
    embedding_layer = model.get_layer('embedding_1')  # Corrigido: Nome da camada Embedding
    embedding_weights = embedding_layer.get_weights()[0]  # Shape: (vocab_size, embedding_dim)

    # Calcular atenção intrínseca
    attention_weights = np.matmul(embedding_weights, weights)  # Shape: (vocab_size, lstm_units)
    attention_weights = np.abs(attention_weights).sum(axis=1)  # Somar os pesos ao longo das unidades LSTM

    # Normalizar os pesos para o intervalo [0, 1]
    attention_weights = attention_weights / np.max(attention_weights)

    # Obter palavras correspondentes aos pesos
    feature_names = vectorizer.get_feature_names_out()
    keywords = [feature_names[i] for i in attention_weights.argsort()[::-1][:10]]  # Selecionar as 10 palavras-chave mais importantes

    # Exibir as palavras-chave com atenção intrínseca
    print("Palavras-chave com atenção intrínseca:")
    for keyword in keywords:
        print(keyword)

    # Exibir gráfico de atenção intrínseca
    plt.figure(figsize=(8, 6))
    plt.barh(feature_names[:10], attention_weights[:10])
    plt.title('Atenção Intrínseca')
    plt.xlabel('Atenção')
    plt.ylabel('Palavra')
    plt.show()

# Definindo o modelo sequencial
model = Sequential()
model.add(Embedding(5000, 128, input_length=X.shape[1], name='embedding_1'))  # Corrigido: Nome da camada Embedding
model.add(LSTM(128, dropout=0.2, recurrent_dropout=0.2, name='lstm'))
model.add(Dense(3, activation='sigmoid'))

# Compilando o modelo
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# Treinando o modelo
model.fit(X_train, Y_train, epochs=10, batch_size=80)

# Avaliando o modelo
print('Acurácia do modelo - para estilo de tutoria:')
print(model.evaluate(X_test, Y_test)[1])

# Exibindo a atenção intrínseca
show_intrinsic_attention(model, X_test, vectorizer)

import numpy as np
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer

def show_intrinsic_attention(model, X_test, vectorizer):
    # Obter camada LSTM
    lstm_layer = model.get_layer('lstm')

    # Obter pesos do LSTM
    weights = lstm_layer.get_weights()[0]  # Shape: (embedding_dim, lstm_units)

    # Obter vetor de entrada da camada Embedding
    embedding_layer = model.get_layer('embedding_1')  # Corrigido: Nome da camada Embedding
    embedding_weights = embedding_layer.get_weights()[0]  # Shape: (vocab_size, embedding_dim)

    # Calcular atenção intrínseca
    attention_weights = np.matmul(embedding_weights, weights)  # Shape: (vocab_size, lstm_units)
    attention_weights = np.abs(attention_weights).sum(axis=1)  # Somar os pesos ao longo das unidades LSTM

    # Normalizar os pesos para o intervalo [0, 1]
    attention_weights = attention_weights / np.max(attention_weights)

    # Obter palavras correspondentes aos pesos
    feature_names = vectorizer.get_feature_names()
    keywords = [feature_names[i] for i in attention_weights.argsort()[::-1][:10]]  # Selecionar as 10 palavras-chave mais importantes

    # Exibir as palavras-chave com atenção intrínseca
    print("Palavras-chave com atenção intrínseca:")
    for keyword in keywords:
        print(keyword)

    # Exibir gráfico de atenção intrínseca
    plt.figure(figsize=(8, 6))
    plt.barh(feature_names[:10], attention_weights[:10])
    plt.title('Atenção Intrínseca')
    plt.xlabel('Atenção')
    plt.ylabel('Palavra')
    plt.show()

# Vetorização TF-IDF dos dados de entrada (X)
vectorizer = TfidfVectorizer()
X_tfidf = vectorizer.fit_transform(X_train)  # Ajustar e transformar os dados de treinamento

# Definindo o modelo sequencial
model = Sequential()
model.add(Embedding(5000, 128, input_length=X.shape[1], name='embedding_1'))  # Corrigido: Nome da camada Embedding
model.add(LSTM(128, dropout=0.2, recurrent_dropout=0.2, name='lstm'))
model.add(Dense(3, activation='sigmoid'))

# Compilando o modelo
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# Treinando o modelo
model.fit(X_tfidf, Y_train, epochs=10, batch_size=80)

# Transformar os dados de teste usando o vetorizador ajustado
X_test_tfidf = vectorizer.transform(X_test)

# Avaliando o modelo
print('Acurácia do modelo - para estilo de tutoria:')
print(model.evaluate(X_test_tfidf, Y_test)[1])

# Exibindo a atenção intrínseca
show_intrinsic_attention(model, X_test_tfidf, vectorizer)

"""## Gráficos"""

import matplotlib.pyplot as plt

history = model.fit(X_train, Y_train, epochs=10, batch_size=80, validation_data=(X_test, Y_test))

plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Acurácia do Modelo')
plt.xlabel('Época')
plt.ylabel('Acurácia')
plt.legend(['Treinamento', 'Teste'], loc='upper left')
plt.show()

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Perda do Modelo')
plt.xlabel('Época')
plt.ylabel('Perda')
plt.legend(['Treinamento', 'Teste'], loc='upper right')
plt.show()

from sklearn.metrics import confusion_matrix
import seaborn as sns

# Fazer as previsões no conjunto de teste
Y_pred = model.predict(X_test)
Y_pred = (Y_pred > 0.5)

# Converter as previsões em rótulos de classe
Y_pred_classes = np.argmax(Y_pred, axis=1)
Y_test_classes = np.argmax(Y_test, axis=1)

# Calcular a matriz de confusão
confusion_mtx = confusion_matrix(Y_test_classes, Y_pred_classes)

# Plotar a matriz de confusão
plt.figure(figsize=(8, 6))
sns.heatmap(confusion_mtx, annot=True, fmt="d", cmap='Blues')
plt.title('Matriz de Confusão')
plt.xlabel('Previsão')
plt.ylabel('Real')
plt.show()

class_counts = df['classe'].value_counts()

plt.figure(figsize=(8, 6))
bars = plt.bar(class_counts.index, class_counts.values)
plt.title('Distribuição das Classes')
plt.xlabel('Classes')
plt.ylabel('Contagem')
plt.xticks(rotation=60)

for bar in bars:
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width() / 2, yval, yval, ha='center', va='bottom')

plt.show()

class_names = ['tecnicas', 'outras', 'pessoais', 'sem_resposta']
class_accuracies = model.evaluate(X_test, Y_test)[1]

plt.figure(figsize=(8, 6))
plt.bar(class_names, class_accuracies)
plt.title('Acurácia por Classe')
plt.xlabel('Classes')
plt.ylabel('Acurácia')
plt.xticks(rotation=90)
plt.show()

from sklearn.metrics import roc_curve, auc

# Calcular as probabilidades preditas para cada classe
Y_pred_prob = model.predict_step(X_test)

# Calcular a curva ROC para cada classe
fpr = dict()
tpr = dict()
roc_auc = dict()
for i in range(len(class_names)):
    fpr[i], tpr[i], _ = roc_curve(Y_test[:, i], Y_pred_prob[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])

# Plotar as curvas ROC
plt.figure(figsize=(8, 6))
for i in range(len(class_names)):
    plt.plot(fpr[i], tpr[i], label='Classe {}: AUC = {:.2f}'.format(class_names[i], roc_auc[i]))

plt.plot([0, 1], [0, 1], 'k--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('Taxa de Falso Positivo')
plt.ylabel('Taxa de Verdadeiro Positivo')
plt.title('Curva ROC')
plt.legend(loc='lower right')
plt.show()

# carregando o dataset
df = data

# removendo pontuação e stopwords
nltk.download('stopwords')
stop_words = set(stopwords.words('portuguese'))
def clean_text(text):
    text = text.lower()
    text = re.sub('[%s]' % re.escape(string.punctuation), '', text)
    text = ' '.join([word for word in text.split() if word not in stop_words])
    return text

df['mensagens_estudantes'] = df['mensagens_estudantes'].apply(clean_text)

# tokenização das mensagens
tokenizer = Tokenizer(num_words=5000, split=' ')
tokenizer.fit_on_texts(df['mensagens_estudantes'].values)
X = tokenizer.texts_to_sequences(df['mensagens_estudantes'].values)
X = pad_sequences(X)

# definindo o modelo
model = Sequential()
model.add(Embedding(5000, 128, input_length=X.shape[1]))
model.add(LSTM(128, dropout=0.2, recurrent_dropout=0.2))

#Dense ajutar conforme a quantidade de classe
# ValueError - Dense (numero de classe, ex.: 7)
model.add(Dense(3, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# dividindo em dados de treinamento e teste
Y = pd.get_dummies(df['classe']).values
X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size = 0.2, random_state = 80)

# treinando o modelo
model.fit(X_train, Y_train, epochs=10, batch_size=80)

# avaliando o modelo
print('Acurácia do modelo - para estilo de tutoria:')
print(model.evaluate(X_test,Y_test)[1])

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

# ...
# Código anterior para treinar e avaliar o modelo
# ...

# Fazendo as previsões
y_pred = model.predict(X_test)
y_pred = np.argmax(y_pred, axis=1)
y_true = np.argmax(Y_test, axis=1)

# Criando a matriz de confusão
cm = confusion_matrix(y_true, y_pred)

# Plotando a matriz de confusão
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
plt.title("Matriz de Confusão")
plt.xlabel("Classes Preditas")
plt.ylabel("Classes Verdadeiras")
plt.show()

import numpy as np
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Sentenças
sentence1 = "Sim consegui acessar normalmente."
sentence2 = "Olá! Consegui visualizar sim"
sentence3 = "Não consegui acessar"
sentence4 = "Bom dia não consegui acessar"

# Pré-processamento das sentenças
clean_sentence1 = clean_text(sentence1)
clean_sentence2 = clean_text(sentence2)
clean_sentence3 = clean_text(sentence3)
clean_sentence4 = clean_text(sentence4)

# Tokenização das sentenças
tokenized_sentence1 = tokenizer.texts_to_sequences([clean_sentence1])
tokenized_sentence2 = tokenizer.texts_to_sequences([clean_sentence2])
tokenized_sentence3 = tokenizer.texts_to_sequences([clean_sentence3])
tokenized_sentence4 = tokenizer.texts_to_sequences([clean_sentence4])

# Pad das sequências
padded_sentence1 = pad_sequences(tokenized_sentence1, maxlen=X.shape[1])
padded_sentence2 = pad_sequences(tokenized_sentence2, maxlen=X.shape[1])
padded_sentence3 = pad_sequences(tokenized_sentence3, maxlen=X.shape[1])
padded_sentence4 = pad_sequences(tokenized_sentence4, maxlen=X.shape[1])

# Fazendo as previsões
prediction1 = model.predict(padded_sentence1)
prediction2 = model.predict(padded_sentence2)
prediction3 = model.predict(padded_sentence3)
prediction4 = model.predict(padded_sentence4)

print("Previsões:")
print("Sentença 1:", prediction1)
print("Sentença 2:", prediction2)
print("Sentença 3:", prediction3)
print("Sentença 4:", prediction4)



#Pré-processamento das sentenças
clean_sentence1 = clean_text(sentence1)
clean_sentence2 = clean_text(sentence2)
clean_sentence3 = clean_text(sentence3)
clean_sentence4 = clean_text(sentence4)

#Tokenização das sentenças
tokenized_sentence1 = tokenizer.texts_to_sequences([clean_sentence1])
tokenized_sentence2 = tokenizer.texts_to_sequences([clean_sentence2])
tokenized_sentence3 = tokenizer.texts_to_sequences([clean_sentence3])
tokenized_sentence4 = tokenizer.texts_to_sequences([clean_sentence4])

#Pad das sequências
padded_sentence1 = pad_sequences(tokenized_sentence1, maxlen=X.shape[1])
padded_sentence2 = pad_sequences(tokenized_sentence2, maxlen=X.shape[1])
padded_sentence3 = pad_sequences(tokenized_sentence3, maxlen=X.shape[1])
padded_sentence4 = pad_sequences(tokenized_sentence4, maxlen=X.shape[1])

#Fazendo as previsões
prediction1 = model.predict(padded_sentence1)
prediction2 = model.predict(padded_sentence2)
prediction3 = model.predict(padded_sentence3)
prediction4 = model.predict(padded_sentence4)

print("Previsões:")
print("Sentença 1:", prediction1)
print("Sentença 2:", prediction2)
print("Sentença 3:", prediction3)
print("Sentença 4:", prediction4)

"""# Modelos

"""

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, Conv1D, GlobalMaxPooling1D, LSTM, Dense

# Definir os parâmetros
embedding_dim = 128
num_classes = 4

# Criar o modelo
model = Sequential()

# Adicionar camadas comuns a todos os códigos
model.add(Embedding(5000, embedding_dim, input_length=X.shape[1]))

# Escolher uma arquitetura
# Opção 1: Conv1D e GlobalMaxPooling1D
model.add(Conv1D(filters=64, kernel_size=3, activation='relu'))
model.add(GlobalMaxPooling1D())

# Opção 2: LSTM
# model.add(LSTM(128, dropout=0.2, recurrent_dropout=0.2))

# Adicionar camadas específicas do código
# Opção 1: Código 1
# model.add(Dense(128, activation='relu'))

# Opção 2: Código 2
# Nenhuma camada adicional necessária

# Opção 3: Código 3
# model.add(Dense(128, activation='relu'))
# model.add(Dense(128, activation='relu'))

# Opção 4: Código 4
# Nenhuma camada adicional necessária

# Camada de classificação
model.add(Dense(num_classes, activation='sigmoid'))

# Compilar o modelo
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# dividindo em dados de treinamento e teste
Y = pd.get_dummies(df['classe']).values
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=80)

# treinando o modelo
model.fit(X_train, Y_train, epochs=10, batch_size=80)

# avaliando o modelo
print('Acurácia do modelo - para estilo de tutoria:')
print(model.evaluate(X_test, Y_test)[1])

import matplotlib.pyplot as plt

# Treinamento do modelo
history = model.fit(X_train, Y_train, epochs=10, batch_size=80, validation_data=(X_test, Y_test))

# Obter histórico de perda e acurácia
train_loss = history.history['loss']
val_loss = history.history['val_loss']
train_acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

# Plotar gráfico de perda
plt.plot(train_loss, label='Perda de Treinamento')
plt.plot(val_loss, label='Perda de Validação')
plt.title('Gráfico de Perda')
plt.xlabel('Épocas')
plt.ylabel('Perda')
plt.legend()
plt.show()

# Plotar gráfico de acurácia
plt.plot(train_acc, label='Acurácia de Treinamento')
plt.plot(val_acc, label='Acurácia de Validação')
plt.title('Gráfico de Acurácia')
plt.xlabel('Épocas')
plt.ylabel('Acurácia')
plt.legend()
plt.show()


from sklearn.metrics import confusion_matrix
import seaborn as sns

# Fazer as previsões no conjunto de teste
Y_pred = model.predict(X_test)
Y_pred = (Y_pred > 0.5)

# Converter as previsões em rótulos de classe
Y_pred_classes = np.argmax(Y_pred, axis=1)
Y_test_classes = np.argmax(Y_test, axis=1)

# Calcular a matriz de confusão
confusion_mtx = confusion_matrix(Y_test_classes, Y_pred_classes)

# Definir os rótulos das classes
class_labels = ['tecnicas', 'outras', 'pessoais', 'sem_resposta']

# Plotar a matriz de confusão
plt.figure(figsize=(8, 6))
sns.heatmap(confusion_mtx, annot=True, fmt="d", cmap='Blues', xticklabels=class_labels, yticklabels=class_labels)
plt.title('Matriz de Confusão')
plt.xlabel('Previsão')
plt.ylabel('Real')
plt.show()
