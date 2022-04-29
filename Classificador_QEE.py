# -*- coding: utf-8 -*-
# UNIVERSIDADE FEDERAL DE CAMPINA GRANDE
# CENTRO DE ENGENHARIA ELÉTRICA E INFORMÁTICA
# UNIDADE ACADÊMICA DE ENGENHARIA ELÉTRICA
# INTELIGÊNCIA ARTIFICIAL E CIÊNCIA DE DADOS APLICADAS A SISTEMAS ELÉTRICOS

# Carlos Augusto Soares de Oliveira Filho - 115.111.503 

# Atividade 4.2: Classificador de disturbios de QEE:

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import multilabel_confusion_matrix
from sklearn.metrics import plot_confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score

#pip install scikit-plot

import scikitplot as skplt
import matplotlib.pyplot as plt

import pandas as pd

# Carrega a base de dados
data_in = pd.read_csv('entradas_QEE.csv', header=None)
data_target = pd.read_csv('alvos_QEE.csv', header=None)

entradas = data_in.values
alvos = data_target.idxmax(1).values

# Criar as bases de treinamento e teste
Ent_trein, Ent_test, Alvo_trein, Alvo_test = train_test_split(entradas, alvos, test_size = 0.2, random_state = 10)

# Criar o modelo neural(uma camada oculta com 5 neuronios)
net = MLPClassifier(solver = 'lbfgs', max_iter = 500, hidden_layer_sizes=(100))

# Ajusta o modelo a partir da base de dados de treinamento
modelo_ajustado = net.fit(Ent_trein, Alvo_trein)

# Estima a precisao do modelo a partir da base de teste
score = modelo_ajustado.score(Ent_test, Alvo_test)
print(score)

# Calcula as previsoes do modelo a partir da base de teste
previsoes = modelo_ajustado.predict(Ent_test)
prevpb = modelo_ajustado.predict_proba(Ent_test)

precisao = accuracy_score(Alvo_test, previsoes)
print(precisao)

print(classification_report(Alvo_test, previsoes))

# Calcula a matriz de confusao - diagonal principal sao padroes estimados corretamente
confusao = confusion_matrix(Alvo_test, previsoes)
print(confusao)

# Plota matriz de confusao
opcoes_titulos = [("Matriz de confusao sem normalizacao", None),
                  ("Matriz de confusao normalizada", 'true')]
for titulo, norm in opcoes_titulos:
    disp = plot_confusion_matrix(modelo_ajustado, Ent_test, Alvo_test,
                                 cmap = plt.cm.Blues,
                                 normalize = norm)
    disp.ax_.set_title(titulo)
    
    print(titulo)
    print(disp.confusion_matrix)

plt.show()

# Plotar utilizando a biblioteca scikitplot
skplt.metrics.plot_confusion_matrix(Alvo_test, previsoes)
plt.show()

# Plotar a ROC
skplt.metrics.plot_roc(Alvo_test, prevpb)
plt.show()

skplt.metrics.plot_precision_recall(Alvo_test, prevpb)
plt.show()