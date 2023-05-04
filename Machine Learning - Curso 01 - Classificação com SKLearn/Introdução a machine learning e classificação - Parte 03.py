# Modelo de classificação em BD

Criando e validando modelo para verificar a possibilidade de conclusão de um projeto de website baseado nas expectativas de horas e preço.

# Obtendo o BD.
import pandas as pd

# Colunas de produção de um website, sendo:
# > unfinished = se o projeto foi ou não finalizado
# > expected_hours = expectativa de horas para finalizar o projeto
# > price = preço cobrado

uri = "https://gist.githubusercontent.com/guilhermesilveira/1b7d5475863c15f484ac495bd70975cf/raw/16aff7a0aee67e7c100a2a48b676a2d2d142f646/projects.csv"
dados = pd.read_csv(uri)
dados.head()

# Renomeando o nome das colunas.

a_renomear = {
    'expected_hours' : 'horas_esperadas',
    'price' : 'preco',
    'unfinished' : 'nao_finalizado'
}
dados = dados.rename(columns = a_renomear)
dados.head()

# Ajustando a coluna nao_finalizado para finalizado.

troca = {
    0 : 1,
    1 : 0
}
dados['finalizado'] = dados.nao_finalizado.map(troca)
dados.head()

# Atualizando a biblioteca do seaborn.

!pip install seaborn==0.9.0

# Plotando o gráfico de dispersão entre as variáveis x: horas_esperadas e preco.

import seaborn as sns

sns.scatterplot(x="horas_esperadas", y="preco", data=dados)

# Plotando o gráfico de dispersão horas_esperadas x preco com a legenda sendo a coluna finalizado.

sns.scatterplot(x="horas_esperadas", y="preco", hue="finalizado", data=dados)

# Analisando os gráficos de forma isolada.

sns.relplot(x="horas_esperadas", y="preco", hue="finalizado", col="finalizado", data=dados)

# Preparando as bases para criação do modelo de estimação de y.

x = dados[['horas_esperadas', 'preco']]
y = dados['finalizado']

# Criando o modelo e verificando sua acurácia.

from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score
import numpy as np

SEED = 20
np.random.seed(SEED)

treino_x, teste_x, treino_y, teste_y = train_test_split(x, y,
                                                         random_state = SEED, test_size = 0.25,
                                                         stratify = y)
print("Treinaremos com %d elementos e testaremos com %d elementos" % (len(treino_x), len(teste_x)))

modelo = LinearSVC()
modelo.fit(treino_x, treino_y)
previsoes = modelo.predict(teste_x)

acuracia = accuracy_score(teste_y, previsoes) * 100
print("A acurácia foi %.2f%%" % acuracia)

# Varificando a acurâcia se chutarmos todas as previsões sendo 1, ou seja, uma previsão de base.

# Utilizamos o percentual de acurária da previsão de base ou baseline para verificar se nosso modelo está ou não
# atendendo bem as previsões. Em resumo se espera que nosso modelo seja muito melhor que a acurácia da previsão
# gerada através do baseline.

import numpy as np
previsoes_de_base = np.ones(540)
acuracia = accuracy_score(teste_y, previsoes_de_base) * 100
print("A acurácia do algoritmo de baseline foi %.2f%%" % acuracia)

# Plotando gráfico horas_esperadas x preco do BD de texte_x, jogando como legenda os valores gerados através do teste_y.

sns.scatterplot(x="horas_esperadas", y="preco", hue=teste_y, data=teste_x)

"""### Plotando os pontos fora do gráfico para entender onde nosso modelo está errando as estimativas."""

# Definindo os pontos base/limites do gráfico.

x_min = teste_x.horas_esperadas.min()
x_max = teste_x.horas_esperadas.max()
y_min = teste_x.preco.min()
y_max = teste_x.preco.max()
print(x_min, x_max,y_min,y_max)

# Criando um array com os pontos do eixo x e y.

pixels = 100
eixo_x = np.arange(x_min, x_max, (x_max - x_min)/ pixels)
eixo_y = np.arange(y_min, y_max, (y_max - y_min)/ pixels)

# Criando os pares de localização dos pontos.

xx, yy = np.meshgrid(eixo_x, eixo_y)
pontos = np.c_[xx.ravel(), yy.ravel()]
pontos

# Obtendo as previsões sobre os pontos.

Z = modelo.predict(pontos)
Z = Z.reshape(xx.shape)
Z

# Criando o gráfico base do comparativo.

import matplotlib.pyplot as plt

plt.scatter(teste_x.horas_esperadas, teste_x.preco, c=teste_y, s=1)

# Gerando o gráfico de plot dos pontos estimados.

# No gráfico podemos confirmar que o modelo não preve de forma precisa os resultados.

plt.contourf(xx, yy, Z, alpha=0.3)
plt.scatter(teste_x.horas_esperadas, teste_x.preco, c=teste_y, s=1)

# Utilizando o SVC para teste se melhores a estimação do modelo.

from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import numpy as np

SEED = 5
np.random.seed(SEED)
treino_x, teste_x, treino_y, teste_y = train_test_split(x, y, test_size = 0.25,
                                                         stratify = y)
print("Treinaremos com %d elementos e testaremos com %d elementos" % (len(treino_x), len(teste_x)))

modelo = SVC(gamma='auto')
modelo.fit(treino_x, treino_y)
previsoes = modelo.predict(teste_x)

acuracia = accuracy_score(teste_y, previsoes) * 100
print("A acurácia foi %.2f%%" % acuracia)

# Varificando os pontos.

x_min = teste_x.horas_esperadas.min()
x_max = teste_x.horas_esperadas.max()
y_min = teste_x.preco.min()
y_max = teste_x.preco.max()

pixels = 100
eixo_x = np.arange(x_min, x_max, (x_max - x_min) / pixels)
eixo_y = np.arange(y_min, y_max, (y_max - y_min) / pixels)

xx, yy = np.meshgrid(eixo_x, eixo_y)
pontos = np.c_[xx.ravel(), yy.ravel()]

Z = modelo.predict(pontos)
Z = Z.reshape(xx.shape)

import matplotlib.pyplot as plt

plt.contourf(xx, yy, Z, alpha=0.3)
plt.scatter(teste_x.horas_esperadas, teste_x.preco, c=teste_y, s=1)

# DECISION BOUNDARY

# Sendo ainda baixo os valores de predição do modelo, ajustamos as escalas das variáveis x (horas_esperadas e preco)
# para que tenhão valores mais próximos.

# Para ajustar as escalas reescrevemos o modelo adicionando o scaler.

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

SEED = 5
np.random.seed(SEED)
raw_treino_x, raw_teste_x, treino_y, teste_y = train_test_split(x, y, test_size = 0.25,
                                                         stratify = y)
print("Treinaremos com %d elementos e testaremos com %d elementos" % (len(treino_x), len(teste_x)))

scaler = StandardScaler()
scaler.fit(raw_treino_x)
treino_x = scaler.transform(raw_treino_x)
teste_x = scaler.transform(raw_teste_x)

modelo = SVC(gamma='auto')
modelo.fit(treino_x, treino_y)
previsoes = modelo.predict(teste_x)

acuracia = accuracy_score(teste_y, previsoes) * 100
print("A acurácia foi %.2f%%" % acuracia)

# Varificando o gráfico do novo modelo ajustado.

data_x = teste_x[:,0]
data_y = teste_x[:,1]

x_min = data_x.min()
x_max = data_x.max()
y_min = data_y.min()
y_max = data_y.max()

pixels = 100
eixo_x = np.arange(x_min, x_max, (x_max - x_min) / pixels)
eixo_y = np.arange(y_min, y_max, (y_max - y_min) / pixels)

xx, yy = np.meshgrid(eixo_x, eixo_y)
pontos = np.c_[xx.ravel(), yy.ravel()]

Z = modelo.predict(pontos)
Z = Z.reshape(xx.shape)

import matplotlib.pyplot as plt

plt.contourf(xx, yy, Z, alpha=0.3)
plt.scatter(data_x, data_y, c=teste_y, s=1)