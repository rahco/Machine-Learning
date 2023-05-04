
# Importando a base de estudo

import pandas as pd

resultados_exames = pd.read_csv("exames.csv")
resultados_exames

# Separando o BD em base de teste e treino

from sklearn.model_selection import train_test_split
from numpy import random

SEED = 123143
random.seed(SEED)

valores_exames = resultados_exames.drop(columns=["id", "diagnostico"])
diagnostico = resultados_exames.diagnostico

treino_x, teste_x, treino_y, teste_y = train_test_split(valores_exames, diagnostico)

# Verificando a quantidade de registros vazios no BD

resultados_exames.isnull().sum()

# Na coluna exame_33 detectamos que 73,6% dos registros estão vazios.
# Tendo em sua maior parte resultados vazios, optamos por remover a variável do modelo.

419/569

# Recriando o modelo com a primeira trativa, removendo a coluna exame_33 da variável valores_exames
# e reprocessando o modelo.

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from numpy import random

SEED = 123143
random.seed(SEED)

valores_exames = resultados_exames.drop(columns=["id", "diagnostico"])
diagnostico = resultados_exames.diagnostico
valores_exames_v1 = valores_exames.drop(columns="exame_33")

treino_x, teste_x, treino_y, teste_y = train_test_split(valores_exames_v1, 
                                                        diagnostico,
                                                        test_size = 0.3)

classificador = RandomForestClassifier(n_estimators = 100)
classificador.fit(treino_x, treino_y)
print("Resultado da classificação %.2f%%" % (classificador.score(teste_x, teste_y)* 100))

# Criando um modelo base de verificação simples para comparar com o resultado de classificação anteriormente obtido

from sklearn.dummy import DummyClassifier

SEED = 123143
random.seed(SEED)

classificador_bobo = DummyClassifier(strategy = "most_frequent")
classificador_bobo.fit(treino_x, treino_y)
print("Resultado da classificação boba %.2f%%" % (classificador_bobo.score(teste_x, teste_y)* 100))

Sendo o resultado da classificação de 92,40% muito superior ao obtido através da classificação simples que foi de 66,67%, entendemos que o modelo de classificação está satisfatório.

# 02 - Avançando e explorando dados


# Para analisar o impacto de cada variável na determinação da variável dependente, plotamos o gráfico
# de violino.
# Nele identificamos variáveis que possui 0 influência como o exame_4 que possui valores constantes.

# Na função adicionamos o método pd.melt para criar uma tabela matriz com as colunas diagnostico, exames e valores,
# a função replica os valores de diagnóstico e exames para agrupar na coluna valores os resultados.

import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

padronizador = StandardScaler()
padronizador.fit(valores_exames_v1)
valores_exames_v2 = padronizador.transform(valores_exames_v1)
valores_exames_v2 = pd.DataFrame(data = valores_exames_v2,
                                columns = valores_exames_v1.keys())

dados_plot = pd.concat([diagnostico, valores_exames_v2.iloc[:,0:10]], axis = 1)
dados_plot = pd.melt(dados_plot, id_vars="diagnostico", 
                 var_name="exames",
                 value_name="valores")

plt.figure(figsize=(5,5))

sns.violinplot(x = "exames", y = "valores", 
               hue = "diagnostico", data = dados_plot,
              split = True)

plt.xticks(rotation = 90)

# Analisando os valores constantes do exame_4.

valores_exames_v1.exame_4

# Criando uma função para plotar o gráfico dos demais exames.

def grafico_violino(valores, inicio, fim):

    dados_plot = pd.concat([diagnostico, valores.iloc[:,inicio:fim]], axis = 1)
    dados_plot = pd.melt(dados_plot, id_vars="diagnostico", 
                         var_name="exames",
                         value_name="valores")
    plt.figure(figsize=(10,10))
    sns.violinplot(x = "exames", y = "valores", hue = "diagnostico", 
                    data = dados_plot, split = True)
    plt.xticks(rotation = 90)

grafico_violino(valores_exames_v2, 10, 21)

# No exame_29 observamos o mesmo comportamento de variável constante.

grafico_violino(valores_exames_v2, 21, 32)

# Atualizando a função classificar removendo do conjunto de dados as variáveis constantes que
# não inteferem na definição da variável dependete.

valores_exames_v3 = valores_exames_v2.drop(columns=["exame_29","exame_4"])

def classificar(valores):
    SEED = 123143
    random.seed(SEED)

    treino_x, teste_x, treino_y, teste_y = train_test_split(valores, 
                                                            diagnostico, 
                                                            test_size = 0.3)

    classificador = RandomForestClassifier(n_estimators = 100)
    classificador.fit(treino_x, treino_y)
    classificador.fit(treino_x, treino_y)
    print("Resultado da classificação %.2f%%" % (classificador.score(teste_x, teste_y)* 100))

classificar(valores_exames_v3)

Com o resultado 93,57% superior ao anterior de 92,40%, confirmamos que a remoção das variáveis constantes do modelo favorecel ao modelo.

#03 - Dados correlacionados


# Gerando a matriz de correlação entre as variáveis.

valores_exames_v3.corr()

# Devido a dificuldade de se analisar a carrelação em uma matriz, criamos um gráfico de calor para tornar mais simples o processo
# de análise entre as correlações.

matriz_correlacao = valores_exames_v3.corr()

plt.figure(figsize = (12, 10))

sns.heatmap(matriz_correlacao, annot = True, fmt = ".1f")

# Gerando uma nova matriz de correlação, filtrando apenas valores > 0.99.

matriz_correlacao_v1 =  matriz_correlacao[matriz_correlacao>0.99]
matriz_correlacao_v1

# Utilizando a função sum para agregar e remover valores NaN.

matriz_correlacao_v2 = matriz_correlacao_v1.sum()
matriz_correlacao_v2

# Filtrando variáveis com alto valor de correlação para remoção no modelo.

variaveis_correlacionadas = matriz_correlacao_v2[matriz_correlacao_v2>1]
variaveis_correlacionadas

# Removendo as 4 variáveis de alto valor de correlação.

valores_exames_v4 = valores_exames_v3.drop(columns=variaveis_correlacionadas.keys())
valores_exames_v4.head()

# Verificando o resultado da nova classificação.

classificar(valores_exames_v4)

# Testando o valor da classificação removendo apenas uma das variáveis que possuem alta correlação mútua.

valores_exames_v5 = valores_exames_v3.drop(columns=["exame_3", "exame_24"])
classificar(valores_exames_v5)

Com os ajustes observamos que a remoção de ao menos uma das variáveis de alta correlação ajudou a melhor os resultados da classificação de 93,57% para 94,15%.

# 04 - Automatizando a seleção


# Com o método SelectKBest do sklearn conseguimos filtrar de forma simplificada,
# as k melhores variáveis a serem mantidas no nosso modelo.

# Na aplicação da função fizemos o teste da classficação selecionando apenas as
# 5 melhores variáveis.

from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2

selecionar_kmelhores = SelectKBest(chi2, k=5)
selecionar_kmelhores

# Gerando os novos valores de treino considerando um filtro das variáveis com apenas as
# 5 melhores variáveis.

# No processo criamos a valores_exames_v6 que repete as trativas feitas de remoção até aqui.

SEED = 1234
random.seed(SEED)


valores_exames_v6 = valores_exames_v1.drop(columns=["exame_4", "exame_29", "exame_3", "exame_24"])

treino_x, teste_x, treino_y, teste_y = train_test_split(valores_exames_v6, diagnostico, test_size = 0.3)
selecionar_kmelhores.fit(treino_x, treino_y)
treino_kbest = selecionar_kmelhores.transform(treino_x)
teste_kbest = selecionar_kmelhores.transform(teste_x)

# Com o kbest temos o novo valore de classificação considerando apenas as 5 variávies principais.

# O resultado foi satisfatório partindo do principio que reduziríamos o volume de exames de 33 para
# 5 para ter um sucesso de 92,40% na definição do resultado.

classificador = RandomForestClassifier(n_estimators = 100, random_state=1234)
classificador.fit(treino_kbest, treino_y)

print("Resultado da classificação %.2f%%" % (classificador.score(teste_kbest, teste_y)* 100))

# Para verificar se o modelo está sendo acertivo tanto para resultados benignos quanto para malignos,
# criamos uma matriz de confução utilizando o método confusion_matrix do sklearn.

# Para gerar os resultados preditos utilizamos o método .predict considerando o modelo das variáveis
# filtradas utilizando o método kbest.

from sklearn.metrics import confusion_matrix

matriz_confusao = confusion_matrix(teste_y, classificador.predict(teste_kbest))
matriz_confusao

# Para otimizar a visualização plotamos o gráfico da matriz de confusão, onde observamos
# uma acurária de 95,2% (100/105) para resultados benignos e de 87,9% (58/66) para resultados
# malignos.

plt.figure(figsize = (5, 4))
sns.set(font_scale = 2)
sns.heatmap(matriz_confusao, annot = True, fmt = "d").set(xlabel = "Predição", ylabel = "Real")

# Utilizando o método RFE do sklearn para de forma automática selecionar as k melhores variáveis.

# De forma similar ao SelectKBest o método seleciona os valores dentro do DF.

# No modelo foi utilizaodo o df valores_exames_v6 (que possui a prévia remoção de algumas variáveis)
# no train_test_split.

from sklearn.feature_selection import RFE

SEED = 1234
random.seed(SEED)

treino_x, teste_x, treino_y, teste_y = train_test_split(valores_exames_v6,
                                                       diagnostico,
                                                       test_size = 0.3)

classificador = RandomForestClassifier(n_estimators=100, random_state = 1234)
classificador.fit(treino_x, treino_y)


selecionador_rfe = RFE(estimator = classificador, n_features_to_select = 5, step = 1)
selecionador_rfe.fit(treino_x, treino_y)
treino_rfe = selecionador_rfe.transform(treino_x)
teste_rfe = selecionador_rfe.transform(teste_x)
classificador.fit(treino_rfe, treino_y)

matriz_confusao = confusion_matrix(teste_y, classificador.predict(teste_rfe))
plt.figure(figsize = (5, 4))
sns.set(font_scale = 2)
sns.heatmap(matriz_confusao, annot = True, fmt = "d").set(xlabel = "Predição", ylabel = "Real")

print("Resultado da classificação %.2f%%" % (classificador.score(teste_rfe, teste_y)* 100))

Como resultado observamos que o método SelectKBest se saiu melhor em relação ao RFE, tando na acurária geral SelectKBest=92,4% e RFE=90,06% quanto nas análises individuais, pois na determinação do tipo maligno o RFE teve uma performance de 81,8% enquanto o resultado do SelectKBest foi de 87,9%.

# 05 - Visualizando os dados no plano


# Utilizando o método RFECV conseguimos um novo modelo estimado através da acurácia,
# onde de forma automática é selecionada a quantidade de features e quais features,
# são mais apropriadas para o melhor resultado na estimação de Y.

from sklearn.feature_selection import RFECV

SEED = 1234
random.seed(SEED)

treino_x, teste_x, treino_y, teste_y = train_test_split(valores_exames_v6,
                                                       diagnostico,
                                                       test_size = 0.3)

classificador = RandomForestClassifier(n_estimators=100, random_state = 1234)
classificador.fit(treino_x, treino_y)


selecionador_rfecv = RFECV(estimator = classificador, cv = 5, scoring = "accuracy", step = 1)
selecionador_rfecv.fit(treino_x, treino_y)
treino_rfecv = selecionador_rfecv.transform(treino_x)
teste_rfecv = selecionador_rfecv.transform(teste_x)
classificador.fit(treino_rfecv, treino_y)

matriz_confusao = confusion_matrix(teste_y, classificador.predict(teste_rfecv))
plt.figure(figsize = (5, 4))
sns.set(font_scale = 2)
sns.heatmap(matriz_confusao, annot = True, fmt = "d").set(xlabel = "Predição", ylabel = "Real")

print("Resultado da classificação %.2f%%" % (classificador.score(teste_rfecv, teste_y)* 100))

# Para tentarmos plotar um scatter plot dos resultados, inicialmente recriamos o modelo,
# selecionando apenas duas features.

from sklearn.feature_selection import RFE

SEED = 1234
random.seed(SEED)

treino_x, teste_x, treino_y, teste_y = train_test_split(valores_exames_v6,
                                                       diagnostico,
                                                       test_size = 0.3)

classificador = RandomForestClassifier(n_estimators=100, random_state = 1234)
classificador.fit(treino_x, treino_y)


selecionador_rfe = RFE(estimator = classificador, n_features_to_select = 2, step = 1)
selecionador_rfe.fit(treino_x, treino_y)
treino_rfe = selecionador_rfe.transform(treino_x)
teste_rfe = selecionador_rfe.transform(teste_x)
classificador.fit(treino_rfe, treino_y)

matriz_confusao = confusion_matrix(teste_y, classificador.predict(teste_rfe))
plt.figure(figsize = (5, 4))
sns.set(font_scale = 2)
sns.heatmap(matriz_confusao, annot = True, fmt = "d").set(xlabel = "Predição", ylabel = "Real")

print("Resultado da classificação %.2f%%" % (classificador.score(teste_rfe, teste_y)* 100))

valores_exames_v7 = selecionador_rfe.transform(valores_exames_v6)
valores_exames_v7.shape

# Com duas features selecionadas pudemos plotar o gráfico, porém utilizando essa abordagem,
# ignoramos todas as demais features que são relevantes para determinação da variável dependete.

import seaborn as sns 

plt.figure(figsize=(10, 6))
sns.scatterplot(x = valores_exames_v7[:,0], y = valores_exames_v7[:,1], hue = diagnostico)

# Utilizando o decomposition do sklearn decompomos as features para um modelo de 2 dimensões,
# permitindo visualizar o scatter dos resultados sem perder a visão do conjunto como um todo.

from sklearn.decomposition import PCA

pca = PCA(n_components = 2)
valores_exames_v8 = pca.fit_transform(valores_exames_v5)

plt.figure(figsize=(10, 6))
sns.scatterplot(x = valores_exames_v8[:,0], y = valores_exames_v8[:,1], hue = diagnostico)

# Outro método para visualização do scatterplot de muitas dimensões é utilizando o manifold.

from sklearn.manifold import TSNE

tsne = TSNE(n_components = 2)
valores_exames_v9 = tsne.fit_transform(valores_exames_v5)

plt.figure(figsize=(10, 6))
sns.scatterplot(x = valores_exames_v9[:,0], y = valores_exames_v9[:,1], hue = diagnostico)