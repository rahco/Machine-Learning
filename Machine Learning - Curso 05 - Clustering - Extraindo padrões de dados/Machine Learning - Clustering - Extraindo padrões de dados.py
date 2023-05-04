
import pandas as pd

dataframe = pd.read_csv("CC GENERAL.csv")
dataframe.head()

# Removendo colunas no DF que não fazem sentido manter para a aplicação da classificação.

#CUST_ID  e TENURE

import pandas as pd

dataframe = pd.read_csv("CC GENERAL.csv")
dataframe.drop(columns=["CUST_ID", "TENURE"], inplace=True)
dataframe.head()

# Verificando a quantidade de registros isna no DF por coluna.

missing = dataframe.isna().sum()
print(missing)

# Preenchendo os valores NA com o median de cada coluna.

dataframe.fillna(dataframe.median(), inplace=True)
missing = dataframe.isna().sum()
print(missing)

# Normalizando os dados das colunas entre 0 e 1 para otimizar a clusterização. 

from sklearn.preprocessing import Normalizer
values = Normalizer().fit_transform(dataframe.values)
print(values)

# Gerando os clusters do DF.

# Ficou definido:
#> um total de 5 clusters iniciais
#> 10 repetições com valores iguais para validação da clusterização.
#> 300 como limite de iterações


from sklearn.cluster import KMeans
kmeans = KMeans(n_clusters=5, n_init=10, max_iter=300)
y_pred = kmeans.fit_predict(values)


# # Entendendo os critérios e métricas de validação

# # Calculando o Índice Silhouette

# Utilizando o metrics do sklearn para validar se a clusterização foi efetiva.

# Com valores que variam de -1 a 1 podemos determinar se a classficação foi ou não efetiva.
# Quanto mais próximo de 1 estiver o valor de silhouette mais separados estão os clusters.

# Com um resultado positivo de 0.36 entendemos como satisfatória a classificação.

from sklearn import metrics
labels = kmeans.labels_
silhouette = metrics.silhouette_score(values, labels, metric='euclidean')
print(silhouette)


# # Calculando o Índice Davies-Bouldin

dbs = metrics.davies_bouldin_score(values, labels)
print(dbs)


# # Calculando o Índice Calinski Harabasz

calinski = metrics.calinski_harabasz_score(values, labels)
print(calinski)


# # Validando os clusters

# Função criada para verificar os resultados das avaliações dos cluster,
# alterando a quantidade de cluster inseridos na função KMeans.

def clustering_algorithm(n_clusters, dataset):
    kmeans = KMeans(n_clusters=n_clusters, n_init=10, max_iter=300)
    labels = kmeans.fit_predict(dataset)
    s = metrics.silhouette_score(dataset, labels, metric='euclidean')
    dbs = metrics.davies_bouldin_score(dataset, labels)
    calinski = metrics.calinski_harabasz_score(dataset, labels)
    return s, dbs, calinski


s1, dbs1, calinski1 = clustering_algorithm(3, values)
print(s1, dbs1, calinski1)


s2, dbs2, calinski2 = clustering_algorithm(5, values)
print(s2, dbs2, calinski2)


s3, dbs3, calinski3 = clustering_algorithm(50, values)
print(s3, dbs3, calinski3)


# Como o resultado diminui a medida que aumentamos a quantidade de cluster,
# optamos por seguir com a quantidade de 5 cluster, pois apresentou um resultado satisfatório e
# com melhor desempenho na avaliação do silhouette.

# Para validarmos nossos clusters, criamos um random dataset para comparar a nossa
# clusteirização, considerando:

# Silhouette=s > melhor
# Davies-Bouldin=dbs < melhor
# Calinski Harabasz=calinski > melhor

import numpy as np
random_data = np.random.rand(8950, 16)
s, dbs, calinski = clustering_algorithm(5, random_data)
print(s, dbs, calinski)
print(s2, dbs2, calinski2)


# # Validando a estabilidade dos clusters

# Para validar a estabilidade dos clusters, dividimos o ds em 3 para verificar
# se os resultados da validação são similares em cada uma das partes.

set1, set2, set3 = np.array_split(values, 3)
s1, dbs1, calinski1 = clustering_algorithm(5, set1)
s2, dbs2, calinski2 = clustering_algorithm(5, set2)
s3, dbs3, calinski3 = clustering_algorithm(5, set3)
print(s1, dbs1, calinski1)
print(s2, dbs2, calinski2)
print(s3, dbs3, calinski3)


# # Visualizando os clusters

# Gerando gráfico de visualização dos cluster para duas variáveis.

import matplotlib.pyplot as plt
plt.scatter(dataframe['PURCHASES'], dataframe['PAYMENTS'], c=labels, s=5, cmap='rainbow')
plt.xlabel("Valor total pago")
plt.ylabel("Valor total gasto")
plt.show()


# Gráfico geral da relação entre todas as variáveis.


import seaborn as sns
dataframe["cluster"] = labels
sns.pairplot(dataframe[0:], hue="cluster")


# # Entendendo os valores dos atributos no cluster

# Gerando uma tabela geral com os valores estatísticos de todos os atributos contido no df.

dataframe.groupby("cluster").describe()


# Como temos muitos atributos no df precisamos de um método para selecionar os principais,
# para focar as analises da clusteirização por eles.

# Para iniciar a filtragem, podemos utilizar o centroid de cada atributo por cluster,
# a partir dessa ideia filtramos os atributos que possuem uma maior diferença de centroid,
# entre as classes, pois certamente esses atributos mostraram melhor as diferenças existentes,
# entre os cluster.

centroids = kmeans.cluster_centers_
print(centroids)


# Gerando os valores de variância dos atributos para seleção dos maiores.

max = len(centroids[0])
for i in range(max):
    print(dataframe.columns.values[i],"\n{:.4f}".format(centroids[:, i].var()))


# Gerando as tabelas de média dos valores dos principais atributos por cluster e a quantidade de clientes.

description = dataframe.groupby("cluster")["BALANCE", "PURCHASES", "CASH_ADVANCE", "CREDIT_LIMIT", "PAYMENTS"]
n_clients = description.size()
description = description.mean()
description['n_clients'] = n_clients
print(description)


# # Interpretando os clusters

# **CLUSTER 0**: Clientes que gastam pouco. Clientes com o maior limite. Bons pagadores. Maior número de clientes.
# 
# **CLUSTER 1**: Clientes que mais gastam. O foco deles é o saque. Piores pagadores. Boa quantidade de clientes.
# 
# **CLUSTER 2**: Clientes que gastam muito com compras. Melhores pagadores.
# 
# **CLUSTER 3**: Clientes que gastam muito com saques. Pagam as vezes.
# 
# **CLUSTER 4**: Clientes com o menor limite. Não são bons pagadores. Menor quantidade de clientes.

dataframe.groupby("cluster")["PRC_FULL_PAYMENT"].describe()

