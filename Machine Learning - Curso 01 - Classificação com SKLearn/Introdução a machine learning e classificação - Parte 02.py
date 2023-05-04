# Modelo de classificação em BD

Criando modelo para verificar a possibilidade de compra de um cliente, a partir das páginas que o mesmo acessou no site da empresa.

# Importando novo BD.
# Sendo:
# > home, how_it_works, contact = páginas acessadas
# > bought = se teve compra ou não

import pandas as pd

uri = "https://gist.githubusercontent.com/guilhermesilveira/2d2efa37d66b6c84a722ea627a897ced/raw/10968b997d885cbded1c92938c7a9912ba41c615/tracking.csv"
dados = pd.read_csv(uri)
dados.head()

# Renomeando as colunas.
 mapa = { 
    "home" : "principal",
    "how_it_works" : "como_funciona",
    "contact" : "contato",
    "bought" : "comprou"
}
dados = dados.rename(columns = mapa)

# Seperando as variáveis x e y.
x= dados[["principal","como_funciona","contato"]]
y= dados[["comprou"]]

x.head()

# Verificando o tamanho de BD.
dados.shape

# Definindo as bases de treino e teste do BD.
treino_x = x[:75]
treino_y = y[:75]
teste_x = x[75:]
teste_y = y[75:]

print("Treinaremos com %d elementos e testaremos com %d elementos" % (len(treino_x), len(teste_x)))

# Criano o modelo de previsão e verificando sua acurácia.
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score

modelo = LinearSVC()
modelo.fit(treino_x, treino_y)
previsoes = modelo.predict(teste_x)

acuracia = accuracy_score(teste_y, previsoes) * 100
print("A acurácia foi %.2f%%" % acuracia)

"""# Utilizando o train_test_split do sklearn para separar de forma mais simples e rápida o BD em base de treino e de teste."""

# Cada vez que o código é executado ele gera uma nova matriz de BD de treino e de teste.

# Para evitar que a cada vez que o teste for executado ele gere uma nova matriz de BD de treino e teste, foi estabelecido
# o modelo randômico 20 como padrão.

# O parâmetro stratify = y foi utilizado para que o o split seja feito respeitando/adequando o grau de proporcionalidade entre 
# o BD de treino e de teste. 

from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score

SEED = 20

treino_x, teste_x, treino_y, teste_y = train_test_split(x, y,
                                                         random_state = SEED, test_size = 0.25,
                                                         stratify = y)
print("Treinaremos com %d elementos e testaremos com %d elementos" % (len(treino_x), len(teste_x)))

modelo = LinearSVC()
modelo.fit(treino_x, treino_y)
previsoes = modelo.predict(teste_x)

acuracia = accuracy_score(teste_y, previsoes) * 100
print("A acurácia foi %.2f%%" % acuracia)