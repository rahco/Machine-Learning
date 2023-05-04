# Sistema de Classificação

# Verificação porco ou cachorro?

# features (1 sim, 0 não)

# Características avaliadas para classificação:
# pelo longo? 
# perna curta?
# faz auau?

porco1 = [0, 1, 0]
porco2 = [0, 1, 1]
porco3 = [1, 1, 0]

cachorro1 = [0, 1, 1]
cachorro2 = [1, 0, 1]
cachorro3 = [1, 1, 1]

# 1 => porco, 0 => cachorro
treino_x = [porco1, porco2, porco3, cachorro1, cachorro2, cachorro3]
treino_y = [1,1,1,0,0,0] # labels / etiqueta

# Importando o LinearSVC para criar o modelo.
from sklearn.svm import LinearSVC

model = LinearSVC()

# Inserindo os dados no modelo para o aprendizado supervisionado.
model.fit(treino_x, treino_y)

# Testando a predição do modelo criado.
animal_misterioso = [1,1,1]
model.predict([animal_misterioso])

# Testando o modelo em uma lista de animais.
misterio1 = [1,1,1]
misterio2 = [1,1,0]
misterio3 = [0,1,1]

teste_x = [misterio1, misterio2, misterio3]
teste_y = [0, 1, 1]

# Valores esperados
previsoes = model.predict(teste_x)

# Calculando a taxa de acerto do modelo.
corretos = (previsoes == teste_y).sum()
total = len(teste_x)
taxa_de_acerto = corretos/total
print("Taxa de acerto %.2f" % (taxa_de_acerto *100))

# Obtendo a taxa de acerto utilizando o método accuracy_score do sklearn
from sklearn.metrics import accuracy_score

taxa_de_acerto = accuracy_score(teste_y, previsoes)
print("Taxa de acerto %.2f" % (taxa_de_acerto *100))