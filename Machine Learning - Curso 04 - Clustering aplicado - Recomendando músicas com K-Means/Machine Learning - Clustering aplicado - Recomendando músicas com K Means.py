
# Aula 1 - Introdução

## Aula 1.3 Dicionário dos dados

[Spotify API](https://developer.spotify.com/documentation/web-api/reference/#/operations/get-several-audio-features)

* `Acousticness/Acústica:` Variável numérica, medida de confiança de 0,0 a 1,0 se a faixa é acústica. 1.0 representa alta confiança de que a faixa é acústica.

* `Danceability/Dançabilidade:` Variável numérica, a dançabilidade descreve o quão adequada uma faixa é para dançar com base em uma combinação de elementos musicais, incluindo tempo, estabilidade do ritmo, força da batida e regularidade geral. Um valor de 0,0 é o menos dançável e 1,0 é o mais dançável.

* `Duration_ms:`Variável numérica, a duração da trilha em milissegundos.

* `Duration_min:` Variável numérica, a duração da faixa em minutos.

* `Energy/Energia:` Variável numérica, Energia é uma medida de 0,0 a 1,0 e representa uma medida perceptiva de intensidade e atividade. Normalmente, as faixas energéticas parecem rápidas, altas e barulhentas. Por exemplo, o death metal tem alta energia, enquanto um prelúdio de Bach tem uma pontuação baixa na escala. As características perceptivas que contribuem para este atributo incluem faixa dinâmica, intensidade percebida, timbre, taxa de início e entropia geral.

* `Explicit/Explícito:` Variável categórica, se a faixa tem ou não letras explícitas (verdadeiro = sim (1); falso = não(0), não OU desconhecido).

* `Id:` O ID do Spotify para a faixa.

* `Instrumentalness/Instrumentalidade:` Variável numérica, prevê se uma faixa não contém vocais. Os sons “Ooh” e “aah” são tratados como instrumentais neste contexto. Faixas de rap ou de palavras faladas são claramente “vocais”. Quanto mais próximo o valor de instrumentalidade estiver de 1,0, maior a probabilidade de a faixa não conter conteúdo vocal. Valores acima de 0,5 destinam-se a representar faixas instrumentais, mas a confiança é maior à medida que o valor se aproxima de 1,0.

* `Key/Chave:`Variável numérica, a chave geral estimada da faixa. Os inteiros são mapeados para pitchs usando a notação padrão de Pitch Class. Por exemplo. 0 = C, 1 = C#/Db, 2 = D, e assim por diante. Se nenhuma chave foi detectada, o valor é -1.

* `Liveness/ Ao vivo:` Variável numérica, detecta a presença de um público na gravação. Valores mais altos de vivacidade representam uma probabilidade maior de que a faixa tenha sido executada ao vivo. Um valor acima de 0,8 fornece uma forte probabilidade de que a faixa esteja ativa.

* `Loudness/ Volume em dB:` Variável numérica, volume geral de uma faixa em decibéis (dB). Os valores de volume são calculados em média em toda a faixa e são úteis para comparar o volume relativo das faixas. A sonoridade é a qualidade de um som que é o principal correlato psicológico da força física (amplitude). Os valores típicos variam entre -60 e 0 db.

* `Mode/ Modo:` Variável numérica, o modo indica a modalidade (maior ou menor) de uma faixa, o tipo de escala da qual seu conteúdo melódico é derivado. Maior é representado por 1 e menor é 0.

* `Popularity/Popularidade:` Variável numérica, a popularidade de uma faixa é um valor entre 0 e 100, sendo 100 o mais popular. A popularidade é calculada por algoritmo e é baseada, em grande parte, no número total de execuções que a faixa teve e quão recentes são essas execuções.

* `Speechiness/Fala:` Variável numérica, a fala detecta a presença de palavras faladas em uma faixa. Quanto mais exclusivamente falada a gravação (por exemplo, talk show, audiolivro, poesia), mais próximo de 1,0 o valor do atributo. Valores acima de 0,66 descrevem faixas que provavelmente são feitas inteiramente de palavras faladas. Valores entre 0,33 e 0,66 descrevem faixas que podem conter música e fala, seja em seções ou em camadas, incluindo casos como música rap. Os valores abaixo de 0,33 provavelmente representam músicas e outras faixas que não são de fala.

* `Tempo:` Variável numérica, Tempo estimado geral de uma faixa em batidas por minuto (BPM). Na terminologia musical, tempo é a velocidade ou ritmo de uma determinada peça e deriva diretamente da duração média da batida.

* `Valence/Valência:` Variável numérica, Medida de 0,0 a 1,0 descrevendo a positividade musical transmitida por uma faixa. Faixas com alta valência soam mais positivas (por exemplo, feliz, alegre, eufórica), enquanto faixas com baixa valência soam mais negativas (por exemplo, triste, deprimida, irritada).

* `Year/Ano:` Ano em que a música foi lançada.

## Aula 1.4 Analise dos dados

**Bases udadas**

* [Dados gerais de músicas](https://raw.githubusercontent.com/sthemonica/music-clustering/main/Dados/Dados_totais.csv)

* [Dados relacionados à gêneros](https://raw.githubusercontent.com/sthemonica/music-clustering/main/Dados/data_by_genres.csv)

* [Dados relacionados aos anos](https://raw.githubusercontent.com/sthemonica/music-clustering/main/Dados/data_by_year.csv)


import pandas as pd
import numpy as np

dados = pd.read_csv('Dados_totais.csv')
dados_generos = pd.read_csv('data_by_genres.csv')
dados_anos = pd.read_csv('data_by_year.csv')

dados.head(2)

# Verificando a quantidade de anos no BD.

dados["year"].unique()

dados.shape

# Removendo colunas que não serão utilizadas no modelo.

dados = dados.drop(["explicit", "key", "mode"], axis=1)
dados.head(2)

dados.shape

# Verificando se existem dados nulos.

dados.isnull().sum()

dados.isna().sum()

dados_generos.head(2)

# Removendo colunas que não serão utilizadas no modelo.

dados_generos = dados_generos.drop(["key", "mode"], axis=1)
dados_generos.head(2)

dados_generos.isnull().sum()

dados_generos.isna().sum()

dados_anos.head(2)

dados_anos["year"].unique()

# Limpando do BD anos que não possuem registros na base Dados.

dados_anos = dados_anos[dados_anos["year"]>=2000]
dados_anos["year"].unique()

# Removendo colunas que não serão utilizadas no modelo.

dados_anos = dados_anos.drop(["key", "mode"], axis=1)
dados_anos.head(2)

# Resetando o index que foi quebrado com a tratativa de remoção dos anos.

dados_anos.reset_index()

dados_anos.isnull().sum()

dados_anos.isna().sum()

## Aula 1.5 Análise gráfica

import plotly.express as px

# Plotando gráfico linha da evolução da variável loudness no tempo.

fig = px.line(dados_anos, x="year", y="loudness", markers= True, title='Variação do loudness conforme os anos')
fig.show()

import plotly.graph_objects as go

# Gerando o gráfico de evolução temporal das demais variáveis.

fig = go.Figure()

fig.add_trace(go.Scatter(x=dados_anos['year'], y=dados_anos['acousticness'],
                    name='Acousticness'))
fig.add_trace(go.Scatter(x=dados_anos['year'], y=dados_anos['valence'],
                    name='Valence'))
fig.add_trace(go.Scatter(x=dados_anos['year'], y=dados_anos['danceability'],
                    name='Danceability'))
fig.add_trace(go.Scatter(x=dados_anos['year'], y=dados_anos['energy'],
                    name='Energy'))
fig.add_trace(go.Scatter(x=dados_anos['year'], y=dados_anos['instrumentalness'],
                    name='Instrumentalness'))
fig.add_trace(go.Scatter(x=dados_anos['year'], y=dados_anos['liveness'],
                    name='Liveness'))
fig.add_trace(go.Scatter(x=dados_anos['year'], y=dados_anos['speechiness'],
                    name='Speechiness'))

fig.show()

# Criando a tabela de correlação entre as variáveis.

fig = px.imshow(dados.corr(), text_auto=True)
fig.show()

# Aula 2 - Clusterização por gênero

## Aula 2.1 PCA e SdandardScaler


dados_generos

dados_generos['genres'].value_counts().sum()

# Como a informação de genres não se repete, excluímos a coluna pois podemos utilizar a índice no modelo.

dados_generos1 = dados_generos.drop('genres', axis=1)
dados_generos1

Agora vamos utilizar vários conceitos em um processo de pipeline, então a primeira coisa que vamos fazer é importar o método **Pipeline** do [sklearn.pipeline](https://scikit-learn.org/stable/modules/generated/sklearn.pipeline.Pipeline.html) e esse método faz literalmente uma pipeline de machine learning, mas de uma forma automática, onde aplica sequencialmente uma lista de transformações até um resultado final. Então o que precisamos passar é o que a nossa pipeline vai fazer, como o primeiro passo e o que queremos de resultado final dela. 

Neste ponto precisamos reduzir a dimensionalidade da tabela que está com várias colunas, porém se utilizarmos um processo de redução diretamente, sem fazer a padronização dos dados na parte de pré processamento, os resultados ficarão totalmente desbalanceados, trazendo maior peso para as variáveis que têm uma amplitude maior, como por exemplo o loudness em relação às outras variáveis que compõem a música. 

Para resolver esse problema, o primeiro passo da pipeline vai ser usar o [**StandardScaler**](https://scikit-learn.org/stable/modules/preprocessing.html) para trazer essa padronização e redução de escala para que no próximo passo seja feita a redução de dimensionalidade com um método de decomposição, no nosso caso vamos escolher o 
PCA.

PCA significa Análise de componentes principais e ele trás consigo uma série de análises matemáticas que são feitas para que possamos transformar aquelas milhares de colunas que temos em uma quantidade menor, com um valor n que escolhermos, porém, quanto mais colunas a gente tem no dataset original e menos colunas queremos no dataset final, o aprendizado depois pode ser prejudicado.

Na parte **n_components** podemos colocar a quantidade de % de explicação que queremos que o algoritmo tenha no final, como por exemplo 0.3, que seria 30%, ou um valor como por exemplo um valor X de colunas.

Depois de feita a pipeline, vamos salvar em um arquivo chamado projection, com as colunas x e y, que são as posições dos pontos na cluster.


from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

SEED = 1224
np.random.seed(1224)

pca_pipeline = Pipeline([('scaler', StandardScaler()), ('PCA', PCA(n_components=2, random_state=SEED))])

genre_embedding_pca = pca_pipeline.fit_transform(dados_generos1)
projection = pd.DataFrame(columns=['x', 'y'], data=genre_embedding_pca)

# Dados padronizados com o standardscaler.

projection

## Aula 2.2 K-Means

from sklearn.cluster import KMeans

# Utilizando o método kmeans para clusterizar de forma não supervisionada o BD.

kmeans_pca = KMeans(n_clusters=5, verbose=True, random_state=SEED)

kmeans_pca.fit(projection)

dados_generos['cluster_pca'] = kmeans_pca.predict(projection)
projection['cluster_pca'] = kmeans_pca.predict(projection)

projection

# Adicionando a coluna genres no DF projection junto aos clusters.

projection['generos'] = dados_generos['genres']

projection

## Aula 2.3 Plotando a clustering

# Gerando o gráfico scatter dos clusters.

fig = px.scatter(
   projection, x='x', y='y', color='cluster_pca', hover_data=['x', 'y', 'generos'])
fig.show()

# Verificando a porcentagem do quanto os dados estão sendo explicados com a clusteirização realizada.

pca_pipeline[1].explained_variance_ratio_.sum()

# Verificando a quantidade de colunas que estão sendo explicadas com o modelo.

pca_pipeline[1].explained_variance_.sum()

# Aula 3 - Clusterização por música

## Aula 3.1 Redução de dimensionalidade com PCA


dados.head()

# Verificando a quantidade de artistas.

dados['artists'].value_counts()

# Verificando a quantidade de músicas.

dados['artists_song'].value_counts()

Ao trabalharmos com variáveis categóricas em modelos de machine learning, deve-se realizar sua transformação para variáveis binárias, também conhecidas como variáveis dummies. Esse processo denomina-se codificação “one-hot” ou codificação distribuída que transforma seus dados categóricos em uma representação vetorial binária, ou seja, para cada valor único em uma coluna, uma nova coluna é criada. Os valores nesta nova coluna são representados como 1s e 0s.

Essa codificação pode ser realizada de forma automática com o método get_dummies() da biblioteca pandas ou ainda com o método OneHotEncoder da biblioteca sklearn. Mas qual devemos utilizar?

Para modelos de machine learning, é sempre preferível utilizar o OneHotEncoder, caso seja apenas uma analise mais simples pode-se utilizar o get_dummies(). A grande vantagem é que o sklearn cria um transformador que pode ser aplicado a um novo conjunto de dados que tenham as mesmas features categoricas, logo pode-se utilizar esse transformador em um pipeline do sklearn para tornar o processo ainda mais automatizado.

Outra vantagem do OneHotEncoder é que com esse método é possível lidar com categorias desconhecidas durante a transformação nativa através do parâmetro handle_unknown, caso fosse utilizado o método get_dummies() isso não seria possível.

Diante disso, podemos concluir que os dois métodos retornam os mesmos resultados, então sua utilização depende de qual contexto está sendo trabalhado. Caso o processo seja para criação de modelos de machine learning, é preferível utilizar o OneHotEncoder e para atividades de análise de dados pode-se utilizar o get_dummies sem perda produtividade.


# Utilizando o método OneHotEncoder para criar colunas no DF sinalizando um dummy se o o artista é 1 na coluna ou 0.
# No processo chamamos a coluna com o nome do artista e removemos a coluna artists.

from sklearn.preprocessing import OneHotEncoder

ohe = OneHotEncoder(dtype=int)
colunas_ohe = ohe.fit_transform(dados[['artists']]).toarray()
dados2 = dados.drop('artists', axis=1)

dados_musicas_dummies = pd.concat([dados2, pd.DataFrame(colunas_ohe, columns=ohe.get_feature_names_out(['artists']))], axis=1)
dados_musicas_dummies

dados.shape

dados_musicas_dummies.shape

# Utilizando o PCA para reduzir o volume de features considerando 70% de explicação dos dados.

# No processo removemos as colunas ['id','name','artists_song'] do DF pois o método PCA não aceita strings.

pca_pipeline = Pipeline([('scaler', StandardScaler()), ('PCA', PCA(n_components=0.7, random_state=SEED))])


music_embedding_pca = pca_pipeline.fit_transform(dados_musicas_dummies.drop(['id','name','artists_song'], axis=1))
projection_m = pd.DataFrame(data=music_embedding_pca)

# Como resultado saimos de um DF de 890 colunas para um com 612.

pca_pipeline[1].n_components_

## Aula 3.2 Aplicação do cluster com K-Means

# Utilizando o KMeans para gerar a clusteirização em 50 do DF.

kmeans_pca_pipeline = KMeans(n_clusters=50, verbose=False, random_state=SEED)

kmeans_pca_pipeline.fit(projection_m)

dados['cluster_pca'] = kmeans_pca_pipeline.predict(projection_m)
projection_m['cluster_pca'] = kmeans_pca_pipeline.predict(projection_m)

# Retornando as colunas de artists e artists_song no DF.

projection_m['artist'] = dados['artists']
projection_m['song'] = dados['artists_song']

projection_m

## Aula 3.3 Analisando o cluster

# Plotando o gráfico scatter considerando apenas as 2 primeiras colunas geradas pelo PCA,
# que representam as duas colunas que melhor explicam os dados no modelo.

fig = px.scatter(
   projection_m, x=0, y=1, color='cluster_pca', hover_data=[0, 1, 'song'])
fig.show()

# Gerando a visão 3d do scatter.

fig = px.scatter_3d(
   projection_m, x=0, y=1, z=2, color='cluster_pca',hover_data=['song'])
fig.update_traces(marker_size = 2)
fig.show()

# Confirmando a explicação dos dados em 70%.

pca_pipeline[1].explained_variance_ratio_.sum()

# Com o modelo explicamos todas as colunas do modelo.

pca_pipeline[1].explained_variance_.sum()

# Aula 4 - Sistemas de Recomendação

## Aula 4.1 Recomendação da música


nome_musica = 'Ed Sheeran - Shape of You'

# Utilizando o recomendador, para trazer as top 10 músicas mais recomendadas conforme variável nome_musica.

# No modelo calulamos a distância euclidianda entreas músicas para ordenar as mais recomendadas.

from pandas.core.dtypes.cast import maybe_upcast
from sklearn.metrics.pairwise import euclidean_distances

cluster = list(projection_m[projection_m['song']== nome_musica]['cluster_pca'])[0]
musicas_recomendadas = projection_m[projection_m['cluster_pca']== cluster][[0, 1, 'song']]
x_musica = list(projection_m[projection_m['song']== nome_musica][0])[0]
y_musica = list(projection_m[projection_m['song']== nome_musica][1])[0]

#distâncias euclidianas
distancias = euclidean_distances(musicas_recomendadas[[0, 1]], [[x_musica, y_musica]])
musicas_recomendadas['id'] = dados['id']
musicas_recomendadas['distancias']= distancias
recomendada = musicas_recomendadas.sort_values('distancias').head(10)
recomendada

## Aula 4.2 Biblioteca Spotipy

[Spotify for Developers](https://developer.spotify.com/dashboard/)


!pip install spotipy

import spotipy
from spotipy.oauth2 import SpotifyOAuth
from spotipy.oauth2 import SpotifyClientCredentials

**ATENÇÃO!**

Antes de rodar essa parte do código, você precisa fazer uma conta na API do Spotify e gerar suas próprias **client_id** e **client_secret**


scope = "user-library-read playlist-modify-private"
OAuth = SpotifyOAuth(
        scope=scope,         
        redirect_uri='http://localhost:5000/callback',
        client_id = 'ee1423998148499f8e961863395b7d75',
        client_secret = '80590718ccf343cea45d19a2caf6486d')

client_credentials_manager = SpotifyClientCredentials(client_id = 'ee1423998148499f8e961863395b7d75',client_secret = '80590718ccf343cea45d19a2caf6486d')
sp = spotipy.Spotify(client_credentials_manager = client_credentials_manager)

## Aula 4.3 Imagem do álbum

dados.head(1)

# Visualizando a capa do albúm da música selecionada para teste do recomendados.

import matplotlib.pyplot as plt
from skimage import io

#achando o ID
nome_musica = 'Ed Sheeran - Shape of You'
id = dados[dados['artists_song']== nome_musica]['id'].iloc[0]

# na API
track = sp.track(id)
url = track["album"]["images"][1]["url"]
name = track["name"]

# Mexendo com a imagem
image = io.imread(url)
plt.imshow(image)
plt.xlabel(name, fontsize = 10)
plt.show()

# Aula 5 - Recomendador

## Aula 5.1 Buscando os dados da playlist


# Criando um função para coleta dos nomes e url de imagem das músicas
# selecionadas pelo recomendador.

def recommend_id(playlist_id):
  url = []
  name = []
  for i in playlist_id:
        track = sp.track(i)
        url.append(track["album"]["images"][1]["url"])
        name.append(track["name"])
  return name, url

# Salvando o resultado da função em variáveis.

name, url = recommend_id(recomendada['id'])

name, url

## Aula 5.2 Gerando as imagens da playlist

# Gerando a função para plotagem das imagens das músicas recomendadas.

def visualize_songs(name, url):

    plt.figure(figsize=(12,6))
    columns = 5

    for i, u in enumerate(url): 
        # define o ax como o subplot, com a divisão que retorna inteiro do número urls pelas colunas + 1 (no caso, 6)
        ax = plt.subplot(len(url) // columns + 1, columns, i + 1)

        # Lendo a imagem com o Scikit Image
        image = io.imread(u)

        # Mostra a imagem
        plt.imshow(image)

        # Para deixar o eixo Y invisível 
        ax.get_yaxis().set_visible(False)

        # xticks define o local que vamos trocar os rótulos do eixo x, nesse caso, deixar os pontos de marcação brancos
        plt.xticks(color = 'w', fontsize = 0.1)

        # yticks define o local que vamos trocar os rótulos do eixo y, nesse caso, deixar os pontos de marcação brancos
        plt.yticks(color = 'w', fontsize = 0.1)

        # Colocando o nome da música no eixo x
        plt.xlabel(name[i], fontsize = 6)

        # Faz com que todos os parâmetros se encaixem no tamanho da imagem definido
        plt.tight_layout(h_pad=0.7, w_pad=0)

        # Ajusta os parâmetros de layout da imagem.
        # wspace = A largura do preenchimento entre subparcelas, como uma fração da largura média dos eixos.
        # hspace = A altura do preenchimento entre subparcelas, como uma fração da altura média dos eixos.
        plt.subplots_adjust(wspace=None, hspace=None)

        # Remove os ticks - marcadores, do eixo x, sem remover o eixo todo, deixando o nome da música.
        plt.tick_params(bottom = False)

        # Tirar a grade da imagem, gerada automaticamente pelo matplotlib
        plt.grid(visible=None)
    plt.show()

# Testando a visualização das função.

visualize_songs(name, url)

## Aula 5.3 Fazendo uma função final

# Gerando a função do recomendador.

def recomendador(nome_musica):

## Calculando as distâncias
  cluster = list(projection_m[projection_m['song']== nome_musica]['cluster_pca'])[0]
  musicas_recomendadas = projection_m[projection_m['cluster_pca']== cluster][[0, 1, 'song']]
  x_musica = list(projection_m[projection_m['song']== nome_musica][0])[0]
  y_musica = list(projection_m[projection_m['song']== nome_musica][1])[0]
  distancias = euclidean_distances(musicas_recomendadas[[0, 1]], [[x_musica, y_musica]])
  musicas_recomendadas['id'] = dados['id']
  musicas_recomendadas['distancias'] = distancias
  recomendada = musicas_recomendadas.sort_values('distancias').head(10)

  # ## Acessando os dados de cada música com a biblioteca Spotipy (nome e imagem)
  playlist_id = recomendada['id']

  url = []
  name = []
  for i in playlist_id:
      track = sp.track(i)
      url.append(track["album"]["images"][1]["url"])
      name.append(track["name"])

# ## Plotando as figuras
  plt.figure(figsize=(12,6))
  columns = 5
  for i, u in enumerate(url):
      ax = plt.subplot(len(url) // columns + 1, columns, i + 1)
      image = io.imread(u)
      plt.imshow(image)
      ax.get_yaxis().set_visible(False)
      plt.xticks(color = 'w', fontsize = 0.1)
      plt.yticks(color = 'w', fontsize = 0.1)
      plt.xlabel(name[i], fontsize = 6)
      plt.tight_layout(h_pad=0.7, w_pad=0)
      plt.subplots_adjust(wspace=None, hspace=None)
      plt.grid(visible=None)
      plt.tick_params(bottom = False)
  plt.show()

# Testando o plot do recomendador.

recomendador('Ed Sheeran - Shape of You')

# Novo teste do recomendador.

recomendador('Taylor Swift - Blank Space')