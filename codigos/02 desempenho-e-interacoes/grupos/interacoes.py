# Importando as bibliotecas necessárias
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
from sklearn.metrics import silhouette_score
from sklearn.neighbors import NearestNeighbors
from scipy.spatial.distance import pdist, squareform

# Função para calcular a estatística de Hopkins
def hopkins(X):
    d = X.shape[1]
    n = len(X)
    m = int(0.1 * n)  # Tamanho da amostra de 10% da base
    nbrs = NearestNeighbors(n_neighbors=1).fit(X.values)
    rand_X = np.random.uniform(np.min(X.values, axis=0), np.max(X.values, axis=0), (m, d))
    ujd = []
    wjd = []
    for j in range(0, m):
        u_dist, _ = nbrs.kneighbors([rand_X[j]], 2, return_distance=True)
        ujd.append(u_dist[0][1])
        w_dist, _ = nbrs.kneighbors([X.sample(n=1).values[0]], 2, return_distance=True)
        wjd.append(w_dist[0][1])
    H = sum(ujd) / (sum(ujd) + sum(wjd))
    return H

# Gerando dados aleatórios a partir de uma base de exemplo
np.random.seed(123)
# Substitua 'df' pelo seu DataFrame real
df = pd.DataFrame(np.random.rand(100, 10), columns=[f'Var{i}' for i in range(1, 11)])  # Exemplo
random_df = pd.DataFrame(np.random.uniform(df.min().min(), df.max().max(), df.shape), columns=df.columns)

# Padronização dos dados
scaler = StandardScaler()
df_scaled = scaler.fit_transform(df)
random_df_scaled = scaler.fit_transform(random_df)

# Análise de Componentes Principais (PCA)
pca = PCA()
df_pca = pca.fit_transform(df_scaled)
random_df_pca = pca.fit_transform(random_df_scaled)

# Visualização PCA dos dados originais
plt.figure(figsize=(10, 6))
sns.scatterplot(x=df_pca[:, 0], y=df_pca[:, 1])
plt.title('PCA - Base Estudantes Ensino Médio')
plt.show()

# Visualização PCA dos dados aleatórios
plt.figure(figsize=(10, 6))
sns.scatterplot(x=random_df_pca[:, 0], y=random_df_pca[:, 1])
plt.title('PCA - Random dados')
plt.show()

# K-means para o conjunto de dados original
km_res1 = KMeans(n_clusters=4, random_state=123).fit(df_scaled)
plt.figure(figsize=(10, 6))
sns.scatterplot(x=df_pca[:, 0], y=df_pca[:, 1], hue=km_res1.labels_)
plt.title('K-means Clusters - Base Estudantes')
plt.show()

# K-means para o conjunto de dados aleatório
km_res2 = KMeans(n_clusters=4, random_state=123).fit(random_df_scaled)
plt.figure(figsize=(10, 6))
sns.scatterplot(x=random_df_pca[:, 0], y=random_df_pca[:, 1], hue=km_res2.labels_)
plt.title('K-means Clusters - Dados Aleatórios')
plt.show()

# Clustering Hierárquico no conjunto de dados aleatório
linked = linkage(random_df_scaled, method='ward')
plt.figure(figsize=(10, 7))
dendrogram(linked, orientation='top', distance_sort='descending', show_leaf_counts=False)
plt.title('Dendrograma - Random dados')
plt.show()

# Estatística de Hopkins para o conjunto de dados original e aleatório
print('Hopkins para dados originais:', hopkins(pd.DataFrame(df_scaled)))
print('Hopkins para dados aleatórios:', hopkins(pd.DataFrame(random_df_scaled)))

# Estimando o número ideal de Clusters com o método Elbow
def plot_elbow(data):
    sse = []
    for k in range(1, 10):
        kmeans = KMeans(n_clusters=k, random_state=123).fit(data)
        sse.append(kmeans.inertia_)
    plt.figure(figsize=(8, 5))
    plt.plot(range(1, 10), sse, '-o')
    plt.xlabel('Número de Clusters')
    plt.ylabel('SSE')
    plt.title('Elbow Method')
    plt.axvline(x=4, linestyle='--', color='red')
    plt.show()

plot_elbow(df_scaled)

# Método Silhouette para estimar o número de Clusters
silhouette_scores = []
for k in range(2, 10):
    kmeans = KMeans(n_clusters=k, random_state=123).fit(df_scaled)
    score = silhouette_score(df_scaled, kmeans.labels_)
    silhouette_scores.append(score)

plt.figure(figsize=(8, 5))
plt.plot(range(2, 10), silhouette_scores, '-o')
plt.xlabel('Número de Clusters')
plt.ylabel('Silhouette Score')
plt.title('Silhouette Method')
plt.show()

# Clustering Hierárquico com método 'ward.D2'
Z = linkage(df_scaled, method='ward')
plt.figure(figsize=(10, 7))
dendrogram(Z, truncate_mode='lastp', p=12, leaf_rotation=45., leaf_font_size=15., show_contracted=True)
plt.title('Dendrograma - Clustering Hierárquico')
plt.xlabel('Índice do Cluster')
plt.ylabel('Distância')
plt.show()

# Exemplo de correlação cophenética
from scipy.cluster.hierarchy import cophenet
coph_corr, coph_dists = cophenet(Z, pdist(df_scaled))
print('Correlação Cophenética:', coph_corr)
