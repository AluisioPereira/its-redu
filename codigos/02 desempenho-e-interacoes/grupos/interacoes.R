# Dados aleatórios gerados a partir do conjunto de dados base_grao_estudante
random_df <- apply(df, 2, function(x){runif(length(x), min(x), (max(x)))})
random_df <- as.data.frame(random_df)
# Padronizar os conjuntos de dados
df <- fundamental.scaled <- scale(df)
random_df <- scale(random_df)
#Inspeção visual dos dados
# Traçar conjunto de dados fiéis por id_turma da base_grao_estudante
fviz_pca_ind(prcomp(df), title = "PCA - Base Estudantes Ensino Médio",
habillage = fundamental$id_turma, palette = "jco",
geom = "point", ggtheme = theme_classic(),
legend = "bottom")
# Trace o df aleatório
fviz_pca_ind(prcomp(random_df), title = "PCA - Random dados",
geom = "point", ggtheme = theme_classic())
set.seed(123)
# K-means no conjunto de dados da base ensino fundamental (para os 4 Clusters)
km.res1 <- kmeans(df, 4)
fviz_Cluster(list(data = df, Cluster = km.res1$Cluster),
ellipse.type = "norm", geom = "point", stand = FALSE,
palette = "jco", ggtheme = theme_classic())
# K-means no conjunto de dados aleatório (para os 4 Clusters)
km.res2 <- kmeans(random_df, 4)
fviz_Cluster(list(data = random_df, Cluster = km.res2$Cluster),
ellipse.type = "norm", geom = "point", stand = FALSE,
palette = "jco", ggtheme = theme_classic())
# Clustering hierárquico no conjunto de dados aleatório (para os 4 Clusters)
fviz_dend(hclust(dist(random_df)), k = 4, k_colors = "jco",
as.ggplot = TRUE, show_labels = FALSE)
### MÉTODO ESTATÍSTICO
library(Clustertend)
# Estatística de Hopkins para cálculo do conjunto de dados ensino fundamental
set.seed(123)
hopkins(df, n = nrow(df)-1)


# Estatística de Hopkins para um conjunto de dados aleatório
set.seed(123)
hopkins(random_df, n = nrow(random_df)-1)
# Gerando os gráficos com os resultados
fviz_dist(dist(df), show_labels = FALSE)+
labs(title = "Base - Ensino Fundamental")
fviz_dist(dist(random_df), show_labels = FALSE)+
labs(title = "Base - Aleatória")
Estimando o número ideal de grupos (Clusters) para os dados coletados refer
entes aos estudantes do Ensino Fundamental.
# ESTIMANDO NÚMERO IDEAL DE CLUSTERS - ENSINO FUNDAMENTAL
# 1 - método Elbow
fviz_nbclust(dffun, kmeans, method = "wss") + geom_vline(xintercept = 4, linetype = 2)+ labs(sub
title = "Elbow method")
# 2 - método Silhouette
fviz_nbclust(dffun, kmeans, method = "silhouette")+labs(subtitle = "Silhouette method")
# 3 - Método Gap statistic (Estatística de lacunas)
# nboot = 50 para manter a função rápida.
# valor sugerido: nboot= 500 (interações) para a análise.
# Use verbose = FALSE para ocultar a progressão do processo de cálculo.
set.seed(123)
fviz_nbclust(dffun, kmeans, nstart = 25, method = "gap_stat", nboot = 500)+
labs(subtitle = "Gap statistic method")
set.seed(123)
km.res.fun <- kmeans(dffun, 4, nstart = 25)
print(km.res.fun)
# Calculando a média de cada variável por Clusters usando os dados originais:
aggregate(fundamental[8:17], by=list(Cluster=km.res.fun$Cluster), mean)
ddfun <- cbind(fundamental, Cluster = km.res.fun$Cluster)
head(ddfun)


# ANÁLISE DOS DADOS DO ENSINO FUNDAMENTAL - k-means
#K-MEANS E A CLUSTERING
# Atribui a base (normalizado2) a variável df ignorando as variáveis (id_estudante, id_turma, seri
e, ensino e var11)
df <- scale(fundamental[8:17])
# calculando o K-means com K = 4
km.res <- kmeans(df, 4, nstart = 25)
# Imprimindo o resultado
print(km.res)
# Calculando a média de cada variável por Clusters usando os dados originais:
aggregate(fundamental[8:17], by=list(Cluster=km.res$Cluster), mean)


# Calculando os Clusters
dd <- cbind(fundamental, Cluster = km.res$Cluster)
Validação dos Clusters formados para os dados coletados dos estudantes do
Ensino Fundamental.
# VALIDAÇÃO DOS AGRUPAMENTOS - ENSINO FUNDAMENTAL
# Calculando a matriz de dissimilaridade
res.dist.fun <- dist(dffun, method = "euclidean")
as.matrix(res.dist.fun)[1:6, 1:6]
#Calculando a árvore hierárquica
# d: uma estrutura de dissimilaridade produzida pela dist () função.
# método: O método de aglomeração (ligação) a ser usado para calcular a distância entre Cluste
rs. Os valores permitidos são “ward.D”, “ward.D2”, “single”, “complete”, “average”, “mcquitty”, “me
dian” ou “centroid”.
res.hc.fun <- hclust(d = res.dist.fun, method = "ward.D2")
# Dendograma
# cex: tamanho do rótulo
fviz_dend(res.hc.fun, cex = 0.5)
# visualização para o Dendograma de outros modos
# fviz_dend(res.hc.fun, cex = 0.5, k = 4, k_colors = "jco", type = "circular")
# require("igraph")
# fviz_dend(res.hc.fun, k = 4, k_colors = "jco", type = "phylogenic", repel = TRUE)
### Verifique a árvore do Cluster
# Calculando as distâncias cofenéticas para agrupamento hierárquico
# Calcular distância copêntica
res.coph.fun <- cophenetic(res.hc.fun)
# Correlação entre distância cofenética e a distância original
cor(res.dist.fun, res.coph.fun)
# Executa novamente o método de ligação de médias
res.hc2.fun <- hclust(res.dist.fun, method = "average")
cor(res.dist.fun, cophenetic(res.hc2.fun))
# Com o método = “average” o resultado foi: 0.9626782.
# Coeficiente de correlação mostra que o uso de um método de ligação diferente cria uma árvore
que representa as distâncias originais um pouco melhor.
#cortando a árvore em 4 grupos
grp <- cutree(res.hc.fun, k = 4)
head(grp, n = 4)
# Número de membros em cada Cluster
table(grp)
# Obtenha os nomes dos membros do Cluster 1
rownames (fundamental)[grp == 1]
#colorindo o dendograma
# Cortar em 4 grupos e colorir por grupos
fviz_dend(res.hc.fun, k = 4, # Corte em quatro grupos
 cex = 0.5, # tamanho da etiqueta
 k_colors = c("#2E9FDF", "#00AFBB", "#E7B800", "#FC4E07"),
 color_labels_by_k = TRUE, # rótulos de cores por grupos
 rect = TRUE # Adicionar retângulo ao redor dos grupos
)
#Gráfico de Clusters
fviz_Cluster(list(data = dffun, Cluster = grp),
 palette = c("#2E9FDF", "#00AFBB", "#E7B800", "#FC4E07"),
 ellipse.type = "convex", # Elipse de concentração
 repel = TRUE, # Evite o overplotting de rótulo (lento)
 show.clust.cent = FALSE, ggtheme = theme_minimal())
#ANÁLISE DOS CLUSTERS
# calculando o K-means com K = 4
km.res <- eclust(df, "kmeans", k = 4, nstart = 25, graph = FALSE)
# print(km.res)
# Visualização k-means Clusters
fviz_Cluster(km.res, geom = "point", ellipse.type = "norm", palette = "jco", ggtheme = theme_mini
mal())
# Hierárquico Clustering
hc.res <-
eclust(df, "hclust", k = 4, hc_metric = "euclidean", hc_method = "ward.D2", graph = FALSE)
# Visualização do Dendograma
fviz_dend(hc.res, show_labels = FALSE, palette = "jco", as.ggplot = TRUE)
#VALIDAÇÃO DOS CLUSTERS
# fviz_silhouette() [factoextra package]
fviz_silhouette(km.res, palette = "jco", ggtheme = theme_classic())






# Dados aleatórios gerados a partir do conjunto de dados base_grao_estudante
random_df <- apply(df, 2, function(x){runif(length(x), min(x), (max(x)))})
random_df <- as.data.frame(random_df)
# Padronizar os conjuntos de dados
df <- medio.scaled <- scale(df)
random_df <- scale(random_df)
#Inspeção visual dos dados
# Traçar conjunto de dados fiéis por id_turma da base_grao_estudante
fviz_pca_ind(prcomp(df), title = "PCA - Base Estudantes Ensino Médio",
habillage = medio$id_turma, palette = "jco",
geom = "point", ggtheme = theme_classic(),
legend = "bottom")
# Trace o df aleatório
fviz_pca_ind(prcomp(random_df), title = "PCA - Random dados",
geom = "point", ggtheme = theme_classic())
set.seed(123)



# K-means no conjunto de dados da base ensino médio (para os 4 Clusters)
km.res1 <- kmeans(df, 4)
fviz_Cluster(list(data = df, Cluster = km.res1$Cluster),
ellipse.type = "norm", geom = "point", stand = FALSE,
palette = "jco", ggtheme = theme_classic())
# K-means no conjunto de dados aleatório (para os 4 Clusters)
km.res2 <- kmeans(random_df, 4)
fviz_Cluster(list(data = random_df, Cluster = km.res2$Cluster),
ellipse.type = "norm", geom = "point", stand = FALSE,
palette = "jco", ggtheme = theme_classic())
# Clustering hierárquico no conjunto de dados aleatório (para os 4 Clusters)
fviz_dend(hclust(dist(random_df)), k = 4, k_colors = "jco",
as.ggplot = TRUE, show_labels = FALSE)
### MÉTODO ESTATÍSTICO
library(Clustertend)
# Estatística de Hopkins para cálculo do conjunto de dados Ensino Médio
set.seed(123)
hopkins(df, n = nrow(df)-1)
# Estatística de Hopkins para um conjunto de dados aleatório
set.seed(123)
hopkins(random_df, n = nrow(random_df)-1)
fviz_dist(dist(df), show_labels = FALSE)+
labs(title = "Base - Ensino Médio")
fviz_dist(dist(random_df), show_labels = FALSE)+
labs(title = "Base - Aleatória")



#Estimando o K para o algoritmo
# 1 - método Elbow
fviz_nbclust(dfmed, kmeans, method = "wss") + geom_vline(xintercept = 4, linetype = 2)+ labs(su
btitle = "Elbow method")
# 2 - método Silhouette
fviz_nbclust(dfmed, kmeans, method = "silhouette")+labs(subtitle = "Silhouette method")
# 3 - Método Gap statistic (Estatística de lacunas)
# nboot = 50 para manter a função rápida.
# valor sugerido: nboot= 500 (interações) para a análise.
# Use verbose = FALSE para ocultar a progressão do processo de cálculo.
set.seed(123)
fviz_nbclust(dfmed, kmeans, nstart = 25, method = "gap_stat", nboot = 500)+
labs(subtitle = "Gap statistic method")
set.seed(123)
km.res.med <- kmeans(dfmed, 4, nstart = 25)
print(km.res.med)
# Calculando a média de cada variável por Clusters usando os dados originais:
aggregate(medio[8:17], by=list(Cluster=km.res.med$Cluster), mean)
ddmed <- cbind(medio, Cluster = km.res.med$Cluster)
head(ddmed)


# ANÁLISE DOS DADOS DO ENSINO MÉDIO - k-means
#K-MEANS E A CLUSTERING
# Atribui a base (normalizado2) a variável df ignorando as variáveis (id_estudante, id_turma, serie, en
sino e var11)
df <- scale(medio[8:17])
# calculando o K-means com K = 4
km.res <- kmeans(df, 4, nstart = 25)
# Imprimindo o resultado
print(km.res)
# Calculando a média de cada variável por Clusters usando os dados originais:
aggregate(medio[8:17], by=list(Cluster=km.res$Cluster), mean)
# Calculando os Clusters em R
dd <- cbind(medio, Cluster = km.res$Cluster)


# Calcule a matriz de dissimilaridade
res.dist.med <- dist(dfmed, method = "euclidean")
as.matrix(res.dist.med)[1:6, 1:6]
#Calculando a árvore hierárquica
# d: uma estrutura de dissimilaridade produzida pela dist () função.
# método: O método de aglomeração (ligação) a ser usado para calcular a distância entre Clusters. O
s valores permitidos são “ward.D”, “ward.D2”, “single”, “complete”, “average”, “mcquitty”, “median” ou “
centroid”.
res.hc.med <- hclust(d = res.dist.med, method = "ward.D2")
# Dendrogram
# cex: tamanho do rótulo
fviz_dend(res.hc.med, cex = 0.5)
### Verifique a árvore do Cluster
# Calculando as distâncias cofenéticas para agrupamento hierárquico
# Calcular distância copêntica
res.coph.med <- cophenetic(res.hc.med)
# Correlação entre distância cofenética e
# a distância original
cor(res.dist.med, res.coph.med)
# Executa novamente o método de ligação de médias
res.hc2.med <- hclust(res.dist.med, method = "average")
cor(res.dist.med, cophenetic(res.hc2.med))
# Com o método = “average” o resultado foi: 0.9626782.

# O coeficiente de correlação mostra que o uso de um método de ligação diferente cria uma árvore qu
e representa as distâncias originais um pouco melhor.
#cortando a árvore em 4 grupos
grp <- cutree(res.hc.med, k = 4)
head(grp, n = 4)
# Número de membros em cada Cluster
table(grp)
# Obtenha os nomes dos membros do Cluster 1
rownames (medio)[grp == 1]
# Cortar em 4 grupos e colorir por grupos
fviz_dend(res.hc.med, k = 4, # Corte em quatro grupos
 cex = 0.5, # tamanho da etiqueta
 k_colors = c("#2E9FDF", "#00AFBB", "#E7B800", "#FC4E07"),
 color_labels_by_k = TRUE, # rótulos de cores por grupos
 rect = TRUE # Adicionar retângulo ao redor dos grupos
)
#Gráfico de Clusters
fviz_Cluster(list(data = dfmed, Cluster = grp),
 palette = c("#2E9FDF", "#00AFBB", "#E7B800", "#FC4E07"),
 ellipse.type = "convex", # Elipse de concentração
 repel = TRUE, # Evite o overplotting de rótulo (lento)
 show.clust.cent = FALSE, ggtheme = theme_minimal())
#ANÁLISE DOS CLUSTERS
# calculando o K-means com K = 4
km.res <- eclust(df, "kmeans", k = 4, nstart = 25, graph = FALSE)
# print(km.res)
# Visualização k-means Clusters
fviz_Cluster(km.res, geom = "point", ellipse.type = "norm", palette = "jco", ggtheme = theme_minimal()
)
# Hierárquico Clustering
hc.res <- eclust(df, "hclust", k = 4, hc_metric = "euclidean", hc_method = "ward.D2", graph = FALSE)
# Visualização do Dendograma
fviz_dend(hc.res, show_labels = FALSE, palette = "jco", as.ggplot = TRUE)
#VALIDAÇÃO DOS CLUSTERS
# fviz_silhouette() [factoextra package]
fviz_silhouette(km.res, palette = "jco", ggtheme = theme_classic())
