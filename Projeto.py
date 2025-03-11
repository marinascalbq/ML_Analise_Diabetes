import pandas as pd  # Manipulação de dados
import numpy as np  # Cálculos numéricos eficientes
import time  # Medir tempo de execução
import matplotlib.pyplot as plt  # Criar gráficos
import matplotlib.cm as cm  # Definir cores dos gráficos
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN  # Algoritmos de agrupamento
from sklearn.preprocessing import MinMaxScaler  # Normalização dos dados
from sklearn.decomposition import PCA  # Redução de dimensionalidade
from sklearn.metrics import silhouette_score, davies_bouldin_score, silhouette_samples  # Avaliação dos clusters

# KMeans, AgglomerativeClustering e DBSCAN são métodos de clustering 
# serão usados para segmentação dos pacientes

# Justificativa dos algoritmos escolhidos
# K-Means: Escolhido por ser um dos algoritmos mais populares de agrupamento baseado em centroides, eficiente para grandes volumes de dados.
# Agglomerative Clustering: Escolhido para permitir análise hierárquica dos grupos formados.
# DBSCAN: Algoritmo baseado em densidade, útil para identificar outliers e clusters de formato irregular.


# Carregar os dados
data = pd.read_csv("diabetes2.csv")

# Seleção dos atributos relevantes para a análise: BMI (Índice de Massa Corporal) e Age (Idade)
# Avaliar o impacto do IMC e da idade no agrupamento dos pacientes
features = data[["BMI", "Age"]]

# Remover outliers usando o método IQR
# Eliminar outliers melhora a qualidade da segmentação, evitando clusters distorcidos
Q1 = features.quantile(0.25)
Q3 = features.quantile(0.75)
IQR = Q3 - Q1
filtered_data = features[~((features < (Q1 - 1.5 * IQR)) | (features > (Q3 + 1.5 * IQR))).any(axis=1)]
print(f"Shape do dataset após remoção de outliers: {filtered_data.shape}")  
if filtered_data.empty:
    print("Erro: Nenhum dado restante após remoção de outliers!")

# Normalizar os dados para um intervalo entre 0 e 1 usando MinMaxScaler(). 
# Isso garante que todas as variáveis tenham a mesma escala, evitando que uma domine os cálculos.
scaler = MinMaxScaler()
normalized_data = scaler.fit_transform(filtered_data)

# Para otimizar o agrupamento, aplicamos PCA (Análise de Componentes Principais) 
# reduzindo a dimensionalidade dos dados enquanto preservamos 95% da variância.
pca = PCA(n_components=0.95)
pca_data = pca.fit_transform(normalized_data)

# Definir o número de execuções conforme solicitado
num_experiments = 50
random_state = 42

# Inicializar estrutura para armazenar resultados
results = {
    "algorithm": [],
    "silhouette_score": [],
    "davies_bouldin_score": [],
    "execution_time": []
}

# Executar os experimentos para cada algoritmo
# Cada algoritmo é executado 50 vezes, garantindo que os resultados não sejam influenciados pelo acaso. 
# Calculamos as métricas para cada execução.

k = 3
for _ in range(num_experiments):
        for algorithm_name, model in zip([
            f"KMeans_k={k}", "AgglomerativeClustering", "DBSCAN"],
            [KMeans(n_clusters=k, random_state=random_state),
            AgglomerativeClustering(n_clusters=k),
            DBSCAN(eps=0.5, min_samples=10)]):

            start_time = time.time()
            labels = model.fit_predict(pca_data)
            end_time = time.time()
            
            # Calcular métricas
            # Índice de Silhueta: Mede a qualidade dos clusters.
            silhouette = silhouette_score(pca_data, labels) if len(set(labels)) > 1 else -1
            # Índice de Davies-Bouldin: Avalia a separação dos clusters.
            db_score = davies_bouldin_score(pca_data, labels) if len(set(labels)) > 1 else -1
            # Tempo de Execução: Mede a eficiência computacional.
            execution_time = end_time - start_time

            # Armazenar resultados
            results["algorithm"].append(algorithm_name)
            results["silhouette_score"].append(silhouette)
            results["davies_bouldin_score"].append(db_score)
            results["execution_time"].append(execution_time)
            
            # Os resultados são armazenados e processados para calcular a média das métricas por algoritmo.

# Converter resultados em DataFrame
results_df = pd.DataFrame(results)

# Calcular médias por algoritmo
summary = results_df.groupby("algorithm").mean()
print(summary)

# Salvar resultados para análise posterior
results_df.to_csv("clustering_results.csv", index=False)
summary.to_csv("clustering_summary.csv")

# Visualização dos clusters
kmeans = KMeans(n_clusters=3, random_state=random_state)
cluster_labels = kmeans.fit_predict(pca_data)

# Calcular coeficientes de silhouette para geração do grafico da Análise de Silhueta
silhouette_vals = silhouette_samples(pca_data, cluster_labels)

# Gráfico 1: Silhouette plot
# O gráfico de silhueta nos ajuda a visualizar o quão bem os pontos estão agrupados.

fig, ax1 = plt.subplots(figsize=(8, 6))
y_lower, y_upper = 0, 0
for i in range(3):
    cluster_silhouette_vals = silhouette_vals[cluster_labels == i]
    cluster_silhouette_vals.sort()
    y_upper += len(cluster_silhouette_vals)
    ax1.fill_betweenx(np.arange(y_lower, y_upper), 0, cluster_silhouette_vals)
    y_lower = y_upper
ax1.set_title("Silhouette plot for various clusters")
ax1.set_xlabel("Silhouette coefficient values")
ax1.set_ylabel("Cluster label")
ax1.axvline(x=silhouette_score(pca_data, cluster_labels), color="red", linestyle="--")
plt.show()

# Gráfico 2: Visualização dos clusters
fig, ax2 = plt.subplots(figsize=(8, 6))
colors = cm.nipy_spectral(cluster_labels.astype(float) / 3)
ax2.scatter(pca_data[:, 0], pca_data[:, 1], marker="o", s=50, c=colors, edgecolor='k')
ax2.set_title("Cluster Visualization for K=3")
ax2.set_xlabel("PCA Component 1")
ax2.set_ylabel("PCA Component 2")
plt.show()

# Esses gráficos permitem avaliar se os clusters estão bem definidos ou sobrepostos.
