import pandas as pd
import numpy as np
import time
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import silhouette_score, davies_bouldin_score, silhouette_samples

# Carregar os dados
data = pd.read_csv("diabetes2.csv")

# Selecionar apenas as colunas de interesse
features = data[["BMI", "Age"]]

# Remover outliers usando o método IQR
Q1 = features.quantile(0.25)
Q3 = features.quantile(0.75)
IQR = Q3 - Q1
filtered_data = features[~((features < (Q1 - 1.5 * IQR)) | (features > (Q3 + 1.5 * IQR))).any(axis=1)]
print(f"Shape do dataset após remoção de outliers: {filtered_data.shape}")  
if filtered_data.empty:
    print("Erro: Nenhum dado restante após remoção de outliers!")

# Normalizar os dados
scaler = MinMaxScaler()
normalized_data = scaler.fit_transform(filtered_data)

# Definir o número de execuções
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
for _ in range(num_experiments):
    for algorithm_name, model in zip([
        "KMeans", "AgglomerativeClustering", "DBSCAN"],
        [KMeans(n_clusters=3, random_state=random_state),
         AgglomerativeClustering(n_clusters=3),
         DBSCAN(eps=0.3, min_samples=5)]):

        start_time = time.time()
        labels = model.fit_predict(normalized_data)
        end_time = time.time()
        
        # Calcular métricas
        silhouette = silhouette_score(normalized_data, labels) if len(set(labels)) > 1 else -1
        db_score = davies_bouldin_score(normalized_data, labels) if len(set(labels)) > 1 else -1
        execution_time = end_time - start_time

        # Armazenar resultados
        results["algorithm"].append(algorithm_name)
        results["silhouette_score"].append(silhouette)
        results["davies_bouldin_score"].append(db_score)
        results["execution_time"].append(execution_time)

# Converter resultados em DataFrame
results_df = pd.DataFrame(results)

# Calcular médias por algoritmo
summary = results_df.groupby("algorithm").mean()
print(summary)

# Salvar resultados para análise posterior
results_df.to_csv("clustering_results.csv", index=False)
summary.to_csv("clustering_summary.csv")

# Realizar clustering com KMeans para n_clusters=2
kmeans = KMeans(n_clusters=2, random_state=random_state)
cluster_labels = kmeans.fit_predict(normalized_data)

# Calcular coeficientes de silhouette
silhouette_vals = silhouette_samples(normalized_data, cluster_labels)

# Gráfico 1: Silhouette plot
fig, ax1 = plt.subplots(figsize=(8, 6))
y_lower, y_upper = 0, 0
for i in range(2):
    cluster_silhouette_vals = silhouette_vals[cluster_labels == i]
    cluster_silhouette_vals.sort()
    y_upper += len(cluster_silhouette_vals)
    ax1.fill_betweenx(np.arange(y_lower, y_upper), 0, cluster_silhouette_vals)
    y_lower = y_upper
ax1.set_title("Silhouette plot for various clusters")
ax1.set_xlabel("Silhouette coefficient values")
ax1.set_ylabel("Cluster label")
ax1.axvline(x=silhouette_score(normalized_data, cluster_labels), color="red", linestyle="--")
plt.show()

# Gráfico 2: Visualização dos clusters
fig, ax2 = plt.subplots(figsize=(8, 6))
colors = cm.nipy_spectral(cluster_labels.astype(float) / 2)
ax2.scatter(filtered_data.iloc[:, 0], filtered_data.iloc[:, 1], marker="o", s=50, c=colors, edgecolor='k')
ax2.set_title("Feature space for the 1st feature x Feature space for the 2nd feature")
ax2.set_xlabel("Feature 1: BMI")
ax2.set_ylabel("Feature 2: Age")
plt.show()
