import matplotlib.pyplot as plt
import numpy as np
from scipy.cluster.hierarchy import dendrogram
from seaborn import scatterplot, histplot

def plot_ierarhie(h:np.ndarray,etichete,titlu,threshold=0):
    fig = plt.figure(figsize=(10,6))
    ax = fig.add_subplot(1,1,1)
    ax.set_title(titlu,fontdict={"fontsize":16,"color":"b"})
    dendrogram(h,ax=ax,labels=etichete,color_threshold=threshold)

def show():
    plt.show()

def plot_partitie(z:np.ndarray,p,titlu,clase,etichete=None):
    fig = plt.figure(figsize=(8,7))
    ax = fig.add_subplot(1,1,1)
    ax.set_title(titlu,fontdict={"fontsize":16,"color":"b"})
    ax.set_xlabel("Z1")
    ax.set_ylabel("Z2")
    scatterplot(x=z[:,0],y=z[:,1],hue=p,hue_order=clase,ax=ax,legend=True)
    if etichete is not None:
        for i in range(len(etichete)):
            ax.text(x=z[i,0],y=z[i,1],s=etichete[i])

def plot_silhouette(x, labels):
    from sklearn.metrics import silhouette_samples
    silhouette_vals = silhouette_samples(x, labels)
    y_ticks = []
    y_lower = 0
    for i, cluster in enumerate(np.unique(labels)):
        cluster_silhouette_vals = silhouette_vals[labels == cluster]
        cluster_silhouette_vals.sort()
        y_upper = y_lower + len(cluster_silhouette_vals)
        plt.barh(range(y_lower, y_upper), cluster_silhouette_vals, edgecolor="none")
        y_ticks.append((y_lower + y_upper) / 2)
        y_lower = y_upper
    plt.axvline(np.mean(silhouette_vals), color="red", linestyle="--")
    plt.yticks(y_ticks, np.unique(labels))
    plt.ylabel("Cluster")
    plt.xlabel("Silhouette Coefficient")

def plot_histograms(x, labels, columns):
    unique_clusters = np.unique(labels)
    for col_idx, col_name in enumerate(columns):
        plt.figure(figsize=(8, 6))
        for cluster in unique_clusters:
            # Extract numeric part of cluster labels
            cluster_num = int(cluster[1:])
            cluster_data = x[labels == cluster, col_idx]
            histplot(cluster_data, label=f"Cluster {cluster_num}", kde=True)
        plt.title(f"Histogram for {col_name}")
        plt.legend()
        plt.show()
