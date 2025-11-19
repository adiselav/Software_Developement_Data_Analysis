import numpy as np
import pandas as pd

from functii import nan_replace_t, calcul_partitie
from scipy.cluster.hierarchy import linkage
from grafice import plot_ierarhie, show, plot_partitie, plot_silhouette, plot_histograms
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA

np.set_printoptions(suppress=True)

# Incarcarea si procesarea setului de date
set_date = pd.read_csv("data/wine.csv")
nan_replace_t(set_date)

#Pregatim variabilele pentru gruparea pe clusteri
variabile_observate = list(set_date.columns)

x = set_date[variabile_observate].values
metoda = "ward"

h = linkage(x,metoda)
print(h)

# Graficul dendrograma
plot_ierarhie(h, None, "Grafic ierarhie - Metoda " + metoda)
show()

# Determinarea partitiei optimale folosind metoda Elbow
p_opt,threshold_opt = calcul_partitie(h)
plot_ierarhie(h, None, "Partitia optimala - Metoda " + metoda, threshold=threshold_opt)
show()

# Calcularea scorului silhouette
silhouette_avg = silhouette_score(x, p_opt)
print(f"Silhouette Score (Optimal Partition): {silhouette_avg}")

# Trasformam datele folosind PCA
pca = PCA(n_components=2)
z = pca.fit_transform(x)

# Impartirea lotului in componente principale
plot_partitie(z, p_opt, "Plot partitie optimala. Metoda " + metoda, np.unique(p_opt))
show()

# Grafic cu scorurile silhouette
plot_silhouette(x, p_opt)
show()

# Deseneaza histograme pentru fiecare cluster
plot_histograms(x, p_opt, variabile_observate)

# Afisarea componentei partitiei
partitie_df = pd.DataFrame({"Instance": np.arange(len(p_opt)), "Cluster": p_opt})
for cluster in np.unique(p_opt):
    cluster_instances = partitie_df[partitie_df["Cluster"] == cluster]["Instance"].values
    print(f"Cluster {cluster}: {list(cluster_instances)}")
