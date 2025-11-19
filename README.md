# Software_Developement_for_Data_Analysis
Hierarchical Agglomerative Clustering (HAC) Analysis on the UCI Wine Dataset

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.9+-informational)](https://www.python.org/downloads/)

This repository contains an academic, reproducible project completed for the **Software Developement for Data Analysis** course.  
It demonstrates an end-to-end **unsupervised learning** workflow-data cleaning, dimensionality reduction (PCA), **hierarchical agglomerative clustering (HAC)**, cluster quality evaluation (silhouette score), and visualization (dendrograms, scatter plots, histograms)-using the classic **UCI Wine** dataset.

## Repository Structure

```text
sdad-clustering-wine/
|-- src/
|   |-- main.py            # Pipeline entrypoint: load -> preprocess -> PCA -> cluster -> evaluate -> plot
|   |-- functii.py         # Data cleaning and helper functions
|   |-- grafice.py         # Visualization utilities (dendrogram, silhouette, histograms)
|-- data/
|   |-- wine.csv           # UCI Wine dataset (tabular)
|-- docs/
|   |-- Course-Requirements.pdf
|   |-- Project-Report.docx
|-- CITATION.cff
|-- LICENSE
|-- requirements.txt
|-- .gitignore
```

## Methodology

### Clustering Approach

**Category:** Hierarchical Clustering (Unsupervised Learning)

**Algorithm:** Agglomerative Hierarchical Clustering

**Linkage Method:** Ward's method - minimizes the variance within clusters, providing compact and well-separated groups.

### Technologies Used

- **Python** (>= 3.9.13) - Core programming language
- **NumPy** (>= 2.0.2) - Numerical computing and array operations
- **Pandas** (>= 2.3.3) - Data manipulation and analysis
- **SciPy** (>= 1.13.1) - Hierarchical clustering implementation
- **Scikit-learn** (>= 1.6.1) - PCA transformation and silhouette scoring
- **Matplotlib** (>= 3.9.4) - Data visualization
- **Seaborn** (>= 0.13.2) - Statistical data visualization

### Analysis Pipeline

1. **Data Loading and Preprocessing**
   - Load the UCI Wine dataset from CSV
   - Handle missing values using mean imputation for numeric features
   - Standardize numeric features for distance-based clustering

2. **Hierarchical Clustering**
   - Apply agglomerative hierarchical clustering using Ward's linkage method
   - Generate linkage matrix for dendrogram visualization

3. **Optimal Partition Detection**
   - Determine the optimal number of clusters using the Elbow method
   - Calculate the optimal cutting threshold for the dendrogram

4. **Quality Evaluation**
   - Compute Silhouette Score to assess cluster quality
   - Analyze silhouette coefficients for individual samples

5. **Dimensionality Reduction**
   - Apply Principal Component Analysis (PCA) with 2 components
   - Transform high-dimensional data for 2D visualization

6. **Visualization and Interpretation**
   - Generate dendrograms showing hierarchical structure
   - Create scatter plots of clusters in PCA space
   - Plot silhouette analysis for each cluster
   - Generate histograms for feature distributions per cluster

## Results & Discussion

The report in `docs/Project-Report.docx` includes interpretation of the clusters, silhouette analysis, and how PCA components separate wine classes. For a hiring manager, focus on:

- **Code quality & structure** (clear modules, docstrings, typing where applicable)
- **Reproducibility** (locked deps, one-command run)
- **Communication** (README, report, and diagrams)

## Dataset

This project uses the **Wine** dataset (donated 1991-06-30), available from:

- [Kaggle: Wine Dataset for Clustering](https://www.kaggle.com/datasets/harrywang/wine-dataset-for-clustering)
- [UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/109/wine)

**Creators:** Stefan Aeberhard, M. Forina

**Citation (IEEE):**  
S. Aeberhard and M. Forina. "Wine," UCI Machine Learning Repository, 1992. [Online]. Available: <https://doi.org/10.24432/C5PC7J>

**DOI:** [10.24432/C5PC7J](https://doi.org/10.24432/C5PC7J)

## Citing

If this work helps your research or teaching, please cite via the metadata in `CITATION.cff` (GitHub's "Cite this repository" button will render a BibTeX/APA reference).

## License

MIT - see [LICENSE](LICENSE).
