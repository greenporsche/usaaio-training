# Problem 13B Variations: End-to-End ML - Unsupervised Learning (COMPREHENSIVE)

> Original Problem 13 focuses on supervised classification. This companion file covers unsupervised learning pipelines.
> Core Skills: Clustering, dimensionality reduction, anomaly detection, evaluation without labels, scikit-learn

---

## CATEGORY A: Clustering with Different Datasets

### Variation A1: Iris Clustering (Ground Truth Available)

You are given the **Iris dataset** for clustering. While labels exist, you must cluster **without using them** during training.

```python
import numpy as np
import pandas as pd
from sklearn.datasets import load_iris

# Load dataset
data = load_iris()
X = pd.DataFrame(data.data, columns=data.feature_names)
y_true = data.target  # Only for evaluation, NOT for training
```

This dataset has 3 natural clusters (species). Your task is to discover them.

### Submission Requirements

Submit a single Jupyter notebook containing:
1. Data preprocessing
2. Clustering model construction
3. Model fitting
4. Cluster assignment logic

### Inference Requirements

```python
def my_clustering(X_test):
    ###INSERT YOUR CODE HERE###
    return cluster_labels  # Integer labels 0, 1, 2, ...
```

Your clustering will be evaluated using **Adjusted Rand Index (ARI)** against true labels.

### Model Constraints

You must use **K-Means** as your clustering approach.

<details>
<summary>Solution A1</summary>

```python
import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, silhouette_score
from sklearn.pipeline import Pipeline

# Load data
data = load_iris()
X = pd.DataFrame(data.data, columns=data.feature_names)
y_true = data.target

# Create pipeline with preprocessing and K-Means
pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('kmeans', KMeans(n_clusters=3, random_state=2026, n_init=10))
])

# Fit the pipeline
pipeline.fit(X)

# Get cluster labels
cluster_labels = pipeline.predict(X)

# Evaluate (using ground truth for comparison)
ari = adjusted_rand_score(y_true, cluster_labels)
silhouette = silhouette_score(X, cluster_labels)

print(f"Adjusted Rand Index: {ari:.4f}")
print(f"Silhouette Score: {silhouette:.4f}")

# Analyze cluster centers
scaler = pipeline.named_steps['scaler']
kmeans = pipeline.named_steps['kmeans']
centers_original = scaler.inverse_transform(kmeans.cluster_centers_)
print("\nCluster Centers (original scale):")
print(pd.DataFrame(centers_original, columns=X.columns))

def my_clustering(X_test):
    return pipeline.predict(X_test)
```

**Key Insights**:
- K-Means requires scaling (distance-based algorithm)
- n_init=10 runs K-Means 10 times with different initializations
- ARI measures agreement between clusterings, adjusted for chance
- Silhouette score doesn't need ground truth (internal metric)

</details>

### Variation A2: Blob Dataset with Varying Density

Synthetic blobs with **different densities**:

```python
import numpy as np
from sklearn.datasets import make_blobs

np.random.seed(2026)
# Create blobs with different standard deviations
X, y_true = make_blobs(
    n_samples=[100, 300, 500],  # Different sizes
    centers=[[0, 0], [5, 5], [10, 0]],
    cluster_std=[0.5, 1.5, 0.8],  # Different densities
    random_state=2026
)
```

### Model Constraints

Use **DBSCAN** which can handle varying densities better than K-Means.

<details>
<summary>Solution A2</summary>

```python
import numpy as np
import pandas as pd
from sklearn.datasets import make_blobs
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN
from sklearn.metrics import adjusted_rand_score, silhouette_score
from sklearn.neighbors import NearestNeighbors
import matplotlib.pyplot as plt

# Create data
np.random.seed(2026)
X, y_true = make_blobs(
    n_samples=[100, 300, 500],
    centers=[[0, 0], [5, 5], [10, 0]],
    cluster_std=[0.5, 1.5, 0.8],
    random_state=2026
)

# Scale data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Find optimal eps using k-distance graph
k = 5  # min_samples - 1
nbrs = NearestNeighbors(n_neighbors=k+1).fit(X_scaled)
distances, _ = nbrs.kneighbors(X_scaled)
k_distances = np.sort(distances[:, k])

plt.figure(figsize=(10, 4))
plt.plot(k_distances)
plt.xlabel('Points (sorted by distance)')
plt.ylabel(f'{k}-th Nearest Neighbor Distance')
plt.title('K-Distance Graph for eps Selection')
plt.axhline(y=0.5, color='r', linestyle='--', label='Candidate eps=0.5')
plt.legend()
plt.show()

# Apply DBSCAN with chosen parameters
dbscan = DBSCAN(eps=0.5, min_samples=5)
cluster_labels = dbscan.fit_predict(X_scaled)

# Handle noise points (labeled as -1)
n_clusters = len(set(cluster_labels)) - (1 if -1 in cluster_labels else 0)
n_noise = list(cluster_labels).count(-1)

print(f"Number of clusters: {n_clusters}")
print(f"Number of noise points: {n_noise}")

# Evaluate (excluding noise for silhouette)
mask = cluster_labels != -1
if mask.sum() > 0 and len(set(cluster_labels[mask])) > 1:
    ari = adjusted_rand_score(y_true[mask], cluster_labels[mask])
    silhouette = silhouette_score(X_scaled[mask], cluster_labels[mask])
    print(f"ARI (excluding noise): {ari:.4f}")
    print(f"Silhouette (excluding noise): {silhouette:.4f}")

# Visualization
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.scatter(X[:, 0], X[:, 1], c=y_true, cmap='viridis', alpha=0.6)
plt.title('True Labels')

plt.subplot(1, 2, 2)
plt.scatter(X[:, 0], X[:, 1], c=cluster_labels, cmap='viridis', alpha=0.6)
plt.title('DBSCAN Clusters')
plt.show()

# For inference, we need to assign new points
# DBSCAN doesn't have predict(), so we use nearest cluster assignment
from sklearn.neighbors import KNeighborsClassifier

# Train a classifier on core points
core_mask = cluster_labels != -1
if core_mask.sum() > 0:
    knn = KNeighborsClassifier(n_neighbors=5)
    knn.fit(X_scaled[core_mask], cluster_labels[core_mask])

def my_clustering(X_test):
    X_test_scaled = scaler.transform(X_test)
    # Use KNN to assign to nearest cluster
    return knn.predict(X_test_scaled)
```

**Key Insights**:
- DBSCAN doesn't require specifying k (number of clusters)
- eps: maximum distance between neighbors
- min_samples: minimum points to form dense region
- Points not in any cluster are labeled -1 (noise)
- K-distance graph helps choose eps (look for "elbow")
- DBSCAN has no predict() - need workaround for new data

</details>

### Variation A3: High-Dimensional Text Clustering

Cluster **text documents** represented as TF-IDF vectors:

```python
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer

# Load a subset of newsgroups
categories = ['sci.med', 'sci.space', 'rec.sport.baseball']
newsgroups = fetch_20newsgroups(
    subset='train',
    categories=categories,
    remove=('headers', 'footers', 'quotes'),
    random_state=2026
)

X_text = newsgroups.data
y_true = newsgroups.target

# Convert to TF-IDF
vectorizer = TfidfVectorizer(max_features=1000, stop_words='english')
X = vectorizer.fit_transform(X_text)
```

This is high-dimensional sparse data (1000 features).

<details>
<summary>Solution A3</summary>

```python
import numpy as np
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans, MiniBatchKMeans
from sklearn.decomposition import TruncatedSVD
from sklearn.pipeline import Pipeline
from sklearn.metrics import adjusted_rand_score, silhouette_score
from sklearn.preprocessing import Normalizer

# Load data
categories = ['sci.med', 'sci.space', 'rec.sport.baseball']
newsgroups = fetch_20newsgroups(
    subset='train',
    categories=categories,
    remove=('headers', 'footers', 'quotes'),
    random_state=2026
)

X_text = newsgroups.data
y_true = newsgroups.target

# TF-IDF vectorization
vectorizer = TfidfVectorizer(max_features=1000, stop_words='english')
X_tfidf = vectorizer.fit_transform(X_text)

print(f"TF-IDF shape: {X_tfidf.shape}")

# Option 1: K-Means directly on TF-IDF (sparse)
kmeans_direct = KMeans(n_clusters=3, random_state=2026, n_init=10)
labels_direct = kmeans_direct.fit_predict(X_tfidf)
ari_direct = adjusted_rand_score(y_true, labels_direct)
print(f"K-Means on TF-IDF - ARI: {ari_direct:.4f}")

# Option 2: Dimensionality reduction + K-Means
# TruncatedSVD works with sparse matrices (unlike PCA)
pipeline = Pipeline([
    ('svd', TruncatedSVD(n_components=100, random_state=2026)),
    ('normalizer', Normalizer()),  # L2 normalize for cosine similarity
    ('kmeans', KMeans(n_clusters=3, random_state=2026, n_init=10))
])

labels_svd = pipeline.fit_predict(X_tfidf)
ari_svd = adjusted_rand_score(y_true, labels_svd)
print(f"SVD + K-Means - ARI: {ari_svd:.4f}")

# Analyze top terms per cluster
def get_top_terms(cluster_centers, feature_names, n_terms=10):
    for i, center in enumerate(cluster_centers):
        top_indices = center.argsort()[-n_terms:][::-1]
        top_terms = [feature_names[idx] for idx in top_indices]
        print(f"Cluster {i}: {', '.join(top_terms)}")

# For direct K-Means
print("\nTop terms per cluster (direct K-Means):")
feature_names = vectorizer.get_feature_names_out()
get_top_terms(kmeans_direct.cluster_centers_, feature_names)

# Final model
final_model = pipeline

def my_clustering(X_test_text):
    X_test_tfidf = vectorizer.transform(X_test_text)
    return final_model.predict(X_test_tfidf)
```

**Key Insights**:
- TF-IDF creates sparse high-dimensional vectors
- TruncatedSVD (LSA) works with sparse matrices; PCA doesn't
- Normalizer after SVD → cosine similarity behavior in K-Means
- MiniBatchKMeans faster for large datasets
- Analyzing cluster centers reveals topic themes

</details>

---

## CATEGORY B: Different Evaluation Metrics

### Variation B1: Silhouette Score Optimization

Optimize clustering using **Silhouette Score** (no ground truth needed):

```python
# Your predictions will be evaluated using Silhouette Score
# Higher is better (range: -1 to 1)
```

<details>
<summary>Solution B1</summary>

```python
import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, silhouette_samples
import matplotlib.pyplot as plt

# Load data
data = load_iris()
X = pd.DataFrame(data.data, columns=data.feature_names)

# Scale
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Find optimal k using silhouette score
k_range = range(2, 11)
silhouette_scores = []

for k in k_range:
    kmeans = KMeans(n_clusters=k, random_state=2026, n_init=10)
    labels = kmeans.fit_predict(X_scaled)
    score = silhouette_score(X_scaled, labels)
    silhouette_scores.append(score)
    print(f"k={k}: Silhouette = {score:.4f}")

# Plot
plt.figure(figsize=(10, 4))
plt.plot(k_range, silhouette_scores, 'bo-')
plt.xlabel('Number of Clusters (k)')
plt.ylabel('Silhouette Score')
plt.title('Silhouette Score vs k')
plt.grid(True, alpha=0.3)
plt.show()

# Best k
best_k = k_range[np.argmax(silhouette_scores)]
print(f"\nBest k = {best_k} with Silhouette = {max(silhouette_scores):.4f}")

# Fit final model
best_kmeans = KMeans(n_clusters=best_k, random_state=2026, n_init=10)
best_kmeans.fit(X_scaled)

# Silhouette analysis per sample
labels = best_kmeans.labels_
sample_silhouettes = silhouette_samples(X_scaled, labels)

# Plot silhouette diagram
fig, ax = plt.subplots(figsize=(8, 6))
y_lower = 10

for i in range(best_k):
    cluster_silhouettes = sample_silhouettes[labels == i]
    cluster_silhouettes.sort()

    cluster_size = len(cluster_silhouettes)
    y_upper = y_lower + cluster_size

    ax.fill_betweenx(np.arange(y_lower, y_upper),
                      0, cluster_silhouettes,
                      alpha=0.7, label=f'Cluster {i}')
    y_lower = y_upper + 10

ax.axvline(x=silhouette_score(X_scaled, labels), color='red',
           linestyle='--', label='Average')
ax.set_xlabel('Silhouette Coefficient')
ax.set_ylabel('Cluster')
ax.legend()
plt.title('Silhouette Diagram')
plt.show()

def my_clustering(X_test):
    X_test_scaled = scaler.transform(X_test)
    return best_kmeans.predict(X_test_scaled)
```

**Key Insights**:
- Silhouette Score: (b - a) / max(a, b) where a=intra-cluster, b=inter-cluster
- Range: -1 (wrong cluster) to +1 (well-clustered)
- No ground truth needed → useful for real clustering
- Silhouette diagram shows per-sample quality
- Negative silhouettes indicate misassigned points

</details>

### Variation B2: Davies-Bouldin Index

Optimize using **Davies-Bouldin Index** (lower is better):

<details>
<summary>Solution B2</summary>

```python
import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import davies_bouldin_score
import matplotlib.pyplot as plt

# Load and scale data
data = load_iris()
X = StandardScaler().fit_transform(data.data)

# Find optimal k using Davies-Bouldin
k_range = range(2, 11)
db_scores = []

for k in k_range:
    kmeans = KMeans(n_clusters=k, random_state=2026, n_init=10)
    labels = kmeans.fit_predict(X)
    score = davies_bouldin_score(X, labels)
    db_scores.append(score)
    print(f"k={k}: Davies-Bouldin = {score:.4f}")

# Best k (LOWER is better for DB)
best_k = k_range[np.argmin(db_scores)]
print(f"\nBest k = {best_k} with Davies-Bouldin = {min(db_scores):.4f}")

# Plot
plt.figure(figsize=(10, 4))
plt.plot(k_range, db_scores, 'ro-')
plt.xlabel('Number of Clusters (k)')
plt.ylabel('Davies-Bouldin Index (lower is better)')
plt.title('Davies-Bouldin Index vs k')
plt.grid(True, alpha=0.3)
plt.show()
```

**Key Insights**:
- Davies-Bouldin: Average similarity between clusters
- LOWER is better (0 = perfect separation)
- Measures ratio of within-cluster to between-cluster distances
- Tends to favor convex clusters (like Silhouette)

</details>

### Variation B3: Calinski-Harabasz Index

Optimize using **Calinski-Harabasz Index** (higher is better):

<details>
<summary>Solution B3</summary>

```python
import numpy as np
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import calinski_harabasz_score
import matplotlib.pyplot as plt

# Load and scale
X = StandardScaler().fit_transform(load_iris().data)

# Evaluate different k
k_range = range(2, 11)
ch_scores = []

for k in k_range:
    kmeans = KMeans(n_clusters=k, random_state=2026, n_init=10)
    labels = kmeans.fit_predict(X)
    score = calinski_harabasz_score(X, labels)
    ch_scores.append(score)
    print(f"k={k}: Calinski-Harabasz = {score:.2f}")

# Best k (HIGHER is better)
best_k = k_range[np.argmax(ch_scores)]
print(f"\nBest k = {best_k}")

# Plot
plt.figure(figsize=(10, 4))
plt.plot(k_range, ch_scores, 'go-')
plt.xlabel('Number of Clusters (k)')
plt.ylabel('Calinski-Harabasz Index (higher is better)')
plt.title('Variance Ratio Criterion')
plt.grid(True, alpha=0.3)
plt.show()
```

**Key Insights**:
- Calinski-Harabasz = Variance Ratio Criterion
- Ratio of between-cluster to within-cluster variance
- HIGHER is better
- Computationally faster than Silhouette
- Also known as the Variance Ratio Criterion (VRC)

</details>

### Variation B4: Multiple Metrics Comparison

Compare **all internal metrics** and make a decision:

<details>
<summary>Solution B4</summary>

```python
import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import (
    silhouette_score,
    davies_bouldin_score,
    calinski_harabasz_score
)
import matplotlib.pyplot as plt

# Load and scale
X = StandardScaler().fit_transform(load_iris().data)

# Evaluate multiple metrics
k_range = range(2, 11)
results = []

for k in k_range:
    kmeans = KMeans(n_clusters=k, random_state=2026, n_init=10)
    labels = kmeans.fit_predict(X)

    results.append({
        'k': k,
        'silhouette': silhouette_score(X, labels),
        'davies_bouldin': davies_bouldin_score(X, labels),
        'calinski_harabasz': calinski_harabasz_score(X, labels),
        'inertia': kmeans.inertia_
    })

df = pd.DataFrame(results)
print(df.to_string(index=False))

# Normalize and combine (for visualization)
df_norm = df.copy()
df_norm['silhouette_norm'] = df['silhouette'] / df['silhouette'].max()
df_norm['db_norm'] = 1 - df['davies_bouldin'] / df['davies_bouldin'].max()  # Invert
df_norm['ch_norm'] = df['calinski_harabasz'] / df['calinski_harabasz'].max()

# Plot all metrics
fig, axes = plt.subplots(2, 2, figsize=(12, 10))

axes[0, 0].plot(df['k'], df['silhouette'], 'bo-')
axes[0, 0].set_title('Silhouette Score (↑ better)')
axes[0, 0].set_xlabel('k')

axes[0, 1].plot(df['k'], df['davies_bouldin'], 'ro-')
axes[0, 1].set_title('Davies-Bouldin Index (↓ better)')
axes[0, 1].set_xlabel('k')

axes[1, 0].plot(df['k'], df['calinski_harabasz'], 'go-')
axes[1, 0].set_title('Calinski-Harabasz Index (↑ better)')
axes[1, 0].set_xlabel('k')

axes[1, 1].plot(df['k'], df['inertia'], 'mo-')
axes[1, 1].set_title('Inertia / Elbow Method (↓ better)')
axes[1, 1].set_xlabel('k')

plt.tight_layout()
plt.show()

# Decision logic
print("\n=== Metric Recommendations ===")
print(f"Silhouette suggests k = {df.loc[df['silhouette'].idxmax(), 'k']}")
print(f"Davies-Bouldin suggests k = {df.loc[df['davies_bouldin'].idxmin(), 'k']}")
print(f"Calinski-Harabasz suggests k = {df.loc[df['calinski_harabasz'].idxmax(), 'k']}")

# Voting
from collections import Counter
votes = [
    df.loc[df['silhouette'].idxmax(), 'k'],
    df.loc[df['davies_bouldin'].idxmin(), 'k'],
    df.loc[df['calinski_harabasz'].idxmax(), 'k']
]
consensus_k = Counter(votes).most_common(1)[0][0]
print(f"\nConsensus k = {consensus_k}")
```

**Key Insights**:
- Different metrics may suggest different k
- Silhouette: Most interpretable, but O(n²)
- Davies-Bouldin: Fast, but sensitive to outliers
- Calinski-Harabasz: Fast, favors dense spherical clusters
- Elbow method (inertia): Look for "elbow" point
- Use voting or domain knowledge to decide

</details>

---

## CATEGORY C: Different Clustering Algorithms

### Variation C1: Hierarchical Clustering

You must use **Agglomerative Hierarchical Clustering**:

<details>
<summary>Solution C1</summary>

```python
import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import adjusted_rand_score, silhouette_score
from scipy.cluster.hierarchy import dendrogram, linkage
import matplotlib.pyplot as plt

# Load and scale
data = load_iris()
X = StandardScaler().fit_transform(data.data)
y_true = data.target

# Create dendrogram to visualize hierarchy
plt.figure(figsize=(12, 5))
Z = linkage(X, method='ward')
dendrogram(Z, truncate_mode='lastp', p=30)
plt.title('Hierarchical Clustering Dendrogram')
plt.xlabel('Sample Index')
plt.ylabel('Distance')
plt.show()

# Compare different linkage methods
linkage_methods = ['ward', 'complete', 'average', 'single']
results = []

for method in linkage_methods:
    # Ward only works with euclidean
    agg = AgglomerativeClustering(
        n_clusters=3,
        linkage=method
    )
    labels = agg.fit_predict(X)

    ari = adjusted_rand_score(y_true, labels)
    sil = silhouette_score(X, labels)
    results.append({
        'linkage': method,
        'ARI': ari,
        'Silhouette': sil
    })

print(pd.DataFrame(results).to_string(index=False))

# Best method
best_method = max(results, key=lambda x: x['ARI'])['linkage']
print(f"\nBest linkage: {best_method}")

# Final model
final_agg = AgglomerativeClustering(n_clusters=3, linkage=best_method)
final_agg.fit(X)

# Note: AgglomerativeClustering has no predict()
# For new data, use nearest centroid or retrain
from sklearn.neighbors import NearestCentroid
nc = NearestCentroid()
nc.fit(X, final_agg.labels_)

scaler = StandardScaler()
scaler.fit(data.data)

def my_clustering(X_test):
    X_test_scaled = scaler.transform(X_test)
    return nc.predict(X_test_scaled)
```

**Key Insights**:
- Hierarchical clustering builds a tree (dendrogram)
- Linkage methods:
  - Ward: Minimizes variance (usually best)
  - Complete: Maximum distance between clusters
  - Average: Mean distance (UPGMA)
  - Single: Minimum distance (prone to chaining)
- No predict() method → need workaround for new data
- Dendrogram helps choose number of clusters visually

</details>

### Variation C2: Gaussian Mixture Models (GMM)

You must use **Gaussian Mixture Models** for soft clustering:

<details>
<summary>Solution C2</summary>

```python
import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
from sklearn.mixture import GaussianMixture
from sklearn.metrics import adjusted_rand_score, silhouette_score
import matplotlib.pyplot as plt

# Load and scale
data = load_iris()
X = StandardScaler().fit_transform(data.data)
y_true = data.target

# Find optimal number of components using BIC/AIC
n_components_range = range(1, 10)
bic_scores = []
aic_scores = []

for n in n_components_range:
    gmm = GaussianMixture(n_components=n, random_state=2026, n_init=5)
    gmm.fit(X)
    bic_scores.append(gmm.bic(X))
    aic_scores.append(gmm.aic(X))

# Plot
plt.figure(figsize=(10, 4))
plt.plot(n_components_range, bic_scores, 'bo-', label='BIC')
plt.plot(n_components_range, aic_scores, 'rs-', label='AIC')
plt.xlabel('Number of Components')
plt.ylabel('Information Criterion (lower is better)')
plt.title('Model Selection for GMM')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()

# Best n_components
best_n = n_components_range[np.argmin(bic_scores)]
print(f"Best n_components (BIC): {best_n}")

# Fit final model
gmm = GaussianMixture(n_components=3, random_state=2026, n_init=5)
gmm.fit(X)

# Hard clustering
hard_labels = gmm.predict(X)
print(f"ARI: {adjusted_rand_score(y_true, hard_labels):.4f}")

# Soft clustering (probabilities)
soft_labels = gmm.predict_proba(X)
print(f"\nSoft clustering example (first 5 samples):")
print(pd.DataFrame(soft_labels[:5], columns=[f'Cluster {i}' for i in range(3)]))

# Analyze learned parameters
print(f"\nCluster weights: {gmm.weights_}")
print(f"Converged: {gmm.converged_}")
print(f"Iterations: {gmm.n_iter_}")

# GMM has predict() built-in
scaler = StandardScaler()
scaler.fit(data.data)
gmm_final = GaussianMixture(n_components=3, random_state=2026, n_init=5)
gmm_final.fit(scaler.transform(data.data))

def my_clustering(X_test):
    X_test_scaled = scaler.transform(X_test)
    return gmm_final.predict(X_test_scaled)

def my_soft_clustering(X_test):
    X_test_scaled = scaler.transform(X_test)
    return gmm_final.predict_proba(X_test_scaled)
```

**Key Insights**:
- GMM is probabilistic: provides soft assignments
- Models clusters as Gaussian distributions
- BIC/AIC for model selection (lower is better)
- BIC penalizes complexity more than AIC
- Has predict() and predict_proba() built-in
- Can capture elliptical clusters (not just spherical)

</details>

### Variation C3: Spectral Clustering

You must use **Spectral Clustering** for non-convex clusters:

<details>
<summary>Solution C3</summary>

```python
import numpy as np
import pandas as pd
from sklearn.datasets import make_moons, make_circles
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import SpectralClustering, KMeans
from sklearn.metrics import adjusted_rand_score
import matplotlib.pyplot as plt

# Create non-convex data (moons)
X_moons, y_moons = make_moons(n_samples=300, noise=0.1, random_state=2026)

# Create concentric circles
X_circles, y_circles = make_circles(n_samples=300, noise=0.05,
                                     factor=0.5, random_state=2026)

# Compare K-Means vs Spectral on moons
fig, axes = plt.subplots(2, 3, figsize=(15, 10))

for idx, (X, y_true, name) in enumerate([(X_moons, y_moons, 'Moons'),
                                          (X_circles, y_circles, 'Circles')]):

    # True labels
    axes[idx, 0].scatter(X[:, 0], X[:, 1], c=y_true, cmap='viridis')
    axes[idx, 0].set_title(f'{name}: True Labels')

    # K-Means (fails on non-convex)
    kmeans = KMeans(n_clusters=2, random_state=2026)
    km_labels = kmeans.fit_predict(X)
    km_ari = adjusted_rand_score(y_true, km_labels)
    axes[idx, 1].scatter(X[:, 0], X[:, 1], c=km_labels, cmap='viridis')
    axes[idx, 1].set_title(f'K-Means (ARI={km_ari:.3f})')

    # Spectral Clustering (handles non-convex)
    spectral = SpectralClustering(
        n_clusters=2,
        affinity='nearest_neighbors',
        n_neighbors=10,
        random_state=2026
    )
    spec_labels = spectral.fit_predict(X)
    spec_ari = adjusted_rand_score(y_true, spec_labels)
    axes[idx, 2].scatter(X[:, 0], X[:, 1], c=spec_labels, cmap='viridis')
    axes[idx, 2].set_title(f'Spectral (ARI={spec_ari:.3f})')

plt.tight_layout()
plt.show()

# Spectral clustering with different affinity
affinities = ['nearest_neighbors', 'rbf']
for aff in affinities:
    if aff == 'rbf':
        spectral = SpectralClustering(n_clusters=2, affinity='rbf',
                                       gamma=1.0, random_state=2026)
    else:
        spectral = SpectralClustering(n_clusters=2, affinity='nearest_neighbors',
                                       n_neighbors=10, random_state=2026)

    labels = spectral.fit_predict(X_moons)
    ari = adjusted_rand_score(y_moons, labels)
    print(f"Affinity={aff}: ARI={ari:.4f}")

# Note: SpectralClustering has no predict() for new data
# Need to embed and use nearest neighbor
from sklearn.manifold import SpectralEmbedding

embedding = SpectralEmbedding(n_components=2, affinity='nearest_neighbors',
                               n_neighbors=10, random_state=2026)
X_embedded = embedding.fit_transform(X_moons)

# Cluster in embedded space
kmeans_embed = KMeans(n_clusters=2, random_state=2026)
kmeans_embed.fit(X_embedded)

# For new points, would need to embed first (complex)
# Usually: retrain on X_train + X_test
```

**Key Insights**:
- Spectral clustering: Graph-based, handles non-convex shapes
- Uses eigenvalues of similarity matrix (spectral decomposition)
- K-Means fails on non-convex clusters (moons, circles)
- Affinity options: rbf (Gaussian), nearest_neighbors
- No native predict() - requires workarounds for new data
- Computational complexity: O(n³) for eigendecomposition

</details>

### Variation C4: Mean Shift Clustering

You must use **Mean Shift** which automatically determines k:

<details>
<summary>Solution C4</summary>

```python
import numpy as np
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import MeanShift, estimate_bandwidth
from sklearn.metrics import adjusted_rand_score, silhouette_score
import matplotlib.pyplot as plt

# Load and scale
data = load_iris()
X = StandardScaler().fit_transform(data.data)
y_true = data.target

# Estimate bandwidth (kernel size)
bandwidth = estimate_bandwidth(X, quantile=0.2, n_samples=100)
print(f"Estimated bandwidth: {bandwidth:.4f}")

# Fit Mean Shift
meanshift = MeanShift(bandwidth=bandwidth, bin_seeding=True)
labels = meanshift.fit_predict(X)

n_clusters = len(np.unique(labels))
print(f"Number of clusters found: {n_clusters}")
print(f"ARI: {adjusted_rand_score(y_true, labels):.4f}")

# Cluster centers
print(f"\nCluster centers shape: {meanshift.cluster_centers_.shape}")

# Test different bandwidths
bandwidths = [0.5, 1.0, 1.5, 2.0, 2.5]
results = []

for bw in bandwidths:
    ms = MeanShift(bandwidth=bw, bin_seeding=True)
    labels = ms.fit_predict(X)
    n_clusters = len(np.unique(labels))
    ari = adjusted_rand_score(y_true, labels) if n_clusters > 1 else 0
    results.append({
        'bandwidth': bw,
        'n_clusters': n_clusters,
        'ARI': ari
    })

print("\nBandwidth sensitivity:")
print(pd.DataFrame(results).to_string(index=False))

# Mean Shift has predict() for new data
scaler = StandardScaler()
scaler.fit(data.data)
ms_final = MeanShift(bandwidth=bandwidth, bin_seeding=True)
ms_final.fit(scaler.transform(data.data))

def my_clustering(X_test):
    X_test_scaled = scaler.transform(X_test)
    return ms_final.predict(X_test_scaled)
```

**Key Insights**:
- Mean Shift: Mode-finding algorithm (seeks density peaks)
- Automatically determines number of clusters
- Bandwidth controls smoothness (like eps in DBSCAN)
- estimate_bandwidth() helps choose bandwidth
- bin_seeding=True speeds up by discretizing
- Has predict() for new data (assigns to nearest mode)
- Sensitive to bandwidth choice

</details>

---

## CATEGORY D: Dimensionality Reduction

### Variation D1: PCA as Primary Task

Perform **PCA for visualization and analysis** (not just preprocessing):

<details>
<summary>Solution D1</summary>

```python
import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

# Load data
data = load_iris()
X = pd.DataFrame(data.data, columns=data.feature_names)
y = data.target

# Scale (essential before PCA)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Full PCA
pca_full = PCA()
X_pca_full = pca_full.fit_transform(X_scaled)

# Analyze explained variance
explained_var = pca_full.explained_variance_ratio_
cumulative_var = np.cumsum(explained_var)

print("Explained Variance Ratio:")
for i, (var, cum) in enumerate(zip(explained_var, cumulative_var)):
    print(f"  PC{i+1}: {var:.4f} (cumulative: {cum:.4f})")

# Plot explained variance
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

axes[0].bar(range(1, len(explained_var)+1), explained_var, alpha=0.7, label='Individual')
axes[0].plot(range(1, len(cumulative_var)+1), cumulative_var, 'ro-', label='Cumulative')
axes[0].axhline(y=0.95, color='g', linestyle='--', label='95% threshold')
axes[0].set_xlabel('Principal Component')
axes[0].set_ylabel('Explained Variance Ratio')
axes[0].set_title('PCA Explained Variance')
axes[0].legend()

# 2D projection
axes[1].scatter(X_pca_full[:, 0], X_pca_full[:, 1], c=y, cmap='viridis', alpha=0.7)
axes[1].set_xlabel(f'PC1 ({explained_var[0]:.1%})')
axes[1].set_ylabel(f'PC2 ({explained_var[1]:.1%})')
axes[1].set_title('PCA 2D Projection')
plt.tight_layout()
plt.show()

# Loadings analysis (what features contribute to each PC)
loadings = pd.DataFrame(
    pca_full.components_.T,
    columns=[f'PC{i+1}' for i in range(len(X.columns))],
    index=X.columns
)
print("\nPCA Loadings (feature contributions):")
print(loadings.round(3))

# Find components for 95% variance
n_components_95 = np.argmax(cumulative_var >= 0.95) + 1
print(f"\nComponents for 95% variance: {n_components_95}")

# Biplot
def biplot(scores, loadings, labels=None, feature_names=None):
    fig, ax = plt.subplots(figsize=(10, 8))

    # Plot scores
    scatter = ax.scatter(scores[:, 0], scores[:, 1], c=labels, cmap='viridis', alpha=0.6)

    # Plot loadings as arrows
    scale = 3  # Arrow scaling factor
    for i, fname in enumerate(feature_names):
        ax.arrow(0, 0, loadings[i, 0]*scale, loadings[i, 1]*scale,
                 head_width=0.1, head_length=0.05, fc='red', ec='red')
        ax.text(loadings[i, 0]*scale*1.1, loadings[i, 1]*scale*1.1, fname,
                fontsize=9, ha='center')

    ax.set_xlabel('PC1')
    ax.set_ylabel('PC2')
    ax.set_title('PCA Biplot')
    ax.axhline(y=0, color='k', linestyle='-', linewidth=0.5)
    ax.axvline(x=0, color='k', linestyle='-', linewidth=0.5)
    plt.colorbar(scatter)
    return fig

biplot(X_pca_full, pca_full.components_.T, y, X.columns)
plt.show()

# Transform function
pca_final = PCA(n_components=2)
pca_final.fit(X_scaled)

def my_transform(X_test):
    X_test_scaled = scaler.transform(X_test)
    return pca_final.transform(X_test_scaled)
```

**Key Insights**:
- PCA finds orthogonal directions of maximum variance
- MUST scale before PCA (otherwise large-scale features dominate)
- Loadings show feature contributions to each PC
- Biplot: Combine scores (samples) and loadings (features)
- 95% explained variance is common threshold
- PCA is linear → can't capture non-linear structure

</details>

### Variation D2: t-SNE for Visualization

Use **t-SNE** for non-linear dimensionality reduction:

<details>
<summary>Solution D2</summary>

```python
import numpy as np
import pandas as pd
from sklearn.datasets import load_digits
from sklearn.preprocessing import StandardScaler
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

# Load digits (higher dimensional)
data = load_digits()
X = data.data
y = data.target

# Scale
X_scaled = StandardScaler().fit_transform(X)

# t-SNE with different perplexities
perplexities = [5, 30, 50, 100]

fig, axes = plt.subplots(1, len(perplexities), figsize=(20, 5))

for ax, perp in zip(axes, perplexities):
    tsne = TSNE(
        n_components=2,
        perplexity=perp,
        random_state=2026,
        n_iter=1000,
        learning_rate='auto',
        init='pca'  # More stable initialization
    )
    X_tsne = tsne.fit_transform(X_scaled)

    scatter = ax.scatter(X_tsne[:, 0], X_tsne[:, 1], c=y, cmap='tab10',
                         alpha=0.6, s=10)
    ax.set_title(f'Perplexity = {perp}')
    ax.set_xticks([])
    ax.set_yticks([])

plt.suptitle('t-SNE on Digits Dataset')
plt.tight_layout()
plt.show()

# Best perplexity (typically 5-50, depends on data size)
tsne_best = TSNE(n_components=2, perplexity=30, random_state=2026,
                  learning_rate='auto', init='pca')
X_tsne_best = tsne_best.fit_transform(X_scaled)

# Detailed visualization
plt.figure(figsize=(12, 10))
scatter = plt.scatter(X_tsne_best[:, 0], X_tsne_best[:, 1],
                      c=y, cmap='tab10', alpha=0.7, s=20)
plt.colorbar(scatter, label='Digit')
plt.title('t-SNE Visualization of Digits Dataset')
plt.xlabel('t-SNE 1')
plt.ylabel('t-SNE 2')

# Add legend
for digit in range(10):
    mask = y == digit
    centroid = X_tsne_best[mask].mean(axis=0)
    plt.annotate(str(digit), centroid, fontsize=14, fontweight='bold',
                 ha='center', va='center',
                 bbox=dict(boxstyle='circle', facecolor='white', alpha=0.8))
plt.show()

# IMPORTANT: t-SNE has NO transform for new data
# Options:
# 1. Retrain on X_train + X_test
# 2. Use parametric t-SNE (neural network)
# 3. Use UMAP instead (has transform)
print("\nNote: t-SNE cannot transform new data!")
print("For new data, consider UMAP or parametric methods.")
```

**Key Insights**:
- t-SNE: Non-linear, preserves local structure
- Perplexity ≈ "effective number of neighbors" (5-50 typical)
- Different perplexities → different visualizations
- NO transform() for new data (non-parametric)
- Computationally expensive: O(n²) or O(n log n) with approximations
- init='pca' gives more reproducible results
- Good for visualization, NOT for preprocessing

</details>

### Variation D3: UMAP for Dimensionality Reduction

Use **UMAP** as an alternative to t-SNE with transform capability:

<details>
<summary>Solution D3</summary>

```python
import numpy as np
import pandas as pd
from sklearn.datasets import load_digits
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# UMAP needs to be installed: pip install umap-learn
try:
    import umap
except ImportError:
    print("Install UMAP: pip install umap-learn")
    raise

# Load digits
data = load_digits()
X = data.data
y = data.target

# Split data (to demonstrate transform)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=2026, stratify=y
)

# Scale
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# UMAP with different n_neighbors
n_neighbors_list = [5, 15, 50, 100]

fig, axes = plt.subplots(1, len(n_neighbors_list), figsize=(20, 5))

for ax, n_neigh in zip(axes, n_neighbors_list):
    reducer = umap.UMAP(
        n_components=2,
        n_neighbors=n_neigh,
        min_dist=0.1,
        random_state=2026
    )
    X_umap = reducer.fit_transform(X_train_scaled)

    scatter = ax.scatter(X_umap[:, 0], X_umap[:, 1], c=y_train,
                         cmap='tab10', alpha=0.6, s=10)
    ax.set_title(f'n_neighbors = {n_neigh}')
    ax.set_xticks([])
    ax.set_yticks([])

plt.suptitle('UMAP on Digits Dataset')
plt.tight_layout()
plt.show()

# Best parameters
reducer = umap.UMAP(
    n_components=2,
    n_neighbors=15,
    min_dist=0.1,
    metric='euclidean',
    random_state=2026
)

# Fit on training data
X_train_umap = reducer.fit_transform(X_train_scaled)

# Transform test data (UMAP can do this!)
X_test_umap = reducer.transform(X_test_scaled)

# Visualize train and test
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

axes[0].scatter(X_train_umap[:, 0], X_train_umap[:, 1],
                c=y_train, cmap='tab10', alpha=0.6, s=20)
axes[0].set_title('Training Data (fit_transform)')

axes[1].scatter(X_train_umap[:, 0], X_train_umap[:, 1],
                c=y_train, cmap='tab10', alpha=0.3, s=10, label='Train')
axes[1].scatter(X_test_umap[:, 0], X_test_umap[:, 1],
                c=y_test, cmap='tab10', alpha=0.8, s=30, marker='x', label='Test')
axes[1].set_title('Test Data Transformed')
axes[1].legend()

plt.tight_layout()
plt.show()

# For supervised UMAP (uses labels)
reducer_supervised = umap.UMAP(
    n_components=2,
    n_neighbors=15,
    min_dist=0.1,
    random_state=2026
)
X_train_umap_sup = reducer_supervised.fit_transform(X_train_scaled, y_train)

def my_transform(X_new):
    X_new_scaled = scaler.transform(X_new)
    return reducer.transform(X_new_scaled)
```

**Key Insights**:
- UMAP: Like t-SNE but faster and has transform()
- n_neighbors: Local vs global structure (like perplexity)
- min_dist: How tightly points cluster (0.0-1.0)
- Can do supervised UMAP (uses labels to improve separation)
- HAS transform() for new data (unlike t-SNE)
- Good for both visualization AND preprocessing

</details>

---

## CATEGORY E: Anomaly Detection

### Variation E1: Isolation Forest

Detect anomalies using **Isolation Forest**:

<details>
<summary>Solution E1</summary>

```python
import numpy as np
import pandas as pd
from sklearn.datasets import make_blobs
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
from sklearn.metrics import precision_score, recall_score, f1_score
import matplotlib.pyplot as plt

# Create data with outliers
np.random.seed(2026)
X_normal, _ = make_blobs(n_samples=300, centers=1, cluster_std=1.0, random_state=2026)
X_outliers = np.random.uniform(low=-6, high=6, size=(30, 2))
X = np.vstack([X_normal, X_outliers])
y_true = np.array([1]*300 + [-1]*30)  # 1=inlier, -1=outlier

# Scale
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Isolation Forest
iso_forest = IsolationForest(
    n_estimators=100,
    contamination=0.1,  # Expected proportion of outliers
    random_state=2026
)

# Fit and predict
y_pred = iso_forest.fit_predict(X_scaled)
scores = iso_forest.decision_function(X_scaled)  # Anomaly scores

# Evaluate
print("Isolation Forest Results:")
print(f"Precision: {precision_score(y_true, y_pred, pos_label=-1):.4f}")
print(f"Recall: {recall_score(y_true, y_pred, pos_label=-1):.4f}")
print(f"F1 Score: {f1_score(y_true, y_pred, pos_label=-1):.4f}")

# Visualize
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# True labels
axes[0].scatter(X[y_true==1, 0], X[y_true==1, 1], c='blue', label='Inlier', alpha=0.6)
axes[0].scatter(X[y_true==-1, 0], X[y_true==-1, 1], c='red', label='Outlier', alpha=0.8)
axes[0].set_title('True Labels')
axes[0].legend()

# Predictions
axes[1].scatter(X[y_pred==1, 0], X[y_pred==1, 1], c='blue', label='Predicted Inlier', alpha=0.6)
axes[1].scatter(X[y_pred==-1, 0], X[y_pred==-1, 1], c='red', label='Predicted Outlier', alpha=0.8)
axes[1].set_title('Isolation Forest Predictions')
axes[1].legend()

# Anomaly scores
scatter = axes[2].scatter(X[:, 0], X[:, 1], c=scores, cmap='RdYlBu', alpha=0.7)
axes[2].set_title('Anomaly Scores (lower = more anomalous)')
plt.colorbar(scatter, ax=axes[2])

plt.tight_layout()
plt.show()

# Score distribution
plt.figure(figsize=(10, 4))
plt.hist(scores[y_true==1], bins=30, alpha=0.7, label='Inliers', density=True)
plt.hist(scores[y_true==-1], bins=30, alpha=0.7, label='Outliers', density=True)
plt.xlabel('Anomaly Score')
plt.ylabel('Density')
plt.title('Score Distribution')
plt.legend()
plt.show()

# Contamination sensitivity
contaminations = [0.05, 0.1, 0.15, 0.2]
for cont in contaminations:
    iso = IsolationForest(contamination=cont, random_state=2026)
    y_pred = iso.fit_predict(X_scaled)
    f1 = f1_score(y_true, y_pred, pos_label=-1)
    print(f"Contamination={cont}: F1={f1:.4f}")

def my_anomaly_detection(X_test):
    X_test_scaled = scaler.transform(X_test)
    return iso_forest.predict(X_test_scaled)  # 1=inlier, -1=outlier
```

**Key Insights**:
- Isolation Forest: Isolates anomalies via random partitioning
- Anomalies are "easy to isolate" → shorter path lengths
- contamination: Expected proportion of outliers (affects threshold)
- decision_function(): Lower scores = more anomalous
- Works well in high dimensions (unlike distance-based methods)
- No assumption about data distribution

</details>

### Variation E2: One-Class SVM

Use **One-Class SVM** for novelty detection:

<details>
<summary>Solution E2</summary>

```python
import numpy as np
import pandas as pd
from sklearn.datasets import make_blobs
from sklearn.preprocessing import StandardScaler
from sklearn.svm import OneClassSVM
from sklearn.metrics import precision_score, recall_score, f1_score
import matplotlib.pyplot as plt

# Create training data (inliers only)
np.random.seed(2026)
X_train, _ = make_blobs(n_samples=200, centers=1, cluster_std=1.0, random_state=2026)

# Create test data with outliers
X_test_inliers, _ = make_blobs(n_samples=50, centers=1, cluster_std=1.0, random_state=42)
X_test_outliers = np.random.uniform(low=-6, high=6, size=(20, 2))
X_test = np.vstack([X_test_inliers, X_test_outliers])
y_test = np.array([1]*50 + [-1]*20)

# Scale
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# One-Class SVM (novelty detection)
# Train on inliers ONLY
ocsvm = OneClassSVM(
    kernel='rbf',
    gamma='scale',
    nu=0.1  # Upper bound on outlier fraction
)
ocsvm.fit(X_train_scaled)

# Predict on test data
y_pred = ocsvm.predict(X_test_scaled)

# Evaluate
print("One-Class SVM Results:")
print(f"Precision: {precision_score(y_test, y_pred, pos_label=-1):.4f}")
print(f"Recall: {recall_score(y_test, y_pred, pos_label=-1):.4f}")
print(f"F1 Score: {f1_score(y_test, y_pred, pos_label=-1):.4f}")

# Visualize decision boundary
xx, yy = np.meshgrid(
    np.linspace(-8, 8, 100),
    np.linspace(-8, 8, 100)
)
Z = ocsvm.decision_function(scaler.transform(np.c_[xx.ravel(), yy.ravel()]))
Z = Z.reshape(xx.shape)

plt.figure(figsize=(10, 8))
plt.contourf(xx, yy, Z, levels=np.linspace(Z.min(), 0, 10), cmap='Blues_r', alpha=0.5)
plt.contour(xx, yy, Z, levels=[0], linewidths=2, colors='red')
plt.scatter(X_train[:, 0], X_train[:, 1], c='blue', s=30, label='Training (inliers)')
plt.scatter(X_test[y_test==1, 0], X_test[y_test==1, 1], c='green', s=50,
            marker='s', label='Test inliers')
plt.scatter(X_test[y_test==-1, 0], X_test[y_test==-1, 1], c='red', s=50,
            marker='x', label='Test outliers')
plt.legend()
plt.title('One-Class SVM Decision Boundary')
plt.show()

# Compare different nu values
nus = [0.01, 0.05, 0.1, 0.2]
for nu in nus:
    ocsvm = OneClassSVM(kernel='rbf', gamma='scale', nu=nu)
    ocsvm.fit(X_train_scaled)
    y_pred = ocsvm.predict(X_test_scaled)
    f1 = f1_score(y_test, y_pred, pos_label=-1)
    print(f"nu={nu}: F1={f1:.4f}")

def my_novelty_detection(X_new):
    X_new_scaled = scaler.transform(X_new)
    return ocsvm.predict(X_new_scaled)
```

**Key Insights**:
- One-Class SVM: Learns boundary around "normal" data
- Novelty detection: Train on CLEAN data (no outliers)
- Outlier detection: Train on contaminated data
- nu parameter: Upper bound on outlier fraction
- RBF kernel creates flexible boundaries
- gamma controls kernel width (like in regular SVM)

</details>

### Variation E3: Local Outlier Factor (LOF)

Use **Local Outlier Factor** for density-based anomaly detection:

<details>
<summary>Solution E3</summary>

```python
import numpy as np
from sklearn.datasets import make_blobs
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import LocalOutlierFactor
from sklearn.metrics import f1_score
import matplotlib.pyplot as plt

# Create data with outliers
np.random.seed(2026)
X_normal, _ = make_blobs(n_samples=300, centers=2, cluster_std=1.0, random_state=2026)
X_outliers = np.random.uniform(low=-8, high=8, size=(30, 2))
X = np.vstack([X_normal, X_outliers])
y_true = np.array([1]*300 + [-1]*30)

# Scale
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# LOF for outlier detection (novelty=False)
lof = LocalOutlierFactor(
    n_neighbors=20,
    contamination=0.1,
    novelty=False  # For outlier detection (fit_predict)
)

y_pred = lof.fit_predict(X_scaled)
scores = -lof.negative_outlier_factor_  # Higher = more anomalous

print("LOF Results:")
print(f"F1 Score: {f1_score(y_true, y_pred, pos_label=-1):.4f}")

# Visualize
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Predictions
colors = ['blue' if y == 1 else 'red' for y in y_pred]
axes[0].scatter(X[:, 0], X[:, 1], c=colors, alpha=0.6)
axes[0].set_title('LOF Predictions (red=outlier)')

# LOF scores
scatter = axes[1].scatter(X[:, 0], X[:, 1], c=scores, cmap='RdYlBu_r', alpha=0.7)
axes[1].set_title('LOF Scores (higher = more anomalous)')
plt.colorbar(scatter, ax=axes[1])

plt.tight_layout()
plt.show()

# n_neighbors sensitivity
n_neighbors_list = [5, 10, 20, 50]
for n in n_neighbors_list:
    lof = LocalOutlierFactor(n_neighbors=n, contamination=0.1, novelty=False)
    y_pred = lof.fit_predict(X_scaled)
    f1 = f1_score(y_true, y_pred, pos_label=-1)
    print(f"n_neighbors={n}: F1={f1:.4f}")

# For novelty detection (can predict on new data)
lof_novelty = LocalOutlierFactor(n_neighbors=20, contamination=0.1, novelty=True)
lof_novelty.fit(X_scaled[y_true==1])  # Train on inliers only

def my_novelty_lof(X_test):
    X_test_scaled = scaler.transform(X_test)
    return lof_novelty.predict(X_test_scaled)
```

**Key Insights**:
- LOF: Compares local density to neighbors' densities
- novelty=False: Outlier detection (fit_predict, no predict)
- novelty=True: Novelty detection (can predict on new data)
- LOF > 1: Less dense than neighbors (potential outlier)
- n_neighbors affects locality (too small = noisy, too large = misses local outliers)
- Good for data with varying density clusters

</details>

---

## CATEGORY F: From-Scratch Implementations

### Variation F1: K-Means from Scratch

Implement **K-Means from scratch** using NumPy only:

<details>
<summary>Solution F1</summary>

```python
import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import adjusted_rand_score
import matplotlib.pyplot as plt

class KMeansFromScratch:
    """K-Means clustering implemented from scratch."""

    def __init__(self, n_clusters=3, max_iter=300, tol=1e-4, random_state=None):
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.tol = tol
        self.random_state = random_state

    def fit(self, X):
        """Fit K-Means to data."""
        np.random.seed(self.random_state)
        X = np.array(X)
        n_samples, n_features = X.shape

        # Initialize centroids randomly (K-Means++)
        self.centroids_ = self._init_centroids_plus_plus(X)

        for iteration in range(self.max_iter):
            # Assignment step: Assign points to nearest centroid
            labels = self._assign_clusters(X)

            # Update step: Recompute centroids
            new_centroids = self._compute_centroids(X, labels)

            # Check convergence
            centroid_shift = np.sum(np.sqrt(np.sum((new_centroids - self.centroids_)**2, axis=1)))

            if centroid_shift < self.tol:
                print(f"Converged at iteration {iteration}")
                break

            self.centroids_ = new_centroids

        self.labels_ = self._assign_clusters(X)
        self.inertia_ = self._compute_inertia(X, self.labels_)
        self.n_iter_ = iteration + 1

        return self

    def _init_centroids_plus_plus(self, X):
        """K-Means++ initialization for better starting centroids."""
        n_samples = X.shape[0]
        centroids = []

        # First centroid: random
        idx = np.random.randint(n_samples)
        centroids.append(X[idx])

        # Subsequent centroids: proportional to squared distance
        for _ in range(1, self.n_clusters):
            distances = np.array([
                min(np.sum((x - c)**2) for c in centroids)
                for x in X
            ])
            probabilities = distances / distances.sum()
            idx = np.random.choice(n_samples, p=probabilities)
            centroids.append(X[idx])

        return np.array(centroids)

    def _assign_clusters(self, X):
        """Assign each point to nearest centroid."""
        # Vectorized distance computation
        # ||x - c||^2 = ||x||^2 + ||c||^2 - 2*x.c
        X_sq = np.sum(X**2, axis=1).reshape(-1, 1)
        C_sq = np.sum(self.centroids_**2, axis=1).reshape(1, -1)
        cross = 2 * X @ self.centroids_.T
        distances = X_sq + C_sq - cross
        return np.argmin(distances, axis=1)

    def _compute_centroids(self, X, labels):
        """Compute new centroids as mean of assigned points."""
        centroids = np.zeros((self.n_clusters, X.shape[1]))
        for k in range(self.n_clusters):
            mask = labels == k
            if mask.sum() > 0:
                centroids[k] = X[mask].mean(axis=0)
            else:
                # Empty cluster: reinitialize randomly
                centroids[k] = X[np.random.randint(len(X))]
        return centroids

    def _compute_inertia(self, X, labels):
        """Compute sum of squared distances to centroids."""
        inertia = 0
        for k in range(self.n_clusters):
            mask = labels == k
            if mask.sum() > 0:
                inertia += np.sum((X[mask] - self.centroids_[k])**2)
        return inertia

    def predict(self, X):
        """Assign new points to clusters."""
        return self._assign_clusters(np.array(X))

    def fit_predict(self, X):
        """Fit and return labels."""
        self.fit(X)
        return self.labels_

# Test implementation
data = load_iris()
X = StandardScaler().fit_transform(data.data)
y_true = data.target

# Our implementation
kmeans_scratch = KMeansFromScratch(n_clusters=3, random_state=2026)
labels_scratch = kmeans_scratch.fit_predict(X)
ari_scratch = adjusted_rand_score(y_true, labels_scratch)
print(f"From-scratch K-Means: ARI = {ari_scratch:.4f}")
print(f"Inertia: {kmeans_scratch.inertia_:.4f}")
print(f"Iterations: {kmeans_scratch.n_iter_}")

# Compare with sklearn
from sklearn.cluster import KMeans
kmeans_sklearn = KMeans(n_clusters=3, random_state=2026, n_init=1, init='k-means++')
labels_sklearn = kmeans_sklearn.fit_predict(X)
ari_sklearn = adjusted_rand_score(y_true, labels_sklearn)
print(f"\nSklearn K-Means: ARI = {ari_sklearn:.4f}")
print(f"Inertia: {kmeans_sklearn.inertia_:.4f}")

# Visualization (2D projection)
from sklearn.decomposition import PCA
X_2d = PCA(n_components=2).fit_transform(X)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

axes[0].scatter(X_2d[:, 0], X_2d[:, 1], c=labels_scratch, cmap='viridis', alpha=0.6)
centroids_2d = PCA(n_components=2).fit(X).transform(kmeans_scratch.centroids_)
axes[0].scatter(centroids_2d[:, 0], centroids_2d[:, 1], c='red', s=200, marker='X')
axes[0].set_title(f'From Scratch (ARI={ari_scratch:.3f})')

axes[1].scatter(X_2d[:, 0], X_2d[:, 1], c=labels_sklearn, cmap='viridis', alpha=0.6)
axes[1].set_title(f'Sklearn (ARI={ari_sklearn:.3f})')

plt.tight_layout()
plt.show()
```

**Key Insights**:
- K-Means alternates: assign clusters → update centroids
- K-Means++ initialization improves convergence
- Vectorized distance: ||x-c||² = ||x||² + ||c||² - 2x·c
- Handle empty clusters by reinitialization
- Inertia = sum of squared distances to centroids
- Convergence when centroids stop moving

</details>

### Variation F2: Hierarchical Clustering from Scratch

Implement **Agglomerative Hierarchical Clustering** from scratch:

<details>
<summary>Solution F2</summary>

```python
import numpy as np
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import adjusted_rand_score
import matplotlib.pyplot as plt

class AgglomerativeFromScratch:
    """Agglomerative Hierarchical Clustering from scratch."""

    def __init__(self, n_clusters=3, linkage='ward'):
        self.n_clusters = n_clusters
        self.linkage = linkage

    def fit(self, X):
        """Fit hierarchical clustering."""
        X = np.array(X)
        n_samples = X.shape[0]

        # Initialize: each point is its own cluster
        # clusters[i] = list of sample indices in cluster i
        clusters = {i: [i] for i in range(n_samples)}

        # Distance matrix
        distances = self._compute_distance_matrix(X)

        # Merge history for dendrogram
        self.merge_history_ = []

        # Merge until we have n_clusters
        while len(clusters) > self.n_clusters:
            # Find closest pair of clusters
            min_dist = np.inf
            merge_i, merge_j = None, None

            cluster_ids = list(clusters.keys())
            for i in range(len(cluster_ids)):
                for j in range(i+1, len(cluster_ids)):
                    ci, cj = cluster_ids[i], cluster_ids[j]
                    dist = self._cluster_distance(X, clusters[ci], clusters[cj], distances)
                    if dist < min_dist:
                        min_dist = dist
                        merge_i, merge_j = ci, cj

            # Merge clusters
            new_cluster_id = max(clusters.keys()) + 1
            clusters[new_cluster_id] = clusters[merge_i] + clusters[merge_j]
            del clusters[merge_i]
            del clusters[merge_j]

            self.merge_history_.append((merge_i, merge_j, min_dist, len(clusters[new_cluster_id])))

        # Create labels
        self.labels_ = np.zeros(n_samples, dtype=int)
        for label, (cluster_id, indices) in enumerate(clusters.items()):
            for idx in indices:
                self.labels_[idx] = label

        return self

    def _compute_distance_matrix(self, X):
        """Compute pairwise distance matrix."""
        n = X.shape[0]
        distances = np.zeros((n, n))
        for i in range(n):
            for j in range(i+1, n):
                distances[i, j] = np.sqrt(np.sum((X[i] - X[j])**2))
                distances[j, i] = distances[i, j]
        return distances

    def _cluster_distance(self, X, cluster1, cluster2, distances):
        """Compute distance between two clusters."""
        if self.linkage == 'single':
            # Minimum distance
            min_dist = np.inf
            for i in cluster1:
                for j in cluster2:
                    min_dist = min(min_dist, distances[i, j])
            return min_dist

        elif self.linkage == 'complete':
            # Maximum distance
            max_dist = 0
            for i in cluster1:
                for j in cluster2:
                    max_dist = max(max_dist, distances[i, j])
            return max_dist

        elif self.linkage == 'average':
            # Average distance
            total = 0
            for i in cluster1:
                for j in cluster2:
                    total += distances[i, j]
            return total / (len(cluster1) * len(cluster2))

        elif self.linkage == 'ward':
            # Ward's method: minimize variance increase
            c1_points = X[cluster1]
            c2_points = X[cluster2]
            merged_points = np.vstack([c1_points, c2_points])

            c1_var = np.sum((c1_points - c1_points.mean(axis=0))**2)
            c2_var = np.sum((c2_points - c2_points.mean(axis=0))**2)
            merged_var = np.sum((merged_points - merged_points.mean(axis=0))**2)

            return merged_var - c1_var - c2_var

    def fit_predict(self, X):
        self.fit(X)
        return self.labels_

# Test
data = load_iris()
X = StandardScaler().fit_transform(data.data)
y_true = data.target

# Compare linkage methods
linkages = ['single', 'complete', 'average', 'ward']
results = []

for link in linkages:
    agg = AgglomerativeFromScratch(n_clusters=3, linkage=link)
    labels = agg.fit_predict(X)
    ari = adjusted_rand_score(y_true, labels)
    results.append({'linkage': link, 'ARI': ari})
    print(f"{link}: ARI = {ari:.4f}")

# Compare with sklearn
from sklearn.cluster import AgglomerativeClustering
agg_sklearn = AgglomerativeClustering(n_clusters=3, linkage='ward')
labels_sklearn = agg_sklearn.fit_predict(X)
print(f"\nSklearn ward: ARI = {adjusted_rand_score(y_true, labels_sklearn):.4f}")
```

**Key Insights**:
- Agglomerative: Start with n clusters, merge until k remain
- Linkage methods:
  - Single: min distance (prone to chaining)
  - Complete: max distance (compact clusters)
  - Average: mean distance (balanced)
  - Ward: minimize variance increase (usually best)
- O(n³) naive implementation → O(n² log n) with efficient data structures
- No predict() for new data

</details>

### Variation F3: DBSCAN from Scratch

Implement **DBSCAN** from scratch:

<details>
<summary>Solution F3</summary>

```python
import numpy as np
from sklearn.datasets import make_moons
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import adjusted_rand_score
import matplotlib.pyplot as plt

class DBSCANFromScratch:
    """DBSCAN clustering implemented from scratch."""

    def __init__(self, eps=0.5, min_samples=5):
        self.eps = eps
        self.min_samples = min_samples

    def fit(self, X):
        """Fit DBSCAN."""
        X = np.array(X)
        n_samples = X.shape[0]

        # -1 = unvisited, -2 = noise, >= 0 = cluster label
        self.labels_ = np.full(n_samples, -1)

        # Precompute neighbors for efficiency
        neighbors = self._find_all_neighbors(X)

        cluster_id = 0

        for i in range(n_samples):
            if self.labels_[i] != -1:  # Already processed
                continue

            # Get neighbors
            neighbor_indices = neighbors[i]

            if len(neighbor_indices) < self.min_samples:
                # Mark as noise (may be changed later)
                self.labels_[i] = -2
            else:
                # Start a new cluster
                self._expand_cluster(X, i, neighbor_indices, cluster_id, neighbors)
                cluster_id += 1

        # Convert -2 (noise) to -1 (sklearn convention)
        self.labels_[self.labels_ == -2] = -1

        return self

    def _find_all_neighbors(self, X):
        """Find neighbors within eps for all points."""
        n_samples = X.shape[0]
        neighbors = {}

        for i in range(n_samples):
            distances = np.sqrt(np.sum((X - X[i])**2, axis=1))
            neighbors[i] = np.where(distances <= self.eps)[0].tolist()

        return neighbors

    def _expand_cluster(self, X, point_idx, neighbor_indices, cluster_id, neighbors):
        """Expand cluster from a core point."""
        self.labels_[point_idx] = cluster_id

        # Use a list as a queue (will grow as we add neighbors)
        seeds = list(neighbor_indices)
        i = 0

        while i < len(seeds):
            current_point = seeds[i]

            if self.labels_[current_point] == -2:
                # Was noise, now border point
                self.labels_[current_point] = cluster_id

            elif self.labels_[current_point] == -1:
                # Unvisited
                self.labels_[current_point] = cluster_id

                current_neighbors = neighbors[current_point]

                if len(current_neighbors) >= self.min_samples:
                    # Core point: add its neighbors to seeds
                    seeds.extend(current_neighbors)

            i += 1

    def fit_predict(self, X):
        self.fit(X)
        return self.labels_

# Test on moons dataset
X, y_true = make_moons(n_samples=300, noise=0.1, random_state=2026)

# Our implementation
dbscan_scratch = DBSCANFromScratch(eps=0.2, min_samples=5)
labels_scratch = dbscan_scratch.fit_predict(X)
ari_scratch = adjusted_rand_score(y_true, labels_scratch)

# Sklearn
from sklearn.cluster import DBSCAN
dbscan_sklearn = DBSCAN(eps=0.2, min_samples=5)
labels_sklearn = dbscan_sklearn.fit_predict(X)
ari_sklearn = adjusted_rand_score(y_true, labels_sklearn)

print(f"From-scratch DBSCAN: ARI = {ari_scratch:.4f}")
print(f"Sklearn DBSCAN: ARI = {ari_sklearn:.4f}")
print(f"Clusters found: {len(set(labels_scratch)) - (1 if -1 in labels_scratch else 0)}")
print(f"Noise points: {(labels_scratch == -1).sum()}")

# Visualize
fig, axes = plt.subplots(1, 3, figsize=(15, 4))

axes[0].scatter(X[:, 0], X[:, 1], c=y_true, cmap='viridis')
axes[0].set_title('True Labels')

axes[1].scatter(X[:, 0], X[:, 1], c=labels_scratch, cmap='viridis')
axes[1].set_title(f'From Scratch (ARI={ari_scratch:.3f})')

axes[2].scatter(X[:, 0], X[:, 1], c=labels_sklearn, cmap='viridis')
axes[2].set_title(f'Sklearn (ARI={ari_sklearn:.3f})')

plt.tight_layout()
plt.show()
```

**Key Insights**:
- DBSCAN: Density-based, finds arbitrary-shaped clusters
- Core point: has ≥ min_samples neighbors within eps
- Border point: not core but within eps of a core point
- Noise: neither core nor border
- Expands clusters by transitively adding neighbors
- O(n²) naive → O(n log n) with spatial indexing (KD-tree)

</details>

### Variation F4: PCA from Scratch

Implement **PCA** from scratch using eigendecomposition:

<details>
<summary>Solution F4</summary>

```python
import numpy as np
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

class PCAFromScratch:
    """PCA implemented from scratch using eigendecomposition."""

    def __init__(self, n_components=None):
        self.n_components = n_components

    def fit(self, X):
        """Fit PCA."""
        X = np.array(X)
        n_samples, n_features = X.shape

        # Center the data (assume already scaled)
        self.mean_ = X.mean(axis=0)
        X_centered = X - self.mean_

        # Compute covariance matrix
        # Cov = X^T X / (n - 1)
        cov_matrix = X_centered.T @ X_centered / (n_samples - 1)

        # Eigendecomposition
        eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)

        # Sort by eigenvalue (descending)
        sorted_idx = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[sorted_idx]
        eigenvectors = eigenvectors[:, sorted_idx]

        # Select top k components
        if self.n_components is None:
            self.n_components_ = n_features
        else:
            self.n_components_ = self.n_components

        self.components_ = eigenvectors[:, :self.n_components_].T  # Shape: (n_components, n_features)
        self.explained_variance_ = eigenvalues[:self.n_components_]
        self.explained_variance_ratio_ = eigenvalues[:self.n_components_] / eigenvalues.sum()

        return self

    def transform(self, X):
        """Project data onto principal components."""
        X = np.array(X)
        X_centered = X - self.mean_
        return X_centered @ self.components_.T

    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)

    def inverse_transform(self, X_transformed):
        """Reconstruct original data from transformed."""
        return X_transformed @ self.components_ + self.mean_

# Test
data = load_iris()
X = StandardScaler().fit_transform(data.data)
y = data.target

# Our implementation
pca_scratch = PCAFromScratch(n_components=2)
X_scratch = pca_scratch.fit_transform(X)

# Sklearn
from sklearn.decomposition import PCA
pca_sklearn = PCA(n_components=2)
X_sklearn = pca_sklearn.fit_transform(X)

print("Explained Variance Ratio:")
print(f"  From scratch: {pca_scratch.explained_variance_ratio_}")
print(f"  Sklearn:      {pca_sklearn.explained_variance_ratio_}")

# Visualize (note: signs may be flipped - that's OK)
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

axes[0].scatter(X_scratch[:, 0], X_scratch[:, 1], c=y, cmap='viridis', alpha=0.7)
axes[0].set_xlabel(f'PC1 ({pca_scratch.explained_variance_ratio_[0]:.1%})')
axes[0].set_ylabel(f'PC2 ({pca_scratch.explained_variance_ratio_[1]:.1%})')
axes[0].set_title('From Scratch PCA')

axes[1].scatter(X_sklearn[:, 0], X_sklearn[:, 1], c=y, cmap='viridis', alpha=0.7)
axes[1].set_xlabel(f'PC1 ({pca_sklearn.explained_variance_ratio_[0]:.1%})')
axes[1].set_ylabel(f'PC2 ({pca_sklearn.explained_variance_ratio_[1]:.1%})')
axes[1].set_title('Sklearn PCA')

plt.tight_layout()
plt.show()

# Test reconstruction
X_reconstructed = pca_scratch.inverse_transform(X_scratch)
reconstruction_error = np.mean((X - X_reconstructed)**2)
print(f"\nReconstruction error (MSE): {reconstruction_error:.6f}")

# Loadings
print("\nLoadings (components):")
print(pca_scratch.components_)
```

**Key Insights**:
- PCA: Find orthogonal directions of maximum variance
- Covariance matrix: C = X^T X / (n-1) for centered data
- Eigenvalues = variance explained by each component
- Eigenvectors = principal component directions
- np.linalg.eigh for symmetric matrices (faster, more stable)
- Sign of components is arbitrary (may flip vs sklearn)

</details>

---

## CATEGORY G: Coding Constraints

### Variation G1: Memory-Efficient Clustering

Cluster a large dataset with **limited memory** using Mini-Batch K-Means:

<details>
<summary>Solution G1</summary>

```python
import numpy as np
from sklearn.datasets import make_blobs
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import MiniBatchKMeans, KMeans
from sklearn.metrics import adjusted_rand_score
import time

# Create large dataset
n_samples = 100000
X, y_true = make_blobs(n_samples=n_samples, centers=5, random_state=2026)
X = StandardScaler().fit_transform(X)

print(f"Dataset size: {X.nbytes / 1e6:.2f} MB")

# Standard K-Means (memory intensive for large n)
start = time.time()
kmeans = KMeans(n_clusters=5, random_state=2026, n_init=1)
labels_km = kmeans.fit_predict(X)
time_km = time.time() - start
ari_km = adjusted_rand_score(y_true, labels_km)
print(f"K-Means: ARI={ari_km:.4f}, Time={time_km:.2f}s")

# Mini-Batch K-Means (memory efficient)
batch_sizes = [100, 500, 1000, 5000]

for batch_size in batch_sizes:
    start = time.time()
    mbkm = MiniBatchKMeans(
        n_clusters=5,
        batch_size=batch_size,
        random_state=2026,
        n_init=1
    )
    labels_mb = mbkm.fit_predict(X)
    time_mb = time.time() - start
    ari_mb = adjusted_rand_score(y_true, labels_mb)
    print(f"MiniBatch (batch={batch_size}): ARI={ari_mb:.4f}, Time={time_mb:.2f}s")

# Incremental learning (streaming data)
print("\n--- Incremental Learning ---")
mbkm_incremental = MiniBatchKMeans(n_clusters=5, random_state=2026)

chunk_size = 10000
for i in range(0, n_samples, chunk_size):
    chunk = X[i:i+chunk_size]
    mbkm_incremental.partial_fit(chunk)

labels_incremental = mbkm_incremental.predict(X)
ari_incremental = adjusted_rand_score(y_true, labels_incremental)
print(f"Incremental MiniBatch: ARI={ari_incremental:.4f}")
```

**Key Insights**:
- MiniBatchKMeans: Uses random subsets per iteration
- Much faster for large datasets
- Slightly lower quality than full K-Means
- partial_fit(): For streaming/online learning
- batch_size: Trade-off between speed and quality

</details>

### Variation G2: Time-Constrained Clustering

Your clustering must complete in **under 5 seconds** for 50,000 samples:

<details>
<summary>Solution G2</summary>

```python
import numpy as np
import time
from sklearn.datasets import make_blobs
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, MiniBatchKMeans, Birch
from sklearn.metrics import adjusted_rand_score

# Create dataset
n_samples = 50000
X, y_true = make_blobs(n_samples=n_samples, centers=5, random_state=2026)
X = StandardScaler().fit_transform(X)

# Time constraint
TIME_LIMIT = 5.0

def time_clustering(name, clusterer, X, y_true):
    start = time.time()
    labels = clusterer.fit_predict(X)
    elapsed = time.time() - start
    ari = adjusted_rand_score(y_true, labels)
    status = "✓" if elapsed < TIME_LIMIT else "✗"
    print(f"{status} {name}: ARI={ari:.4f}, Time={elapsed:.3f}s")
    return elapsed < TIME_LIMIT

# Test different algorithms
algorithms = {
    'KMeans (n_init=1)': KMeans(n_clusters=5, n_init=1, random_state=2026),
    'KMeans (n_init=10)': KMeans(n_clusters=5, n_init=10, random_state=2026),
    'MiniBatchKMeans': MiniBatchKMeans(n_clusters=5, batch_size=1000, random_state=2026),
    'Birch': Birch(n_clusters=5, threshold=0.5, branching_factor=50),
}

print(f"Time limit: {TIME_LIMIT}s for {n_samples} samples\n")

for name, alg in algorithms.items():
    time_clustering(name, alg, X, y_true)

# Optimized implementation
print("\n--- Optimized Solution ---")

class FastClustering:
    """Optimized clustering for time constraints."""

    def __init__(self, n_clusters=5):
        self.n_clusters = n_clusters
        # Use MiniBatchKMeans with optimized settings
        self.clusterer = MiniBatchKMeans(
            n_clusters=n_clusters,
            batch_size=1024,
            n_init=3,
            max_iter=100,
            random_state=2026
        )

    def fit_predict(self, X):
        return self.clusterer.fit_predict(X)

fast = FastClustering(n_clusters=5)
time_clustering('FastClustering', fast, X, y_true)
```

**Key Insights**:
- n_init: Number of random initializations (reduce for speed)
- MiniBatchKMeans: Almost always faster than KMeans
- Birch: Designed for large datasets, builds CF-tree
- For time constraints: MiniBatch > Birch > KMeans
- Trade-off: Speed vs. clustering quality

</details>

---

## KEY FORMULAS SUMMARY

| Concept | Formula/Definition |
|---------|-------------------|
| **K-Means Objective** | J = Σᵢ Σⱼ ‖xᵢ - μⱼ‖² (minimize intra-cluster variance) |
| **Silhouette Score** | s(i) = (b(i) - a(i)) / max(a(i), b(i)) |
| **Davies-Bouldin Index** | DB = (1/k) Σᵢ maxⱼ≠ᵢ (σᵢ + σⱼ) / d(cᵢ, cⱼ) |
| **Calinski-Harabasz** | CH = [B/(k-1)] / [W/(n-k)] where B=between, W=within |
| **Adjusted Rand Index** | ARI = (RI - E[RI]) / (max(RI) - E[RI]) |
| **PCA Objective** | max w^T Σ w subject to ‖w‖ = 1 |
| **Explained Variance** | λᵢ / Σλⱼ (eigenvalue ratio) |
| **LOF Score** | LOF(p) = Σ (lrd(o) / lrd(p)) / |N(p)| |
| **Isolation Forest Score** | s(x) = 2^(-E[h(x)]/c(n)) |
| **GMM Likelihood** | p(x) = Σₖ πₖ N(x; μₖ, Σₖ) |

---

## ATOMIC SKILLS CHECKLIST

- [ ] Load and explore unlabeled datasets
- [ ] Apply K-Means with proper initialization
- [ ] Choose k using elbow method, silhouette, gap statistic
- [ ] Apply DBSCAN with proper eps/min_samples selection
- [ ] Use hierarchical clustering with different linkages
- [ ] Interpret dendrograms
- [ ] Apply Gaussian Mixture Models
- [ ] Select GMM components using BIC/AIC
- [ ] Apply spectral clustering for non-convex clusters
- [ ] Perform PCA and interpret loadings
- [ ] Choose number of PCA components
- [ ] Use t-SNE/UMAP for visualization
- [ ] Detect anomalies with Isolation Forest
- [ ] Detect anomalies with One-Class SVM
- [ ] Evaluate clustering without ground truth
- [ ] Evaluate clustering with ground truth (ARI, NMI)
- [ ] Implement K-Means from scratch
- [ ] Implement DBSCAN from scratch
- [ ] Implement PCA from scratch
- [ ] Handle large datasets with mini-batch methods

---

## COMMON MISCONCEPTIONS

1. **K-Means always finds the global optimum**: WRONG. K-Means converges to local minima. Use multiple initializations (n_init) or K-Means++.

2. **More clusters = better**: WRONG. Overfitting exists in clustering too. Use metrics like silhouette or domain knowledge.

3. **DBSCAN finds the "right" number of clusters automatically**: PARTIALLY TRUE. DBSCAN doesn't need k, but eps and min_samples still affect cluster count.

4. **PCA components are features**: WRONG. PCA components are linear combinations of features. They're not directly interpretable as original features.

5. **t-SNE preserves global structure**: WRONG. t-SNE preserves local structure. Global distances are not meaningful in t-SNE plots.

6. **Scaling doesn't matter for clustering**: WRONG for distance-based methods (K-Means, DBSCAN, hierarchical). Features with larger ranges dominate.

7. **Silhouette score of 1.0 is always best**: NOT NECESSARILY. It might indicate trivial clustering (e.g., k=n).

8. **GMM is just soft K-Means**: OVERSIMPLIFICATION. GMM allows elliptical clusters and provides probabilistic assignments.

9. **Anomaly detection requires labeled anomalies**: WRONG. Methods like Isolation Forest and One-Class SVM are unsupervised.

10. **Clustering evaluation metrics are interchangeable**: WRONG. Different metrics favor different cluster shapes. Always use multiple metrics.

---

## EVALUATION METRICS COMPARISON

| Metric | Range | Optimal | Needs Labels | Notes |
|--------|-------|---------|--------------|-------|
| Silhouette | [-1, 1] | Higher | No | Most interpretable |
| Davies-Bouldin | [0, ∞) | Lower | No | Sensitive to outliers |
| Calinski-Harabasz | [0, ∞) | Higher | No | Fast to compute |
| Adjusted Rand Index | [-1, 1] | Higher | Yes | Chance-adjusted |
| Normalized MI | [0, 1] | Higher | Yes | Information-theoretic |
| Inertia | [0, ∞) | Lower | No | Always decreases with k |

---

## ALGORITHM SELECTION GUIDE

| Scenario | Recommended Algorithm |
|----------|----------------------|
| Unknown k, spherical clusters | K-Means + elbow/silhouette |
| Unknown k, arbitrary shapes | DBSCAN |
| Need soft assignments | GMM |
| Non-convex clusters | Spectral Clustering |
| Very large dataset | MiniBatchKMeans, Birch |
| Hierarchical structure needed | Agglomerative |
| Visualization only | t-SNE, UMAP |
| Anomaly detection | Isolation Forest, LOF |
| Need transform() for new data | K-Means, GMM, UMAP |

