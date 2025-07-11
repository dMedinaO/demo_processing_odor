from sklearn.metrics import (
    silhouette_score,
    davies_bouldin_score,
    calinski_harabasz_score
    )
from sklearn.cluster import (
    KMeans, AffinityPropagation,
    AgglomerativeClustering,
    Birch,
    HDBSCAN,
    BisectingKMeans,
    DBSCAN,
    MeanShift,
    MiniBatchKMeans,
    OPTICS,
    SpectralClustering
    )
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import logging
import itertools
import json
import math

clustering_methods_map = {
    "KMeans": KMeans,
    "SpectralClustering": SpectralClustering,
    "AgglomerativeClustering": AgglomerativeClustering,
    "AffinityPropagation": AffinityPropagation,
    "Birch": Birch,
    "BisectingKMeans": BisectingKMeans,
    "HDBSCAN": HDBSCAN,
    "OPTICS": OPTICS,
    "MeanShift": MeanShift,
    "DBSCAN": DBSCAN
}

def get_kmeans_param_grid(n_clusters_range = range(2, 11)):
    algorithms = ["lloyd", "elkan"]
    return [
        {
            "id_test": f"kmeans_{n}_{algorithm}",
            "name_method": "KMeans",
            "params": {"n_clusters": n,
                        "random_state": 42,
                        "algorithm": algorithm}
        }
        for n in n_clusters_range
        for algorithm in algorithms
    ]

def get_agglomerative_param_grid(n_clusters_range = range(2, 11), metrics = None, linkages = None):
    if metrics is None:
        metrics = ["euclidean", "manhattan", "cosine"]
    if linkages is None:
        linkages = ["ward", "complete", "average", "single"]

    param_dicts = []
    for n, metric, linkage in itertools.product(n_clusters_range, metrics, linkages):
        if linkage == "ward" and metric != "euclidean":
            continue
        param_dicts.append({
            "id_test": f"agglomerative_{n}_{metric}_{linkage}",
            "name_method": "AgglomerativeClustering",
            "params": {
                "n_clusters": n,
                "metric": metric,
                "linkage": linkage
            }
        })
    return param_dicts

def get_spectral_param_grid(n_clusters_range = range(2, 11), affinities = None):
    if affinities is None:
        affinities = ["rbf", "nearest_neighbors"]
    return [
        {
            "id_test": f"spectral_{n}_{affinity}",
            "name_method": "SpectralClustering",
            "params": {
                "n_clusters": n,
                "random_state": 42,
                "affinity": affinity
            }
        }
        for n, affinity in itertools.product(n_clusters_range, affinities)
    ]

def get_affinity_propagation_param_grid(damping_values = None, preference_values = None):
    if damping_values is None:
        damping_values = [0.5, 0.7, 0.9]
    if preference_values is None:
        preference_values = [None, -50, -10, 0, 10, 50]

    param_dicts = []
    for damping, preference in itertools.product(damping_values, preference_values):
        pref_str = "default" if preference is None else preference
        param_dicts.append({
            "id_test": f"affinity_{damping}_{pref_str}",
            "name_method": "AffinityPropagation",
            "params": {
                "damping": damping,
                "preference": preference,
                "random_state": 42,
            }
        })
    return param_dicts

def get_birch_param_grid(n_clusters_range = range(2, 11), thresholds = None, branching_factors = None):
    if thresholds is None:
        thresholds = [0.3, 0.5, 0.7]
    if branching_factors is None:
        branching_factors = [25, 50, 75, 100]

    param_dicts = []
    for n, t, b in itertools.product(n_clusters_range, thresholds, branching_factors):
        param_dicts.append({
            "id_test": f"birch_{n}_{t}_{b}",
            "name_method": "Birch",
            "params": {
                "n_clusters": n,
                "threshold": t,
                "branching_factor": b
            }
        })
    return param_dicts

def get_bisecting_kmeans_param_grid(n_clusters_range = range(2, 11), algorithm = None, bisecting_strategy = None):
    
    if bisecting_strategy is None:
        bisecting_strategy = ["biggest_inertia", "largest_cluster"]
    if algorithm is None:
        algorithm = ["lloyd", "elkan"]

    param_dicts = []
    for n, strategy, algo in itertools.product(n_clusters_range, bisecting_strategy, algorithm):
        param_dicts.append({
            "id_test": f"bisecting_{n}_{strategy}_{algo}",
            "name_method": "BisectingKMeans",
            "params": {
                "n_clusters": n,
                "random_state": 42,
                "algorithm": algo,
                "bisecting_strategy": strategy
            }
        })
    return param_dicts

def generate_selected_param_grids(include = ("KMeans", "AgglomerativeClustering", "SpectralClustering", 
                                        "AffinityPropagation", "Birch", "BisectingKMeans")):
    grids = []

    if "KMeans" in include:
        grids += get_kmeans_param_grid()
    if "AgglomerativeClustering" in include:
        grids += get_agglomerative_param_grid()
    if "SpectralClustering" in include:
        grids += get_spectral_param_grid()
    if "AffinityPropagation" in include:
        grids += get_affinity_propagation_param_grid()
    if "Birch" in include:
        grids += get_birch_param_grid()
    if "BisectingKMeans" in include:
        grids += get_bisecting_kmeans_param_grid()

    return grids


def compute_proportion_labels(labels):
    unique, counts = np.unique(labels, return_counts = True)
    total = float(len(labels))
    proportions = {label: round(100 * count / total, 2) for label, count in zip(unique, counts)}
    return proportions

def estimated_metrics(dataset, labels, name_method):

    try:
        return {
            "name_method" : name_method,
            "silhouette_score" : silhouette_score(dataset, labels),
            "davies_bouldin_score" : davies_bouldin_score(dataset, labels),
            "calinski_harabasz_score" : calinski_harabasz_score(dataset, labels),
            "n_labels" : len(np.unique(labels)),
            "proportion_labels" : compute_proportion_labels(labels)
            }
    except Exception as e:
        logging.warning(f"Metric calculation failed for method {name_method}: {e}")
        return {
            "name_method" : name_method,
            "silhouette_score" : np.nan,
            "davies_bouldin_score" : np.nan,
            "calinski_harabasz_score" : np.nan,
            "n_labels" : np.nan,
            "proportion_labels" : np.nan,
            }

def apply_clustering(name_method, dataset, **kwargs):
    if name_method not in clustering_methods_map:
        raise ValueError(f"Unknown clustering method: {name_method}")
    
    ClusteringClass = clustering_methods_map[name_method]
    model = ClusteringClass(**kwargs)
    model.fit(dataset)
    
    labels = getattr(model, "labels_", None)
    if labels is None and hasattr(model, "predict"):
        labels = model.predict(dataset)
    elif labels is None:
        raise ValueError(f"Unable to extract labels from model: {name_method}")
    
    metrics = estimated_metrics(dataset, labels, name_method)

    return model, metrics

def run_exploration(dataset, dict_params):
    
    matrix_result = []
    models = {}  # Dictionary to store models keyed by their id_test
    
    for param_set in dict_params:
        id_test = param_set.get("id_test", "unknown")
        name_method = param_set.get("name_method")
        params = param_set.get("params", {})
        
        try:
            model, metrics = apply_clustering(name_method, dataset, **params)
            row = {
                "id_test": id_test,
                "name_method": name_method, 
                "silhouette_score": metrics.get("silhouette_score"),
                "davies_bouldin_score": metrics.get("davies_bouldin_score"),
                "calinski_harabasz_score": metrics.get("calinski_harabasz_score"),
                "n_labels": metrics.get("n_labels"),
                "proportion_labels": metrics.get("proportion_labels"),
                }
            models[id_test] = model
            
        except Exception as e:
            logging.warning(f"Failed to apply clustering for test ID '{id_test}': {e}")
            row = {
                "id_test": id_test,
                "name_method": name_method,
                "silhouette_score": np.nan,
                "davies_bouldin_score": np.nan,
                "calinski_harabasz_score": np.nan,
                "n_labels": np.nan,
                "proportion_labels": np.nan
                }
            models[id_test] = None  # Explicitly store None for traceability
                
        matrix_result.append(row)

    return pd.DataFrame(matrix_result), models

def extract_cluster_labels(dataframe, model_dict, selected_models):
    
    dataframe_copy = dataframe.copy()
    
    for model in selected_models:
        if model in model_dict:
            dataframe_copy[model] = model_dict[model].labels_.astype(str)
            
        else:
            print(f"Warning: Model '{model}' not found in the dictionary.")
            
    return dataframe_copy

def extract_clusters(dataframe, selected_models, word_column = "word", save_to_json = False, file_name = "cluster_words"):
    
    dataframe_copy = dataframe.copy()

    clusters = {}
    
    for col in selected_models:
        cluster_dict = {}
        for label in dataframe_copy[col].unique():
            words = dataframe_copy[dataframe_copy[col] == label][word_column].tolist()
            cluster_dict[str(label)] = ', '.join(words)
        clusters[col] = cluster_dict
        
    if save_to_json:
        with open(f"../data/{file_name}.json", 'w', encoding = 'utf-8') as f:
            json.dump(clusters, f, indent = 4, ensure_ascii = False)
    
    return clusters

def plot_cluster_projections(df, clustering_columns, coordinates = "", tsne_cols = ('tsne_1', 'tsne_2'), umap_cols = ('umap_1', 'umap_2'), save_to_png = False, file_name = "clustering_words"):
    """
    Plots scatterplots of clustering results over t-SNE and UMAP coordinate systems.

    Parameters:
    - df: pandas DataFrame containing clustering columns and coordinates.
    - clustering_columns: list of column names (str) with cluster labels.
    - tsne_cols: tuple with column names for t-SNE coordinates.
    - umap_cols: tuple with column names for UMAP coordinates.
    """
    total_plots = len(clustering_columns) * 2  # Each clustering column gets 2 plots (t-SNE + UMAP)
    cols = 4
    rows = math.ceil(total_plots / cols)
    
    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 5 * rows))
    axes = axes.flatten()  # Make axes indexable in 1D

    for i, cluster_col in enumerate(clustering_columns):
        # Plot on t-SNE
        sns.scatterplot(
            data=df,
            x=tsne_cols[0],
            y=tsne_cols[1],
            hue=cluster_col,
            palette="tab10",
            ax=axes[2*i]
        )
        axes[2*i].set_title(f"{cluster_col}_{coordinates} on t-SNE")
        # axes[2*i].legend().remove()

        # Plot on UMAP
        sns.scatterplot(
            data=df,
            x=umap_cols[0],
            y=umap_cols[1],
            hue=cluster_col,
            palette="tab10",
            ax=axes[2*i + 1]
        )
        axes[2*i + 1].set_title(f"{cluster_col}_{coordinates} on UMAP")
        axes[2*i + 1].legend().remove()

    # Remove any extra subplots
    for j in range(2 * len(clustering_columns), len(axes)):
        fig.delaxes(axes[j])

    plt.tight_layout()
    plt.show()
    
    if save_to_png:
        plt.savefig(f'../figures/{file_name}.png', dpi = 300, transparent = True, bbox_inches = "tight")