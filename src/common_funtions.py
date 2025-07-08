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
import numpy as np
import pandas as pd
import logging
import itertools

clustering_methods_map = {
    "KMeans": KMeans,
    "SpectralClustering": SpectralClustering,
    "AgglomerativeClustering": AgglomerativeClustering,
    "AffinityPropagation": AffinityPropagation,
    "Birch": Birch,
    "BisectingKMeans": BisectingKMeans
}

def get_kmeans_param_grid(n_clusters_range = range(2, 11)):
    return [
        {
            "id_test": f"kmeans_{n}",
            "name_method": "KMeans",
            "params": {"n_clusters": n}
        }
        for n in n_clusters_range
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
                "affinity": affinity
            }
        }
        for n, affinity in itertools.product(n_clusters_range, affinities)
    ]

def get_affinity_propagation_param_grid(damping_values = None, preferences = None):
    if damping_values is None:
        damping_values = [0.5, 0.7, 0.9]
    if preferences is None:
        preferences = [None, -50, -100]

    param_dicts = []
    for damping, preference in itertools.product(damping_values, preferences):
        param_dicts.append({
            "id_test": f"affinity_{damping}_{preference}",
            "name_method": "AffinityPropagation",
            "params": {
                "damping": damping,
                "preference": preference
            }
        })
    return param_dicts

def get_birch_param_grid(n_clusters_range = range(2, 11), thresholds = None, branching_factors = None):
    if thresholds is None:
        thresholds = [0.3, 0.5, 0.7]
    if branching_factors is None:
        branching_factors = [25, 50]

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

def get_bisecting_kmeans_param_grid(n_clusters_range = range(2, 11), bisecting_strategy = None):
    if bisecting_strategy is None:
        bisecting_strategy = ["largest_cluster", "largest_variance"]

    param_dicts = []
    for n, strategy in itertools.product(n_clusters_range, bisecting_strategy):
        param_dicts.append({
            "id_test": f"bisecting_{n}_{strategy}",
            "name_method": "BisectingKMeans",
            "params": {
                "n_clusters": n,
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

def apply_clustering(name_method, clustering_methods_map, dataset, **kwargs):
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

def run_exploration(dataset, dict_params, clustering_method_map):
    
    matrix_result = []
    
    for param_set in dict_params:
        id_test = param_set.get("id_test", "unknown")
        name_method = param_set.get("name_method")
        params = param_set.get("params", {})
        
        try:
            model, metrics = apply_clustering(name_method, clustering_methods_map, dataset, **params)
            row = {
                "id_test": id_test,
                "name_method": name_method, 
                "silhouette_score": metrics.get("silhouette_score"),
                "davies_bouldin_score": metrics.get("davies_bouldin_score"),
                "calinski_harabasz_score": metrics.get("calinski_harabasz_score"),
                "n_labels": metrics.get("n_labels"),
                "proportion_labels": metrics.get("proportion_labels"),
                }
            
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
                
        matrix_result.append(row)

    return pd.DataFrame(matrix_result)

def generate_clustering_param_dicts(
    n_clusters_range=range(3, 11),
    metrics = None,
    linkages = None
):
    
    if metrics is None:
        metrics = ["euclidean", "l1", "l2", "manhattan", "cosine", "precomputed"]
    if linkages is None:
        linkages = ["ward", "complete", "average", "single"]

    param_dicts = []

    for n_clusters, metric, linkage in itertools.product(n_clusters_range, metrics, linkages):
        if linkage == "ward" and metric != "euclidean":
            continue
        
        id_test = f"{n_clusters}_{metric}_{linkage}"
        
        param_dicts.append({
            "id_test": id_test,
            "params": {
                "n_clusters": n_clusters,
                "metric": metric,
                "linkage": linkage
            }
        })

    return param_dicts

