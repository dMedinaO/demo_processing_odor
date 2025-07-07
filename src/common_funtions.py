from sklearn.metrics import (silhouette_score, davies_bouldin_score, calinski_harabasz_score)
import numpy as np
import pandas as pd

def estimated_metrics(dataset, labels, name_method):

    try:
        return {
            "name_method" : name_method,
            "silhouette_score" : silhouette_score(dataset, labels),
            "davies_bouldin_score" : davies_bouldin_score(dataset, labels),
            "calinski_harabasz_score" : calinski_harabasz_score(dataset, labels)}
    except:
        return {
            "name_method" : name_method,
            "silhouette_score" : np.nan,
            "davies_bouldin_score" : np.nan,
            "calinski_harabasz_score" : np.nan}
    
def apply_clustering(clustering_method, dataset, name_method, **kwargs):
    
    model = clustering_method(**kwargs)
    model.fit(dataset)
    metrics = estimated_metrics(dataset, model.labels_, name_method)

    return model, metrics

def get_proportion_labels(labels):
    unique, counts = np.unique(labels, return_counts=True)
    proportions = dict(zip(unique, counts / len(labels)))
    return proportions
    
def run_exploration(dataset, dict_params, clustering_method, name_method):
    
    matrix_result = []
    
    for element in dict_params:
        id_test = element.get("id_test", None)
        model, metrics = apply_clustering(clustering_method,
                                      dataset,
                                      name_method
                                      **element.get("params", {}))
        row = {"id_test": id_test, "name_method": name_method, 
               "silhouette_score": metrics["silhouette_score"],
               "n_labels": len(np.unique(model.labels_)),
               "proportion_labels": get_proportion_labels(model.labels_),
               "davies_bouldin_score": metrics["davies_bouldin_score"],
               "calinski_harabasz_score": metrics["calinski_harabasz_score"]}
        matrix_result.append(row)
    
    df_result = pd.DataFrame(matrix_result)
    return df_result