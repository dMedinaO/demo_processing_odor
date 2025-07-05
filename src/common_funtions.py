from sklearn.metrics import (silhouette_score, davies_bouldin_score, calinski_harabasz_score)
import numpy as np

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