import os
from statistics import mean, stdev

import pandas as pd
import torch
if torch.cuda.is_available():
    from cuml.metrics.cluster.silhouette_score import cython_silhouette_samples as cu_silhouette_samples
    from cuml.metrics.cluster.silhouette_score import cython_silhouette_score as cu_silhouette_score
from sklearn.metrics import davies_bouldin_score, silhouette_samples, silhouette_score
from tqdm import tqdm

from supported_config import supported_clustering_metrics
from utils.io import generate_config_str, load, save


class ClusteringEvaluation():
    def __init__(self, config, gpu_eval=False):
        """
        Initialises the clustering evaluation object and sets up a filepath to save/load the final cluster model.

        Args:
            config (dict): Dictionary containing the configuration information including dataset 
            gpu_eval (bool, optional): whether or not to use the GPU for cluster evaluation. Defaults to False
        """
        self.config = config
        self.gpu_eval = gpu_eval

        # Set up filepath to save/load the best models
        self.model_dir = "../models/"
        config_str = generate_config_str(self.config, "clustering_measure")
        self.cluster_model_path = os.path.join(self.model_dir, "_".join([config_str, "cluster_model.pkl"]))
        print("\tStage 1: Clustering Evaluation...")

    def get_best_params(self, X):
        """
        Evaluate all clustering attempts and return (and save) the best according to the specified metric.
        
        Args:
            X (np.array): numpy array of the embeddings
            
        Returns:
            The best cluster model according to the metric
        """
        if self.config["clustering_measure"] not in supported_clustering_metrics:
            print(f"\t\tClustering measure {self.config['clustering_measure']} not supported")
            return
        # Load in and return the best clustering models if they exist on file
        print(f"\t\tCluster model path: {self.cluster_model_path}")
        if os.path.exists(self.cluster_model_path):
            print(f"\t\tModel {self.cluster_model_path} exists")
            print("\t\tLoading existing model(s)...")
            return load(self.cluster_model_path)
        
        # Load in the attempt models
        temp_dir = os.path.join(self.model_dir, "tmp")
        attempts = []
        config = [self.config["dataset"], 
                  self.config["embedding_model_name"], 
                  self.config["clustering_algorithm"]]

        for tmp_cluster_model_path in os.listdir(temp_dir):
            if all(x in tmp_cluster_model_path for x in config):
                if self.config["clustering_algorithm"] == "iter_dbscan":
                    attempts.append(pd.read_csv(os.path.join(temp_dir, tmp_cluster_model_path)))
                else:
                    if "iter_dbscan" in tmp_cluster_model_path:
                        continue
                    attempts.append(load(os.path.join(temp_dir, tmp_cluster_model_path)))
                    
        if self.config["clustering_algorithm"] == "iter_dbscan":
            label_attempts = [attempt["cluster_id"].tolist() for attempt in attempts]
        else:
            label_attempts = [attempt.labels_ for attempt in attempts]

        # Evaluate the models using the specified measure and save and return the best model
        for labels in label_attempts:
            assert(len(labels) == len(X))
        scores = evaluate_clusters(X, label_attempts, self.config['clustering_measure'], self.gpu_eval)
        best_idx = get_idx_of_best_attempt(scores, self.config['clustering_measure'])
        self.model = attempts[best_idx]
        print(f"\t\tNumber clusters in model {best_idx}: {len(set(self.model.labels_))}")
        save(self.cluster_model_path, self.model)
        return self.model


def balanced_score(X, labels, lambda_=0.5, gpu=False):
    """
    Implementation of Liu et al. (2021) balanced score
    """
    assert(len(labels) == len(X))
    if gpu:
        s = cu_silhouette_samples(X, labels)
        return mean(s.get() - penalty(labels, lambda_))

    s = silhouette_samples(X, labels)
    return mean(s - penalty(labels, lambda_))


def evaluate_clusters(X, label_attempts, measure, gpu=False):
    scores = []
    for i, labels in tqdm(enumerate(label_attempts), desc=f'\t\tEvaluating all clustering attempts using {measure}'):
        num_clusters = len(set(labels))
        assert(len(labels) == len(X))
        if measure == 'balanced':
            if num_clusters < 2 or num_clusters > len(labels) - 1:
                scores.append(float('-inf'))
            else:
                assert(len(labels) == len(X))
                scores.append(balanced_score(X, labels, gpu=gpu))
        elif measure == 'silhouette':
            if num_clusters < 2 or num_clusters > len(labels) - 1:
                scores.append(float('-inf'))
            else:
                if gpu:
                    scores.append(cu_silhouette_score(X, labels))
                else:
                    scores.append(silhouette_score(X, labels))
        elif measure == 'davies_bouldin':
            if num_clusters < 2 or num_clusters > len(labels) - 1:
                scores.append(float('inf'))
            else:
                scores.append(davies_bouldin_score(X, labels))
    return scores


def get_idx_of_best_attempt(scores, measure):
    if measure in ['balanced', 'silhouette']:
        best_value = max(scores)
    elif measure in ['davies_bouldin']:
        best_value = min(scores)
    return scores.index(best_value)


def penalty(labels, lambda_):
    N = len(labels)
    K = len(set(labels))
    pen = 0
    cluster_sizes = []
    for k in set(labels):
        mag_k = len([i for i in labels if i == k])
        cluster_sizes.append(mag_k)
        pen += abs((mag_k / N) - (1 / K))
    pen = pen * lambda_ * stdev(cluster_sizes) / mean(cluster_sizes)
    return pen