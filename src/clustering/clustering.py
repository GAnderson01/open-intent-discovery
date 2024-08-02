import os

import numpy as np
import torch
if torch.cuda.is_available():
    from cuml.cluster import KMeans as cuKMeans
    from cuml.cluster import DBSCAN as cuDBSCAN
from sklearn.cluster import DBSCAN, KMeans
from tqdm import tqdm

from clustering.ITER_DBSCAN import ITER_DBSCAN
from supported_config import supported_clustering_algorithms
from utils.io import generate_config_str, save


class ClusteringModel():
    def __init__(self, config, gpu_fit=False):
        """
        Initialises the clustering model object and sets up a filepath to save/load the tmp models.
        Final model is saved in ClusteringEvaluation

        Args:
            config (dict): Dictionary containing the configuration information including dataset 
        """
        self.config = config
        self.gpu_fit = gpu_fit

        # Set up filepath to save/load the models
        tmp_dir = '../models/tmp/'
        if not os.path.exists(tmp_dir):
            os.makedirs(tmp_dir)
        config_str = generate_config_str(self.config, 'clustering_algorithm')
        self.model_path = os.path.join(tmp_dir, "_".join([config_str, "cluster_model_param_idx_%s.pkl"]))
        print("\tStage 1: Clustering...")

    
    def clustering(self, X, algorithm, gpu_fit=False, **params):
        """
        Perform clustering on the given data using the specified clustering algorithm.
        All models are saved to tmp directory.

        Args:
            X (np.array): utterance embeddings
            algorithm (str): clustering algorithm of choice
            gpu_fit (bool, optional): whether or not to use the GPU clustering. Defaults to False.
            params (dict): clustering params to use
        """
        if algorithm != 'iter_dbscan':
            print('\t\tPerforming %s clustering, finding best clustering in the following parameter ranges: %s' % (algorithm, params))
        else:
            print('\t\tPerforming %s clustering, finding best clustering' % algorithm)
        for n in tqdm(params['range'], desc='\t\tClustering progress'):
            if algorithm == 'kmeans':
                if gpu_fit:
                    attempt = cuKMeans(n_clusters=n, random_state=42)
                else:
                    attempt = KMeans(n_clusters=n, random_state=42)
            elif algorithm == 'dbscan':
                if gpu_fit:
                    attempt = cuDBSCAN(eps=params['eps'], min_samples=params['min_samples'], metric='cosine', max_mbytes_per_batch=params['mbpb'])
                else:
                    attempt = DBSCAN(eps=params['eps'], min_samples=params['min_samples'], metric='cosine')
                params['eps'] += params['inc']
            elif algorithm == 'iter_dbscan':
                attempt = ITER_DBSCAN(initial_distance=params['hps'][n]['distance'],
                                    initial_minimum_samples=params['hps'][n]['minimum_samples'],
                                    delta_distance=params['hps'][n]['delta_distance'],
                                    delta_minimum_samples=params['hps'][n]['delta_minimum_samples'],
                                    max_iteration=params['hps'][n]['max_iteration'],
                                    gpu=gpu_fit,
                                    max_mb_per_batch=params['mbpb'],
                                    features='precomputed'
                                     )
            attempt.fit_predict(X)
            # Save the attempt model to file for later evaluation
            temp_cluster_model_path = os.path.join(self.model_path % n)
            save(temp_cluster_model_path, attempt)
    

    def run_clustering(self, X, params):
        """
        Perform clustering on the given data using the specified clustering algorithm.
        All models are saved to tmp directory.

        Args:
            X (np.array): utterance embeddings
            params (dict): clustering params to use
        """
        if self.config["clustering_algorithm"] not in supported_clustering_algorithms:
            print(f"\t\tClustering algorithm {self.config['clustering_algorithm']} not supported")
            return
        # Check if the models already exist
        cluster_model_paths = [self.model_path % n for n in params["range"]]
        if all([os.path.exists(path) for path in cluster_model_paths]):
            print('\t\tAll models exist')
            return

        self.clustering(X,
                        algorithm=self.config['clustering_algorithm'], 
                        gpu_fit=self.gpu_fit, 
                        **params)
