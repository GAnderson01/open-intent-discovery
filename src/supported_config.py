import numpy as np
import os
import random

from utils.io import load, save


def generate_all_iter_dbscan_hyperparameters(n):
    hp_options = {'distance': np.arange(0.01, 0.11, 0.01), 
                      'minimum_samples': range(5, 50, 10), 
                      'delta_distance':  np.arange(0.01, 0.11, 0.01),
                      'delta_minimum_samples': range(1, 10, 10), 
                      'max_iteration': range(25, 100, 25)}
    all_params = []
    for i in range(n):
        all_params.append({'distance': random.choice(hp_options['distance']),
                  'minimum_samples': random.choice(hp_options['minimum_samples']),
                  'delta_distance':  random.choice(hp_options['delta_distance']),
                  'delta_minimum_samples': random.choice(hp_options['delta_minimum_samples'],),
                  'max_iteration': random.choice(hp_options['max_iteration'])})
    return all_params


# Embedding Models
bert_based_models = ['albert-base-v2', 'bert-base-uncased', 'roberta-base']
sentence_transformer_models = ['all-mpnet-base-v2']
tf_hub_models = ['use']
supported_embedding_models = bert_based_models + sentence_transformer_models + tf_hub_models
embedding_model_urls = {'use': "https://tfhub.dev/google/universal-sentence-encoder/4"}

# Clustering
supported_clustering_algorithms = ['kmeans', 'dbscan', 'iter_dbscan']
supported_clustering_metrics = ['balanced', 'davies_bouldin', 'silhouette']

params = {}
# - KMeans parameter tuning range
min_clusters = 2
max_clusters = 200
params["kmeans"] = {"range": range(min_clusters, max_clusters)}
# - DBSCAN parameter tuning range
n = 100
mbpb = 12 * 1024
params["dbscan"] = {"range": range(n), "eps": 0.1, "inc": 0.01, "min_samples": 5, "mbpb": mbpb}
# - ITER_DBSCAN parameter tuning range
iter_dbscan_hps_path = '../models/all_iter_dbscan_hps.pkl'
if os.path.exists(iter_dbscan_hps_path):
    all_iter_dbscan_hps = load(iter_dbscan_hps_path)
else:
    all_iter_dbscan_hps = generate_all_iter_dbscan_hyperparameters(n)
    save(iter_dbscan_hps_path, all_iter_dbscan_hps)
params["iter_dbscan"] = {"range": range(n), "hps": all_iter_dbscan_hps, "mbpb": mbpb}

# Candidate Extraction
supported_extraction_methods = ['action-object', 'action-object-ext', 't0pp-prompting']
extraction_prompt_template = "Given the following utterance: %s. The intent was to"

# Intent Label Generation
supported_label_generation_methods = ['most-frequent', 't0pp-prompting']
ilg_prompt_template = "Given these utterances: %s. What is the best fitting intent, if any, among the following: %s"