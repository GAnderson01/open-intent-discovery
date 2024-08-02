import os
import re
from collections import Counter, OrderedDict

import numpy as np
import pandas as pd
import torch
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm

from supported_config import ilg_prompt_template, supported_label_generation_methods
from utils.io import generate_config_str, load


class IntentLabelGenerator:
    def __init__(self, config):
        """
        Initialises the label selection object and sets up a filepath to save/load the final dataframe with intent labels.

        Args:
            config (dict): Dictionary containing the configuration information including dataset 
        """
        self.config = config
        self.generation_method = config["generation_method"]
        self.clustering_algorithm = config["clustering_algorithm"]

        # Set up the filepath to save/load the final dataframe with intent labels
        results_dir = "../results/"
        config_str = generate_config_str(config)
        config_dir = os.path.join(results_dir, config_str)
        if not os.path.exists(config_dir):
            os.makedirs(config_dir)
        self.generation_df_filename = os.path.join(config_dir, "generation_df.csv")
        print("\tStage 2: Intent Label Generation...")
    

    def generate(self, df, cluster_labels, verbose=False):
        """
        Perform intent label selection on the given data using the specified selection method.
        Loads from file if it already exists.

        Args:
            df (pd.DataFrame): dataframe containing the texts and candidate intents
            cluster_labels (np.array/list): numpy array or list of cluster labels
            verbose (bool, optional): whether or not to output the prompts and response. Defaults to False.

        Returns:
            pd.DataFrame: original dataframe with added 'generated' intent column
        """
        if self.generation_method not in supported_label_generation_methods:
            print(f"\t\tGeneration method: {self.generation_method} is not supported")
            return None
        # Load in and return the final dataframe if it exists on file
        if os.path.exists(self.generation_df_filename):
            print(f"\t\tLoading in {self.generation_df_filename}")
            return pd.read_csv(self.generation_df_filename, index_col=0)
        
        # Perform intent label selection using specified method
        df["cluster_id"] = cluster_labels
        if self.generation_method == "most-frequent":
            df = self.get_n_most_frequent(df, 1)
        elif self.generation_method == "t0pp-prompting":
            df = self.t0pp_prompting_get_intents(df, 3, verbose=verbose)

        # Save final dataframe with intents to file
        df.to_csv(self.generation_df_filename)
        return df

    def get_n_most_frequent(self, df, n=1):
        """Choose intent label as the most frequent candidates in each cluster

        Args:
            df (pd.DataFrame): dataframe containing the texts and candidate intents
            n (int): top n most frequent candidiates are chosen to label the cluster

        Returns:
            pd.DataFrame: original dataframe with added 'generated' intent column
        """
        # Get the unique cluster ids
        cluster_id_set = df["cluster_id"].unique()
        # Get the most common action object pairs in each cluster
        cluster_id_to_generated = {}
        for cluster_id in cluster_id_set:
            df_subset = df.loc[(df['cluster_id'] == cluster_id) & (~df['candidate_intent'].astype('str').str.contains('NONE'))]
            pair_counter = Counter(df_subset['candidate_intent'].tolist())
            most_frequent = pair_counter.most_common(n)
            if len(most_frequent) > 0:
                cluster_id_to_generated[cluster_id] = most_frequent[0][0]
            else:
                cluster_id_to_generated[cluster_id] = ''
        # Add the most common action object pairs to the dataframe as the final intent labels
        df['generated'] = df['cluster_id'].apply(lambda x: cluster_id_to_generated[x])
        return df

    
    def t0pp_prompting_get_intents(self, df, n, verbose=False):
        """Generate intent label by prompting T0pp with all utterances in each clusters and a list of n candidates 

        Args:
            df (pd.DataFrame): dataframe containing the texts and candidate intents
            n (int): number of most frequent candidates to provide to T0pp
            verbose (bool, optional): whether or not to output the prompts and response. Defaults to False.

        Returns:
            pd.DataFrame: original dataframe with added 'generated' intent column
        """
        import utils.t0pp as T0ppModel
        # Get the unique cluster ids
        cluster_id_set= df["cluster_id"].unique()
        
        # Order the utterances in the clusters by their similarity
        # (Only implemented for kmeans)
        if self.clustering_algorithm == "kmeans":
            all_cluster_utterances = order_kmeans_clusters(df, cluster_id_set, self.config)
        
        cluster_id_to_generated = {}
        for c in tqdm(cluster_id_set, desc="\t\tT0pp Prompting label generation progress"):
            # Get samples in this cluster
            df_subset = df.loc[df["cluster_id"] == c]
            # Get the utterances for this cluster
            if self.clustering_algorithm == "kmeans":
                cluster_utterances = list(OrderedDict.fromkeys(all_cluster_utterances[c]))
            else:
                cluster_utterances = df_subset["text"].tolist()
            
            # Get top candidate intents (ignoring any that have NONE in their candidate intent)
            all_candidates = df_subset[(~df_subset["candidate_intent"].astype("str").str.contains("NONE"))]["candidate_intent"].tolist()
            candidates_counter = Counter(all_candidates)
            candidates = [str(cand[0]) for cand in candidates_counter.most_common(top)]
            candidates = ", ".join(candidates)

            # Build the prompts, ensuring we don't go over the 512 token limit for T0pp
            prompts = []
            if type(cluster_utterances) is np.ndarray:
                cluster_utterances = cluster_utterances.tolist()
            prompt_string = ""
            length = len(ilg_prompt_template.split())
            while len(cluster_utterances) > 0:
                if len((", " + cluster_utterances[0]).split()) + length > 512:
                    prompts.append(ilg_prompt_template % (prompt_string, candidates))
                    length = len(ilg_prompt_template.split())
                    prompt_string = ''
                    continue
                u = cluster_utterances.pop(0)
                if len(prompt_string) == 0:
                    prompt_string = u
                else:
                    prompt_string = f"{prompt_string}, {u}"
                length = len((ilg_prompt_template % (prompt_string, candidates)).split())
            prompts.append(ilg_prompt_template % (prompt_string, candidates))
            # Inference
            responses = [T0ppModel.inference(p) for p in prompts]
            response_counter = Counter(responses)
            cluster_id_to_generated[int(c)] = response_counter.most_common(1)[0][0]
            if verbose:
                for i, p in enumerate(prompts):
                    print(f"\t\tPrompt: {p}")
                    print(f"\t\tResponse: {responses[i]}")
                print(f"\t\tGenerated label for {c}: {cluster_id_to_generated[c]}")
        
        # Tidy up model
        torch.cuda.empty_cache()

        # Add generated labels to dataframe
        df["generated"] = df["cluster_id"].apply(lambda x: cluster_id_to_generated[x])
        return df
        

def order_kmeans_clusters(df, cluster_ids, config):
    """Order the utterances by their distance to their cluster centres

    Args:
        df (pd.DataFrame): dataframe containing the texts and candidate intents
        cluster_ids (np.array/list): set of unique cluster ids
        config (dict): Dictionary containing the configuration information including dataset 

    Returns:
        dict: dictionary mapping cluster ids to a sorted list of their utterance members
    """
    embeddings_dir = '../data/interim/embeddings/'
    config_str = generate_config_str(config, 'embedding_model_name')
    utterance_embeddings_filename = '_'.join([config_str, 'utterances.pkl'])
    embeddings_path = os.path.join(embeddings_dir, utterance_embeddings_filename)
    embeddings = load(embeddings_path)

    model_dir = '../models/'
    config_str = generate_config_str(config, 'clustering_measure')
    cluster_model_path = os.path.join(model_dir, '_'.join([config_str, 'cluster_model.pkl']))
    cluster_model = load(cluster_model_path)

    all_cluster_utterances = {}
    for c in cluster_ids:
        indices = df[df['cluster_id'] == c].index.tolist()
        embs = [embeddings[i] for i in indices]
        center = cluster_model.cluster_centers_[c]
        cos_sim = zip(indices, cosine_similarity(embs, center.reshape(1, -1)))
        cos_sim = sorted(cos_sim, key=lambda x: x[1], reverse=True)
        ord_indices = [x[0] for x in cos_sim]
        all_cluster_utterances[c] = [df.loc[i]['text'] for i in ord_indices]
    return all_cluster_utterances
        

def clean_generated_labels(labels):
    labels = [x.replace('_', ' ') if type(x) == str else '' for x in labels]
    labels = [x.replace('-', ' ') for x in labels]
    labels = [x.lower() for x in labels]
    return labels


def clean_intent_labels(labels, pascal_case=True):
    if pascal_case:
        labels = [' '.join(re.findall('[A-Z][^A-Z]*', x)) for x in labels]
    labels = [x.replace('_', ' ') for x in labels]
    labels = [x.replace('-', ' ') for x in labels]
    labels = [x.lower() for x in labels]
    return labels