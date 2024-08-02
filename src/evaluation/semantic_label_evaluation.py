import json
import os

import pandas as pd
import torch

from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm

from evaluation.bart_score import BARTScorer
from intent_label_generation.intent_label_generation import clean_generated_labels, clean_intent_labels
from intent_label_generation.readable_intents import readable_intents
from utils.io import generate_config_str

print('Loading BARTScorer Model')
if torch.cuda.device_count() > 1:
    device = 'cuda:1'
else:
    device='cuda:0'

bart_scorer = BARTScorer(device=device, checkpoint='facebook/bart-large-cnn')
bart_scorer.load(path='../../../BARTScore/models/bart_score.pth')


class SemanticLabelEvaluator:
    def __init__(self, config):
        """
        Initialises the semantic label evaluator object and sets up the filepaths to save/load the results.

        Args:
            config (dict): Dictionary containing the configuration information including dataset 
        """
        self.dataset = config["dataset"]
        # Set up the filepaths to save/load the labels dataframe and scores
        results_dir = "../results"
        config_str = generate_config_str(config)
        config_dir = os.path.join(results_dir, config_str)
        df_labels_filename = "df_labels.csv"
        avg_sim_filename = "avg_sim.json"
        avg_bart_filename = "avg_bart.json"

        self.df_labels_filepath = os.path.join(config_dir, df_labels_filename)
        self.avg_sim_filepath = os.path.join(config_dir, avg_sim_filename)
        self.avg_bart_filepath = os.path.join(config_dir, avg_bart_filename)

        if not os.path.isdir(config_dir):
            os.makedirs(config_dir)
            print(f"Created folder: {config_dir}")
        else:
            print(f"Folder {config_dir} already exists")

    
    def evaluate(self, df, embedding_model):
        """
        Perform evaluation on the specified configuration's results with a given embedding model.

        Args:
            df (pd.DataFrame): dataframe containing the texts and generated intents
            embedding_model (_type_): _description_
        """  
        # Create a mapping from cluster id to generated label
        cluster_assignments = {}
        cluster_ids = sorted(df['cluster_id'].unique().tolist())
        for c in cluster_ids:
            cluster_assignments[c] = df[df['cluster_id'] == c].iloc[0]['generated']
        
        # Create a mapping from cluster id to the most common ground truth
        cluster_id_to_most_common_gt = {}
        for c in cluster_ids:
            cluster_id_to_most_common_gt[c] = df[df['cluster_id'] == c]['intent'].value_counts().index[0]
        
        # Create a dataframe with a entry for each cluster id with its corresponding generated label and most common ground truth intent
        df_labels = pd.DataFrame()
        df_labels['cluster_id'] = cluster_ids
        df_labels['generated'] = df_labels['cluster_id'].apply(lambda x: cluster_assignments[x])
        df_labels['most_common_gt'] = df_labels['cluster_id'].apply(lambda x: cluster_id_to_most_common_gt[x])
        df_labels
        
        # Get the most common ground truth intents and convert to readable intents
        mcgt_intents = list(df_labels['most_common_gt'])
        dataset_readable_intents = readable_intents[self.dataset] if self.dataset in readable_intents else {}
        mcgt_intents = [dataset_readable_intents[intent] if intent in dataset_readable_intents else intent for intent in mcgt_intents]
        print(f"Most common ground truth intents: {mcgt_intents}")

        # Get the generated intents
        generated_intents = list(df_labels['generated'])

        # Check if ground truth labels are in pascal_case
        pascal_case = True
        if self.dataset in ['atis', 'banking77', 'bitext', 'clinc', 'personal_assistant', 'stack_overflow', 'twacs']:
            pascal_case = False

        # Clean the most common ground truth intents
        clean_intents = clean_intent_labels(mcgt_intents, pascal_case)
        print(f"Cleaned most common ground truth intents: {clean_intents}")
        # Clean the generated labels
        clean_generated_intents = clean_generated_labels(generated_intents)
        print(f"Cleaned generated labels: {clean_generated_intents}")
        
        # Get embeddings for both mcgt and generated labels
        intent_embeddings = embedding_model(clean_intents)
        intent_embeddings = intent_embeddings.numpy()
        generated_intent_embeddings = embedding_model(clean_generated_intents)
        generated_intent_embeddings = generated_intent_embeddings.numpy()
        
        # Get cosine similarity scores between the mcgt and generated labels and append to labels dataframe
        sims = []
        for i in tqdm(range(len(generated_intents)), desc="Calculating cosine similarities..."):
            sims.append(cosine_similarity(generated_intent_embeddings[i].reshape(1, -1),
                                          intent_embeddings[i].reshape(1, -1))[0][0])
        df_labels['cosine_sim'] = sims
        
        # Get average cosine similarity and save results to file
        all_intents = df['intent'].unique().tolist()
        avg_sim_per_label = {}
        for intent in tqdm(all_intents, desc="Calculating average cosine similarity..."):
            df_intent = df_labels[df_labels['most_common_gt'] == intent]['cosine_sim']
            if len(df_intent) > 0:
                avg_sim_per_label[intent] = df_intent.sum()/len(df_intent)
            else:
                avg_sim_per_label[intent] = 0.0
                
        avg_sim = sum(v for v in avg_sim_per_label.values())/len(all_intents)
        
        avg_sim_per_label['avg'] = avg_sim
        with open(self.avg_sim_filepath, 'w') as f:
            json.dump(avg_sim_per_label, f)
        
        # Get BART scores between the mcgt and generated labels and append to labels dataframe
        bart_scores = []
        for i in tqdm(range(len(generated_intents)), desc="Calculating BART scores..."):
            bart_scores.append(bart_scorer.score([clean_generated_intents[i]],
                                          [clean_intents[i]]))
        df_labels['bart_score'] = [x[0] for x in bart_scores]
        print(df_labels)
        print('Saving labels dataframe')
        df_labels.to_csv(self.df_labels_filepath)
        
        # Get average BART score and save results to file
        avg_bart_per_label = {}
        for intent in tqdm(all_intents, desc="Calculating average BART score..."):
            df_intent = df_labels[df_labels['most_common_gt'] == intent]['bart_score']
            if len(df_intent) > 0:
                avg_bart_per_label[intent] = df_intent.sum()/len(df_intent)
            else:
                avg_bart_per_label[intent] = bart_scorer.score([''], [intent])[0]
        avg_bart = sum(v for v in avg_bart_per_label.values())/len(all_intents)
        
        avg_bart_per_label['avg'] = avg_bart
        with open(self.avg_bart_filepath, 'w') as f:
            json.dump(avg_bart_per_label, f)
        