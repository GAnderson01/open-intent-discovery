"""
The SemanticRepresentation class is used to obtain the embeddings for a list of utterances from a specified embedding model.
The embeddings are saved to file and will be loaded from file in future if the same config is provided.
"""

import os

import numpy as np
import tensorflow_hub as hub
import torch
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel

from supported_config import bert_based_models, embedding_model_urls, sentence_transformer_models, supported_embedding_models
from utils.io import generate_config_str, load, save

gpu = torch.cuda.is_available()
print(gpu)

class SemanticRepresentation:
    def __init__(self, config):
        """
        Initialises the semantic representation object and sets up a filepath to save/load the embeddings.

        Args:
            config (dict): Dictionary containing the configuration information including dataset 
        """
        self.config = config
        self.model = None
        self.model_name = config["embedding_model_name"]

        # Set up the filepaths to save/load the embeddings
        embeddings_dir = os.path.join("..", "data", "interim", "embeddings")
        config_str = generate_config_str(config, "embedding_model_name")
        config_str_gen = generate_config_str(config, "generation_method")
        
        self.embedding_filepath = os.path.join(embeddings_dir, "_".join([config_str, "utterances.pkl"]))
        print("\tStage 1: Semantic Representation...")
    

    def __load_model(self):
        """
        Load the specified embedding model into memeory.
        """
        if self.model_name not in supported_embedding_models:
            print(f"\t\tModel {self.model_name} not supported")
            return
        print(f"\t\tLoading embedding model {self.model_name}...")
        if self.model_name == "use":
            self.model = load_universal_sentence_encoder()
        elif self.model_name in sentence_transformer_models:
            self.model = load_sentence_transformers_model(self.model_name)
        elif self.model_name in supported_embedding_models:
            self.tokenizer, self.model = load_bert_based_model(self.model_name)
        print(f"\t\tEmbedding model {self.model_name} loaded")
    

    def embed(self, input_texts):
        """
        Obtain the embeddings of the input texts from the chosen embedding model
        
        Args:
            input_texts (list): list of string utterances to embed
            
        Returns:
            np.array: numpy array of the embeddings
        """
        # Load in and return the utterance embeddings if they exist on file
        if os.path.exists(self.embedding_filepath):
            print(f"\t\tLoading in embeddings from {self.embedding_filepath}...")
            return load(self.embedding_filepath)
        
        # Load the specified embedding model
        if self.model == None:
            self.__load_model()
        # Obtain the embeddings
        if self.model_name == "use":
            embeddings = self.model(input_texts)
            embeddings = embeddings.numpy()
        elif self.model_name in bert_based_models:
            embeddings = []
            for batch in tqdm(split_list(input_texts, 32), desc="\t\tEmbedding progress"):
                encoded_input = self.tokenizer(batch, return_tensors="pt", padding=True, truncation=True)
                if gpu:
                    encoded_input = encoded_input.to("cuda:0")
                with torch.no_grad():
                    embeddings.extend(self.model(**encoded_input).pooler_output)
                if gpu:
                    del encoded_input
                    torch.cuda.empty_cache()
            embeddings = np.array([e.cpu().numpy() for e in embeddings])
        elif self.model_name in sentence_transformer_models:
            embeddings = self.model.encode(input_texts)
        # If the input is utterances, save the embeddings
        print("\t\tSaving embeddings...")
        save(self.embedding_filepath, embeddings)
        return embeddings
        


def load_bert_based_model(name="bert-base-uncased"):
    tokenizer = AutoTokenizer.from_pretrained(name)
    model = AutoModel.from_pretrained(name)

    if gpu:
        model = model.to("cuda:0")
    return tokenizer, model


def load_sentence_transformers_model(name="all-mpnet-base-v2"):
    return SentenceTransformer("sentence-transformers/"+name)


def load_universal_sentence_encoder():
    return hub.load(embedding_model_urls["use"])


def split_list(lst, chunk_size):
    for i in range(0, len(lst), chunk_size):
        yield lst[i:i+chunk_size]