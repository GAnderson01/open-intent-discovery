import os

import spacy
import pandas as pd
from tqdm import tqdm

import candidate_extraction.pos_utils as pos_utils
from supported_config import extraction_prompt_template, supported_extraction_methods


class Extractor:
    def __init__(self, config):
        """
        Initialises the extractor object and sets up a filepath to save/load the extracted candidates

        Args:
            dataset (str): name of the dataset (corresponds to the name of the directory in data/raw/)
            extraction_method (str): chosen candidate intent extraction method. Options: action-object, action-object-ext, t0pp-prompting
        """

        self.extraction_method = config["extraction_method"]
        if self.extraction_method in ["action-object", "action-object-ext"]:
            self.nlp = spacy.load("en_core_web_sm")

        # Set up the filepath to save/load the extracted candidates
        candidates_dir = "../data/interim/candidates/"
        extraction_df_filename = "_".join([config["dataset"], self.extraction_method, "extraction_df.csv"])
        self.extraction_df_filepath = os.path.join(candidates_dir, extraction_df_filename)
        print("\tStage 2: Candidate Extraction...")
    

    def extract(self, df=None):
        """
        Perform candidaite intent extraction on the given data using the Extractor's extraction method.
        Loads from file if it already exists.

        Args:
            df (pd.DataFrame, optional): dataframe containing the texts. Defaults to None if user wishes to load from file.

        Returns:
            pd.DataFrame: original dataframe with added candidate_intent column
        """   
        if self.extraction_method not in supported_extraction_methods:
            print(f"\t\tExtraction method: {self.extraction_method} is not supported")
            return None
        # Load in and return the candidates if they exist on file
        if os.path.exists(self.extraction_df_filepath):
            print(f"\t\tLoading in {self.extraction_df_filepath}")
            return pd.read_csv(self.extraction_df_filepath, index_col=0)
        
        # Perform extraction using specified method
        method = self.extraction_method
        if method == "action-object":
            df = self.get_action_object_pairs(df)
        elif method == "action-object-ext":
            df = self.get_action_object_ext_pairs(df)
        elif method == "t0pp-prompting":
            df = self.t0pp_prompting_get_candidates(df)
        
        # Save candidates to file
        df.to_csv(self.extraction_df_filepath)
        return df
    

    def get_action_object_pairs(self, df):
        """Liu et al. 2021 implentation of Action-Object pairs

        Args:
            df (pd.DataFrame): dataframe containing the texts 

        Returns:
            pd.DataFrame: original dataframe with added candidate_intent column 
        """
        action_words = []
        object_words = []
        action_object_pairs = []
        for item in tqdm(df["text"], desc="\t\tCandidate extraction progress"):
            deps = spacy.displacy.parse_deps(self.nlp(item))
            action, object = "NONE", "NONE" # default
            for arc in deps["arcs"]:
                if arc["label"] == "dobj":
                    start = deps["words"][arc["start"]]
                    if start["tag"] == "VERB":
                        action = start["text"].lower()

                    end = deps["words"][arc["end"]]
                    if end["tag"] == "NOUN":
                        object = end["text"].lower()

                    # print(action, object)
                    continue

            action_words.append(action)
            object_words.append(object)
            action_object_pairs.append("{}-{}".format(action, object))

        df["action"] = action_words
        df["object"] = object_words
        df["candidate_intent"] = action_object_pairs
        return df
    

    def get_action_object_ext_pairs(self, df):
        """Action-Object-Extension method.
        Extracts action object candidiates, also considering negations, compound nouns and amods

        Args:
            df (pd.DataFrame): dataframe containing the texts 

        Returns:
            pd.DataFrame: original dataframe with added candidate_intent column 
        """  
        docs = apply_spacy_pipeline(df["text"].tolist(), self.nlp)
        candidate_intents = []
        for item in tqdm(docs, desc="\t\tCandidate extraction progress"):
            pairs = extract_action_object_pairs(item)
            if len(pairs) == 0:
                candidate_intents.append("NONE")
            else:
                candidate_intents.append(pairs[0])
        df["candidate_intent"] = candidate_intents
        return df
    

    def t0pp_prompting_get_candidates(self, df):
        """T0pp prompting method
        Generates candidates by prompting the T0pp model

        Args:
            df (pd.DataFrame): dataframe containing the texts 

        Returns:
            pd.DataFrame: original dataframe with added candidate_intent column 
        """   
        import utils.t0pp as T0ppModel
        prompt_template = extraction_prompt_template
        candidates = []
        utterances = df['text'].tolist()
        for i in tqdm(range(len(utterances)), desc='\t\tCandidate extraction progress'):
            item = utterances[i]
            prompt = prompt_template % item
            candidates.append(T0ppModel.inference(prompt))
        df['candidate_intent'] = candidates
        return df
    

def apply_spacy_pipeline(utterances, nlp_model):
    return [nlp_model(u) for u in utterances]


def get_object(obj_token_idx, doc):
    obj_token = doc[obj_token_idx]
    obj = obj_token.text
    # Look for compounds and amods
    for i in reversed(range(obj_token_idx)):
        prev_token = doc[i]
        if pos_utils.left_is_dep_of_right(prev_token, obj_token, "compound") or pos_utils.left_is_dep_of_right(prev_token, obj_token, "amod"):
            obj = prev_token.text+"_"+obj
    return obj


def get_action(obj_token_idx, doc):
    action_token = doc[obj_token_idx].head
    action = action_token.text
    # Look for negatives
    for neg_tok in doc:
        if pos_utils.is_dep(neg_tok, "neg"):
            if (pos_utils.right_is_head_of_left(neg_tok, action_token) or 
                pos_utils.left_is_dep_of_right(action_token, neg_tok.head, "xcomp")):
                action = neg_tok.text+"_"+action
    return action


def get_action_object_pair(obj_token_idx, doc):
    return get_action(obj_token_idx, doc)+"-"+get_object(obj_token_idx, doc)


def extract_action_object_pairs(doc):
    return [get_action_object_pair(i, doc) for i, token in enumerate(doc) if pos_utils.is_dep(token, "dobj")]