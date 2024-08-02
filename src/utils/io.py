import json
import os
import pickle


def generate_config_str(config, stage=None):
    config_str = config['dataset']
    for c in ['embedding_model_name', 'clustering_algorithm', 'clustering_measure', 'extraction_method', 'generation_method']:
        if c in config:
            config_str += '_'+config[c]
        if c == stage:
            break
    return config_str


def save(filename, obj):
    with open(filename, 'wb') as savefile:
        pickle.dump(obj, savefile)


def load(filename):
    with open(filename, 'rb') as loadfile:
        obj = pickle.load(loadfile)
    return obj


def create_dir_for_filepath_if_not_exists(filepath):
    directory = os.path.dirname(filepath)
    if not os.path.exists(directory):
        os.makedirs(directory)


def save_json(filepath, obj):
    create_dir_for_filepath_if_not_exists(filepath)
    with open(filepath, 'w') as f:
        json.dump(obj, f)


def save_dataframe(filepath, df):
    create_dir_for_filepath_if_not_exists(filepath)
    df.to_csv(filepath)