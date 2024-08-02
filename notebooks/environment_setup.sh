#!/bin/bash
conda install -y -c rapidsai -c conda-forge -c nvidia cuml=23.04 python=3.8 cudatoolkit=11.4
pip install -U accelerate ipywidgets nltk pydantic sentencepiece sentence-transformers spacy tensorflow tensorflow-hub tqdm transformers==4.36.2
python -m spacy download en_core_web_sm
pip install scikit-learn --upgrade