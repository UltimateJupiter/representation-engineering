import sys
sys.path.append('/scratch/network/yc6206/representation-engineering')
from transformers import AutoTokenizer, pipeline, AutoModelForCausalLM
import torch
from datasets import load_from_disk, load_dataset
import copy

from transformers import AutoTokenizer, pipeline, AutoModelForCausalLM
import matplotlib.pyplot as plt
import torch
from tqdm import tqdm
import numpy as np
import pynvml
import pickle

from repe import repe_pipeline_registry
repe_pipeline_registry()

device = "cuda" if torch.cuda.is_available() else "cpu"
if device == "cuda":
    pynvml.nvmlInit()
    handle = pynvml.nvmlDeviceGetHandleByIndex(0)
    print(pynvml.nvmlDeviceGetName(handle))

from sklearn.decomposition import DictionaryLearning
from sklearn.decomposition import SparseCoder


with open('/scratch/network/yc6206/representation-engineering/examples/extraction/reading_vecs_sc.pickle', 'rb') as f:
    reading_vectors = pickle.load(f)

with open('/scratch/network/yc6206/representation-engineering/examples/extraction/sparse_coding.pickle', 'rb') as f:
    sparse_coding_dict = pickle.load(f)

emotions = []
with open ('list_of_emotions.txt', 'r') as f:
    emotions = f.readlines()
    emotions = [emotion.strip('\n') for emotion in emotions]

for n_components in (pbar := tqdm(range(5, 105, 5))):
    sparse_coding_dict[n_components] = {}
    for layer in reading_vectors['Happiness']:
        X = []
        sparse_coding_dict[n_components][layer] = {}
        for emotion in emotions:
            X.append(reading_vectors[emotion][layer])
        X = np.concatenate(X)
        dict_learner = DictionaryLearning(n_components=n_components, transform_algorithm='lasso_lars', transform_alpha=0.1, random_state=42)
        dict = dict_learner.fit(X)
        sparse_coding_dict[n_components][layer]['dictionary'] = dict.components_
        coder = SparseCoder(dictionary=dict.components_, transform_algorithm='lasso_lars', transform_alpha=1e-10)
        for emotion in emotions:
            sparse_coding_dict[n_components][layer][emotion] = coder.transform(X)

    with open('/scratch/network/yc6206/representation-engineering/examples/extraction/sparse_coding.pickle', 'wb') as f:
        pickle.dump(sparse_coding_dict, f)
