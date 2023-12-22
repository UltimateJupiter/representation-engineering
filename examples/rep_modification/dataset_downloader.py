from transformers import AutoTokenizer, pipeline, AutoModelForCausalLM
import torch
from datasets import load_from_disk, load_dataset
import copy

ds = load_dataset('tatsu-lab/alpaca', cache_dir='../../storage/cache')
ds.save_to_disk("../../storage/cache/alpaca_filtered/")