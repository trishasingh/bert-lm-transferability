from transformers import BertModel, BertTokenizer

import matplotlib.pyplot as plt
import numpy as np
import sys
import torch
import torch.nn as nn


MODEL_PATH = '/data/home/bestie/cs224n-project/output/'


def print_weights(m):
  if type(m) == nn.Linear:
    print(m.weight)


def compute_similarity(W1, W2):
#  residuals = np.linalg.lstsq(W1, W2)[1]
  l2 = np.linalg.norm(W1-W2, ord=2)
#  print(np.linalg.inv(W1))
  return l2
    

def get_similarity(models_loaded, model_name, m1, l):
  m2 = list(models_loaded[model_name].children())[l]
  out_layer_1 = m1.output.dense  # dense: name given by Transformers to linear layer
  out_layer_2 = m2.output.dense  
  similarity = compute_similarity(out_layer_1.weight.data, out_layer_2.weight.data)
  return similarity
  
    
def main(argv):
  task_name = argv[0]
  models_to_load = {'bert-base' : 'bert-base-uncased',
            'bert-FT' : MODEL_PATH+task_name+'/start-layer-12/', # untouched FT
            'bert-ST' : MODEL_PATH+task_name+'/start-layer-0/'}  # fully re-init FT

  # Load models for analysis
  models_loaded = {}
  for model_name, path in models_to_load.items():
    tokenizer = BertTokenizer.from_pretrained(path)
    model = BertModel.from_pretrained(path)
    bert_encoder = list(model.children())[1]
    bert_layers = list(bert_encoder.children())[0]
    models_loaded[model_name] = bert_layers

  # Compute similarity
  base_sim = []
  ST_sim = []
  for l, layer in enumerate(models_loaded['bert-FT'].children()):
    # Get the corresponding layer
    base_sim.append(get_similarity(models_loaded, 'bert-base', layer, l))
    ST_sim.append(get_similarity(models_loaded, 'bert-ST', layer, l))
  print("FT vs base:", base_sim)
  print("FT vs ST:", ST_sim)

  x_ax = range(1,13)
  plt.plot(x_ax, base_sim, marker='.',
             markerfacecolor='black', markersize=6,
             color='darkorange', linewidth=1)
  plt.xlabel('Layer')
  plt.ylabel('L2 distance')
  plt.title('BERT base vs fine-tuned. Task: '+task_name)
  plt.savefig('./plots/FTvsBASE.png', dpi=300)
  plt.close()
  
  plt.plot(x_ax, ST_sim, marker='.',
             markerfacecolor='black', markersize=6,
             color='forestgreen', linewidth=1)
  plt.xlabel('Layer')
  plt.ylabel('L2 distance')
  plt.title('BERT supervised-tuned vs fine-tuned. Task: '+task_name)
  plt.savefig('./plots/FTvsST.png', dpi=300)


if __name__ == "__main__":
  main(sys.argv[1:])
