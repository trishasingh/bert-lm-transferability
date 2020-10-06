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
  l2 = np.linalg.norm(W1-W2, ord=2)
  return l2
    

def get_similarity(models_loaded, model_name, m1, l):
  m2 = list(models_loaded[model_name].children())[l]
  out_layer_1 = m1.output.dense  # dense: name given by Transformers to linear layer
  out_layer_2 = m2.output.dense  
  similarity = compute_similarity(out_layer_1.weight.data, out_layer_2.weight.data)
  return similarity


def compute_output(models_to_load):
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
  for l, layer in enumerate(models_loaded['bert-FT'].children()):
    # Get the corresponding layer
    base_sim.append(get_similarity(models_loaded, 'bert-base', layer, l))
    # Normalized
#  base_sim = [(ele - min(base_sim)) / (max(base_sim) - min(base_sim)) for ele in base_sim]
#  base_sim = [ele / max(base_sim) for ele in base_sim]
  return base_sim


def main(argv):
  tasks = []
  for t in range(0, len(argv)):
    tasks.append(argv[t])
  task_label = tasks[0].split('-')[0]

  x_axis = range(1, 13)
  colors = ['slateblue', 'darkorange', 'forestgreen', 'r']
  for i, task_name in enumerate(tasks):
    models_to_load = {'bert-base' : 'bert-base-uncased',
              'bert-FT' : MODEL_PATH+task_name+'/start-layer-12/', # untouched FT
              'bert-ST' : MODEL_PATH+task_name+'/start-layer-0/'}  # fully re-init FT
    base_sim = compute_output(models_to_load)
    plt.plot(x_axis, base_sim, marker='.',
               markerfacecolor='black', markersize=6,
               color=colors[i], linewidth=1)

  plt.xlabel('Layer')
  plt.xticks(x_axis)
  plt.ylabel('L2 distance between weight matrices')
  plt.title('BERT base vs fine-tuned. Task: '+task_label)
  plt.grid()
  if task_label != 'CoLA':
    legend_labels = [t.split('-')[-1]+'%' for t in tasks]
    plt.legend(legend_labels)
  plt.savefig('./plots/L2_w_norm_'+task_label+'.png', dpi=300)


if __name__ == "__main__":
  main(sys.argv[1:])
