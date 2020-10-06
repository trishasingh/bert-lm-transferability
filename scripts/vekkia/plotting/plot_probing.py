import glob
import matplotlib.pyplot as plt
import os
import sys

if len(sys.argv) < 2:
    print("Please provide task name")
    sys.exit()

tasks = []
for t in range(1, len(sys.argv)):
    tasks.append(sys.argv[t])

task_label = tasks[0].split('-')[0]
colors = ['slateblue', 'darkorange', 'forestgreen', 'r']
metrics = {'SST' : 'Accuracy', 'QNLI' : 'Accuracy', 'CoLA' : 'MCC'}
output_path = '/data/home/bestie/cs224n-project/output/frozen/'

task_name = tasks[0]
output_dirs = glob.glob(output_path+task_name+'/start-layer-*')
idxs = [int(l.split('-')[-1]) for l in output_dirs]
formatted = list(zip(idxs, output_dirs))

scores = []
for l, name in sorted(formatted, key=lambda x: -x[0]):
    with open(name+'/eval_results.txt', 'r') as f:
        score = float(f.readline().strip().split(' ')[-1])
        scores.append(score)
num_re_init = [max(idxs)-s for s in sorted(idxs, reverse=True)]

plt.plot(num_re_init, scores, marker='.',
             markerfacecolor='black', markersize=6,
             color=colors[0], linewidth=1)
plt.xticks(range(min(num_re_init), max(num_re_init)+1))
plt.xlabel('Number of layers removed')
plt.ylabel(metrics[task_label] + ' on test set')
plt.title('Probing BERT layers. Task: '+task_label)
plt.grid()
plt.savefig('./plots/probing_'+task_label+'.png', dpi=300)
