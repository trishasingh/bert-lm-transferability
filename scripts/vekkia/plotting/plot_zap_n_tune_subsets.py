import glob
import matplotlib.pyplot as plt
import os
import sys
import numpy as np

if len(sys.argv) < 2:
    print("Please provide task name")
    sys.exit()

colors = ['slateblue', 'darkorange', 'forestgreen', 'r']
metrics = {'SST-2' : 'Accuracy', 'QNLI' : 'Accuracy', 'CoLA' : 'MCC'}

data_path = '/mnt/fs2/atamkin/zap_n_tune_outputs/zapping/'
output_path = os.path.dirname(os.path.dirname(os.getcwd())) + '/plots/'
task = sys.argv[1]
experiments = ['full', '50000', '5000', '500']
experiment_labels = ['full', '50k', '5k', '500']

# helper function to aggregate over trials
def get_plt_args(num_re_init, results, label, errorbar=False):
    means = [np.mean(x) for x in results]
    sds = [np.std(x) for x in results]
    if errorbar:
        return dict(x=num_re_init, y=means, yerr=sds, label=label)
    return num_re_init, means


for i, experiment in enumerate(experiments):
    final_scores = []
    for seed in range(1, 4):
        output_dirs = glob.glob(data_path+task+'/'+experiment+'/'+str(seed)+'/start-layer-*')
        idxs = [int(l.split('-')[-1]) for l in output_dirs]
        formatted = list(zip(idxs, output_dirs))
        scores = []
        for l, name in sorted(formatted, key=lambda x: -x[0]):
            with open(name+'/eval_results.txt', 'r') as f:
                score = float(f.readline().strip().split(' ')[-1])
                scores.append(score)
                # scores.append(([score, l]))
        final_scores.append(scores)
    results = list(zip(final_scores[0], final_scores[1], final_scores[2]))
    # print(results)
    # print('-----------------')
    # PLOT RESULTS FOR EXPERIMENT
    num_re_init = [max(idxs)-s for s in sorted(idxs, reverse=True)]
    args = get_plt_args(num_re_init, results, experiment)
    # plt.errorbar
    plt.plot(args[0], args[1], marker='.',
                 markerfacecolor='black', markersize=6,
                 color=colors[i], linewidth=1)
# PLOT FORMATTING
# print(num_re_init)
plt.xticks(range(min(num_re_init), max(num_re_init)+1))
plt.xlabel('Number of layers re-initialized')
plt.ylabel(metrics[task] + ' on test set')
plt.title('Zap-n-tune BERT for task: ' + task)
plt.grid()
plt.legend(experiment_labels)
plt.savefig(output_path+'zap-n-tune-subsets_'+task+'.png', dpi=300)


"""
# d = pltline(...); plt.errorbar(**d, ...)
idxs = range(len(results))
num_re_init = [max(idxs)-s for s in sorted(idxs, reverse=True)]
# make plot
d = pltline(num_re_init, results, 'test')
plt.errorbar(**d, marker='.',
             markerfacecolor='black', markersize=6,
             color=colors[0], linewidth=1)
plt.xticks(range(min(num_re_init), max(num_re_init)+1))
plt.xlabel('Number of layers re-initialized')
plt.ylabel(metrics[task_label] + ' on test set')
plt.title('Fine-tuned BERT with re-initialized layers. Task: ' + task_label)
plt.grid()
plt.savefig('./plots/'+'test'+'re-init_'+task_label+'.png', dpi=300)
#-------------------------------------
            
num_re_init = [max(idxs)-s for s in sorted(idxs, reverse=True)]
plt.plot(num_re_init, scores, marker='.',
             markerfacecolor='black', markersize=6,
             color=colors[i], linewidth=1)
plt.xticks(range(min(num_re_init), max(num_re_init)+1))
plt.xlabel('Number of layers re-initialized')
plt.ylabel(metrics[task_label] + ' on test set')
plt.title('Fine-tuned BERT with re-initialized layers. Task: ' + task_label)
plt.grid()

if task_label != 'CoLA':
    legend_labels = [t.split('-')[-1]+'%' for t in tasks]
    plt.legend(legend_labels)

plt.savefig('./plots/'+'re-init_'+task_label+'.png', dpi=300)
"""
