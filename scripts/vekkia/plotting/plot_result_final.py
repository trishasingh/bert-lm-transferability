import argparse
import glob
import os
import re

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

colors = ['slateblue', 'darkorange', 'forestgreen', 'r']
colors_2 = ['slateblue', 'forestgreen']
metrics = {'SST-2': 'Accuracy', 'QNLI': 'Accuracy', 'CoLA': 'MCC'}
x_labels = {'zapping': 'Number of preserved layers',
            'probing': 'Number of frozen layers'}
n_layers = {'start-layer': 13, 'only-layer': 12, 'block-start': 10}
plot_titles = {'start-layer': 'Progressive reinitialization', 'block-start': 'Block reinitialization',
               'only-layer': 'Individual layer reinitialization'}
legend_labels = {'SST-2' : ['50k', '5k', '500','probing'],
                 'QNLI' : ['50k', '5k', '500','probing'],
                 'CoLA' : ['50k', '5k', '500','probing']}

#legend_labels = {'SST-2' : ['67k', '50k', '5k', '500','probing'],
#                 'QNLI' : ['105k', '50k', '5k', '500','probing'],
#                 'CoLA' : ['8.5k', '50k', '5k', '500','probing']}

# Matlotlib settings
mpl.rcParams['savefig.dpi'] = 300
mpl.rcParams['legend.fontsize'] = 'small'
mpl.rcParams['axes.titlesize'] = 'large'
mpl.rcParams['errorbar.capsize'] = 2
smallest_font = 12

def get_scores(idxs, dirs):
    """returns list of scores given directories in seed folder"""
    formatted = list(zip(idxs, dirs))
    scores = []
    for l, name in sorted(formatted, key=lambda x: x[0]):
        with open(name + '/eval_results.txt', 'r') as f:
            score = float(f.readline().strip().split(' ')[-1])
            scores.append(score)
    return scores


def get_baseline(args, task_name, size):
    path = '/mnt/fs2/atamkin/zap_n_tune_outputs/'+args.method_used+'/'+task_name+'/'+size+'/*/start-layer-12/eval_results.txt'
    top = []
    for fn in glob.glob(path):
        with open(fn, 'r') as f:
            score = float(f.readline().strip().split(' ')[-1])
            top.append(score)
    path = '/mnt/fs2/atamkin/zap_n_tune_outputs/'+args.method_used+'/'+task_name+'/'+size+'/*/start-layer-0/eval_results.txt'
    bottom = []
    for fn in glob.glob(path):
        with open(fn, 'r') as f:
            score = float(f.readline().strip().split(' ')[-1])
            bottom.append(score)
    return np.mean(top), np.mean(bottom)


def get_plt_args(x_axis, results, n_trials=3, t_val=2.667, errorbar=False):
    """helper function to aggregate over trials"""
    means = [np.mean(x) for x in results]
    sds = [(t_val / np.sqrt(n_trials)) * np.std(x) for x in results]
    if errorbar:
        return dict(x=x_axis, y=means, yerr=sds)
    else:
        return x_axis, means


# 1 on Slack list - Progressive reinitialization
def plot_vertical(tasks, sizes, args):
    fig, axs = plt.subplots(ncols=len(tasks), sharey=False, figsize=(9,5.5))
    for i, task_name in enumerate(tasks):
        for s, size in enumerate(sizes):
            final_scores = []
            if size == '500':
                n_trials = 50
            else:
                n_trials = args.n_trials
            for seed in range(1, n_trials+1):
                # output dirs contains list of layers without layer norm
                if size == '5000':
                    path = args.data_dir+args.method_used+'/'+task_name+'/5000-selective-laye*/'+str(seed)+'/'+args.experiment_name+'-*'
                else:
                    path = args.data_dir+args.method_used+'/'+task_name+'/'+size+'/'+str(seed)+'/'+args.experiment_name+'-*'
                output_dirs = glob.glob(path)
                idxs = [int(l.split('-')[-1]) for l in output_dirs]  # get the start layer
                scores = get_scores(idxs, output_dirs)
                final_scores.append(scores)
            results = list(zip(*final_scores))
            if len(idxs) < 1:
                continue
            x_axis = [min(idxs) + s for s in sorted(idxs, reverse=False)]
            plt_args = get_plt_args(x_axis, results, n_trials, errorbar=True)
            if size == 'full':
                axs[i].errorbar(**plt_args, marker=',', label=legend_labels[task_name][s],
                                markerfacecolor=colors[i], markersize=4, #fillstyle='none',
                                color=colors[i], linewidth=1, linestyle='dashed')
            else:
                axs[i].errorbar(**plt_args, marker='o',  label=legend_labels[task_name][s],
                                markerfacecolor=colors[i], markersize=4, #fillstyle='none',
                                color=colors[i], linewidth=1, alpha=1/(s+1))
        axs[i].legend(loc='lower right')
        axs[i].set_title(task_name+' ('+metrics[task_name]+')')
        axs[i].set_xticks(range(min(x_axis), max(x_axis) + 1, 4))
    axs[0].set_ylabel(metrics['SST-2'], fontsize=smallest_font)
    axs[2].set_ylabel(metrics['CoLA'], fontsize=smallest_font)
    axs[2].yaxis.set_label_position("right")
    axs[1].set_xlabel(x_labels[args.method_used], fontsize=smallest_font+2)
    fig.suptitle(plot_titles[args.experiment_name], fontsize=smallest_font+4)
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    #ciao
    plt.savefig(args.output_dir+'progressive.png', dpi=300, bbox_inches='tight')


# OVERLAY PROBING
def plot_vertical_probing(tasks, sizes, args):
    sizes.append('probing')
    fig, axs = plt.subplots(ncols=len(tasks), sharey=False, figsize=(9,5.5))
    for i, task_name in enumerate(tasks):
        for s, size in enumerate(sizes):
            final_scores = []
            if size == '500':
                n_trials = 50
            elif size == 'probing':
                n_trials = 10
            else:
                n_trials = args.n_trials
            for seed in range(1, n_trials+1):
                # output dirs contains list of layers without layer norm
                if size == 'probing':
                    path = args.data_dir+args.method_used+'/'+task_name+'/'+size+'/'+str(seed)+'/end-layer-*'
                elif size == '5000':
                    path = args.data_dir+args.method_used+'/'+task_name+'/5000-selective-laye*/'+str(seed)+'/'+args.experiment_name+'-*'
                else:
                    path = args.data_dir+args.method_used+'/'+task_name+'/'+size+'/'+str(seed)+'/'+args.experiment_name+'-*'
                output_dirs = glob.glob(path)
                idxs = [int(l.split('-')[-1]) for l in output_dirs]  # get the start layer
                scores = get_scores(idxs, output_dirs)
                final_scores.append(scores)
            results = list(zip(*final_scores))
            if len(idxs) < 1:
                continue
            if size == 'probing':
                x_axis = [min(idxs) + s - 1 for s in sorted(idxs, reverse=False)]
            else:
                x_axis_ = [min(idxs) + s for s in sorted(idxs, reverse=False)]
                x_axis = [min(idxs) + s for s in sorted(idxs, reverse=False)]
            plt_args = get_plt_args(x_axis, results, n_trials, errorbar=True)
            if size == 'full':
                axs[i].errorbar(**plt_args, marker=',', label=legend_labels[task_name][s],
                                markerfacecolor=colors[i], markersize=4, #fillstyle='none',
                                color=colors[i], linewidth=1, linestyle='dashed')
            elif size == 'probing':
                eb = axs[i].errorbar(**plt_args, marker=',', label=legend_labels[task_name][s],
                                markerfacecolor=colors[i], markersize=4, #fillstyle='none',
                                     color=colors[i], linewidth=1, linestyle='dashed', alpha=1/2)
                eb[-1][0].set_linestyle('--')
            else:
                axs[i].errorbar(**plt_args, marker='o', label=legend_labels[task_name][s],
                                markerfacecolor=colors[i], markersize=4, #fillstyle='none',
                                color=colors[i], linewidth=1, alpha=1/(s+1))
        axs[i].legend(loc='upper left')
        axs[i].set_title(task_name+' ('+metrics[task_name]+')')
        axs[i].set_xticks(range(min(x_axis_), max(x_axis_) + 1, 4))
    axs[0].set_ylim(0.45,1.00)
    axs[1].set_ylim(0.45,1.00)
    axs[0].set_ylabel(metrics['SST-2'], fontsize=smallest_font)
    axs[2].set_ylabel(metrics['CoLA'], fontsize=smallest_font)
    axs[2].yaxis.set_label_position("right")
    axs[1].set_xlabel(x_labels[args.method_used], fontsize=smallest_font+2)
    fig.suptitle(plot_titles[args.experiment_name], fontsize=smallest_font+4)
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    plt.savefig(args.output_dir+'progressive_probing.png', dpi=300, bbox_inches='tight')


# VERTICAL PROBING
def plot_probing_vertical(tasks, sizes, args):
    fig, axs = plt.subplots(ncols=len(tasks), sharey=False, figsize=(9,5.5))
    for i, task_name in enumerate(tasks):
        for s, size in enumerate(sizes):
            final_scores = []
            for seed in range(1, args.n_trials+1):
                output_dirs = glob.glob(args.data_dir + args.method_used + '/' + task_name + '/' +
                                        args.subset_size + '/' + str(seed) + '/end-layer-*')
                idxs = [int(l.split('-')[-1]) for l in output_dirs]  # get the start layer
                scores = get_scores(idxs, output_dirs)
                final_scores.append(scores)
            results = list(zip(*final_scores))
            if len(idxs) < 1:
                continue
            x_axis = [min(idxs) + s - 2 for s in sorted(idxs, reverse=False)]
            plt_args = get_plt_args(x_axis, results, errorbar=True)  # n_trials=args.n_trials)
            axs[i].errorbar(**plt_args, marker='o',  # label=legend_block[e],
                            markerfacecolor=colors[i], markersize=4,  # fillstyle='none',
                            color=colors[i], linewidth=1)  # , linestyle=ls_block[e])
            axs[i].set_title(task_name)
            axs[i].set_xticks(range(min(x_axis), max(x_axis) + 1, 4))
            if task_name == 'CoLA':
                axs[i].set_ylim((-0.09, 0.5))
    axs[0].set_ylabel(metrics['SST-2'], fontsize=12)
    axs[2].set_ylabel(metrics['CoLA'], fontsize=12)
    axs[2].yaxis.set_label_position("right")
    axs[1].set_xlabel(x_labels[args.method_used], fontsize=13)
    fig.suptitle(plot_titles[args.experiment_name], fontsize=16)
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    plt.savefig(args.output_dir+'probing_vertical.png', dpi=300, bbox_inches='tight')


# 5 on Slack - Probing
def plot_probing(tasks, args):
    fig, axs = plt.subplots(nrows=3, sharex=True, figsize=(9,5.5))
    for i, task_name in enumerate(tasks):
        final_scores = []
        for seed in range(1, args.n_trials + 1):
            # output dirs contains list of layers without layer norm
            output_dirs = glob.glob(args.data_dir + args.method_used + '/' + task_name + '/' +
                                    args.subset_size + '/' + str(seed) + '/end-layer-*')
            idxs = [int(l.split('-')[-1]) for l in output_dirs]  # get the start layer
            scores = get_scores(idxs, output_dirs)
            final_scores.append(scores)
        results = list(zip(*final_scores))
        if len(idxs) < 1:
            continue
        x_axis = [min(idxs) + s - 2 for s in sorted(idxs, reverse=False)]
        plt_args = get_plt_args(x_axis, results, errorbar=True)# n_trials=args.n_trials)
        axs[i].errorbar(**plt_args, marker='o', #label=legend_block[e],
                        markerfacecolor=colors[i], markersize=4,  # fillstyle='none',
                        color=colors[i], linewidth=1) #, linestyle=ls_block[e])
        axs[i].set_title(task_name)
        axs[i].set_ylabel(metrics[task_name])
        if task_name == 'CoLA':
            axs[i].set_ylim((-0.09, 0.5))
    axs[i].set_xticks(range(min(x_axis), max(x_axis)+1))
    axs[i].set_xlabel('Layer', fontsize=13)
    plt.subplots_adjust(hspace=0.4)
    fig.suptitle('Probing', fontsize=16)
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    plt.savefig(args.output_dir+'probing.png', dpi=300, bbox_inches='tight')


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--task_name",
                        default=None,
                        type=str,
                        required=True)
    parser.add_argument("--subset_size",
                        default="full",
                        type=str)
    parser.add_argument("--method_used",
                        default="zapping",
                        type=str,
                        help="zapping, scrambling, probing")
    parser.add_argument("--experiment_name",
                        default='start-layer',
                        type=str,
                        help="name of the experiment within the size/seed folder")
    parser.add_argument("--ln",
                        default=False,
                        type=bool,
                        help="True if you want to include layer norm experiment")
    parser.add_argument("--n_trials",
                        default=3,
                        type=int,
                        help="Number of trials run")
    parser.add_argument("--data_dir",
                        default=None,
                        type=str,
                        required=True,
                        help="The directory where the results are stored")
    parser.add_argument("--output_dir",
                        default=None,
                        type=str,
                        required=True,
                        help="The output directory where the plots are stored")
    parser.add_argument("--do_scrambling",
                        action="store_true",
                        help="Plot scrambling results")
    parser.add_argument("--plot_vertical",
                         action="store_true",
                         help="Plot results vertically")
    parser.add_argument("--plot_block",
                         action="store_true",
                         help="Plot localized results")
    parser.add_argument("--plot_probing",
                         action="store_true",
                         help="Plot probing results")
    parser.add_argument("--plot_scatter",
                         action="store_true",
                         help="Plot scatter results")    
    args = parser.parse_args()

    tasks = args.task_name.split(",")
    sizes = args.subset_size.split(",")

    if args.plot_vertical:
#        plot_vertical(tasks, sizes, args)
        plot_vertical(tasks, sizes, args)
    # elif args.plot_block:
    #     experiments = args.experiment_name.split(',')
    #     plot_block(tasks, sizes, experiments, args)
    elif args.plot_probing:
        plot_vertical_probing(tasks, sizes, args)
#        plot_probing(tasks, args)
#        plot_probing_vertical(tasks, sizes, args)
    # elif args.plot_scatter:
    #     plot_scatter(tasks, sizes, args)
        
if __name__ == "__main__":
    main()
