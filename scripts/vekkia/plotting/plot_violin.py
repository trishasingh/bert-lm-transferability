import argparse
import glob
import os
import re

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

colors = ['slateblue', 'darkorange', 'forestgreen', 'r']
# colors_ln = ['royalblue4', 'darkorange', 'forestgreen', 'r']
metrics = {'SST-2': 'Accuracy', 'QNLI': 'Accuracy', 'CoLA': 'MCC'}
x_labels = {'zapping': 'Number of preserved layers',
            'probing': 'Number of frozen layers'}
n_layers = {'start-layer': 13, 'only-layer': 12, 'block-start': 10}
plot_titles = {'start-layer': 'Partial reinitialization', 'block-start': 'Block reinitialization',
               'only-layer': 'Individual layer reinitialization'}
legend_labels = {'SST-2' : ['67k', '50k', '5k', '500'],
                 'QNLI' : ['105k', '50k', '5k', '500'],
                 'CoLA' : ['8.5k', '5k', '500']}

# Matlotlib settings
mpl.rcParams['savefig.dpi'] = 300
mpl.rcParams['legend.fontsize'] = 'small'
mpl.rcParams['axes.titlesize'] = 'large'
mpl.rcParams['errorbar.capsize'] = 2

# DEBUGGING
def count_dirs(path, ln=False, experiment_name=''):
    return len(list(filter(lambda x: experiment_name in x, os.listdir(path))))

def split_reinit_ln(dirs):
    output_dirs = list(filter(lambda x: 'reinit-ln' not in x, dirs))
    ln_dirs = list(filter(lambda x: 'reinit-ln' in x, dirs))
    return output_dirs, ln_dirs

def get_scores(idxs, dirs):
    """returns list of scores given directories in seed folder"""
    formatted = list(zip(idxs, dirs))
    scores = []
    for l, name in sorted(formatted, key=lambda x: x[0]):
        with open(name + '/eval_results.txt', 'r') as f:
            score = float(f.readline().strip().split(' ')[-1])
            scores.append(score)
    return scores

def get_plt_args(x_axis, results, errorbar=False):
    """helper function to aggregate over trials"""
    means = [np.mean(x) for x in results]
    sds = [np.std(x) for x in results]
    if errorbar:
        return dict(x=x_axis, y=means, yerr=sds)
    else:
        return x_axis, means

def get_baseline(args, task_name, size):
    path = args.data_dir+'zapping'+'/'+task_name+'/'+size+'/*/start-layer-12/eval_results.txt'
    top = []
    for fn in glob.glob(path):
        with open(fn, 'r') as f:
            score = float(f.readline().strip().split(' ')[-1])
            top.append(score)
    path = args.data_dir+'zapping'+'/'+task_name+'/'+size+'/*/start-layer-0/eval_results.txt'
    bottom = []
    for fn in glob.glob(path):
        with open(fn, 'r') as f:
            score = float(f.readline().strip().split(' ')[-1])
            bottom.append(score)
    return np.mean(top), np.mean(bottom), np.std(top), np.std(bottom)

def plot_subplots(tasks, sizes, args):
    fig, axs = plt.subplots(nrows=3, sharex=True, figsize=(7.5,6.5))
    for i, task_name in enumerate(tasks):
        legend_labels = []
        for s, size in enumerate(sizes):
            # CHECKS--------------------------------------------------------------
            # Size exists for given task and method
            tmp_path = args.data_dir+args.method_used+'/'+task_name+'/'+size
            if not os.path.exists(tmp_path):
                print(tmp_path, "does not exist")   
                continue
            # All seeds present for given task, method, and size
            if count_dirs(tmp_path) != args.n_trials:
                print("Not all seeds present")
                continue
            # All layers for given seed present
            all_layers_present = True
            tmp_n_layers = n_layers[args.experiment_name]
            for seed in range(1,args.n_trials+1):
                if count_dirs(tmp_path+'/'+str(seed), args.ln, args.experiment_name) not in [tmp_n_layers, 2*tmp_n_layers]:
                    print("all layers not present for", tmp_path+'/'+str(seed))
                    all_layers_present = False
            if not all_layers_present:
                continue
            # ----------------------------------------------------------------------
            final_scores = []
            final_scores_ln = []
            for seed in range(1, args.n_trials+1):
                # output dirs contains list of layers without layer norm
                all_dirs = glob.glob(args.data_dir+args.method_used+'/'+task_name+'/'+
                                        size+'/'+str(seed)+'/'+args.experiment_name+'-*')
                output_dirs, ln_dirs = split_reinit_ln(all_dirs)
                idxs = [int(l.split('-')[-1]) for l in output_dirs]  # get the start layer
                scores = get_scores(idxs, output_dirs)
                final_scores.append(scores)
                if args.ln and ln_dirs: 
                    scores_ln = get_scores(idxs, ln_dirs)
                    final_scores_ln.append(scores_ln)
            results = list(zip(*final_scores))
            if args.ln and final_scores_ln: 
                results_ln = list(zip(*final_scores_ln))
            x_axis = [min(idxs) + s for s in sorted(idxs, reverse=False)]
            plt_args = get_plt_args(x_axis, results, errorbar=True)
            if args.ln: plt_args_ln = get_plt_args(x_axis, results_ln, errorbar=True)
            axs[i].errorbar(**plt_args, marker='o',
                            markerfacecolor=colors[i], markersize=4,
                            color=colors[i], linewidth=1, alpha=1/(s+1))
            legend_labels.append(size)
            if args.ln:
                axs[i].errorbar(**plt_args_ln, marker='^',
                                markerfacecolor=colors[i], markersize=4,
                                color=colors[i], linewidth=1, alpha=1/(s+1))
                legend_labels.append(size+'-reinit-ln')
            # plt.plot(plt_args[0], plt_args[1], marker='.',
            #              markerfacecolor='black', markersize=6,
            #              color=colors[i], linewidth=1)
        axs[i].legend(legend_labels, loc='lower right')
        axs[i].set_ylabel(metrics[task_name])
        axs[i].set_title(task_name)
    axs[i].set_xticks(range(min(x_axis), max(x_axis) + 1))
    if args.experiment_name == 'block-start':
        x_tickslabels = [str(i)+'-'+str(i+1)+'-'+str(i+2) for i in range(10)]
        axs[i].set_xticklabels(x_tickslabels, rotation=30)
    axs[i].set_xlabel(x_labels[args.method_used])
    if args.ln:
        if args.experiment_name == 'block-start':
            axs[i].set_xlabel('Re-initialized layers')
        else:
            axs[i].set_xlabel('Re-initialized layer')
    fig.suptitle(plot_titles[args.experiment_name])
    # plt.grid()
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    plt.savefig(args.output_dir+'test_plot.png', dpi=300)


def plot_vertical(tasks, sizes, args):
    fig, axs = plt.subplots(ncols=len(tasks), sharey=True, figsize=(7.5,6.5))
    for i, task_name in enumerate(tasks):
        legend_labels = []
        for s, size in enumerate(sizes):
            final_scores = []
            final_scores_ln = []
            for seed in range(1, args.n_trials+1):
                # output dirs contains list of layers without layer norm
                all_dirs = glob.glob(args.data_dir+args.method_used+'/'+task_name+'/'+
                                        size+'/'+str(seed)+'/'+args.experiment_name+'-*')
                output_dirs, ln_dirs = split_reinit_ln(all_dirs)
                idxs = [int(l.split('-')[-1]) for l in output_dirs]  # get the start layer
                scores = get_scores(idxs, output_dirs)
                final_scores.append(scores)
            results = list(zip(*final_scores))
            x_axis = [min(idxs) + s for s in sorted(idxs, reverse=False)]
            plt_args = get_plt_args(x_axis, results, errorbar=True)
            if size == 'full':
                axs[i].errorbar(**plt_args, marker='o',
                                markerfacecolor=colors[i], markersize=4,
                                color=colors[i], linewidth=1, linestyle='dashed')
            else:
                axs[i].errorbar(**plt_args, marker='o',
                                markerfacecolor=colors[i], markersize=4,
                                color=colors[i], linewidth=1, alpha=1/(s))
        axs[i].legend(legend_labels[task_name], loc='lower right')
        axs[i].set_title(task_name)
        axs[i].set_xticks(range(min(x_axis), max(x_axis) + 1, 4))
    axs[0].set_ylabel(metrics['SST-2'])
    axs[2].set_ylabel(metrics['CoLA'])
    axs[2].yaxis.set_label_position("right")
    plt.xlabel(x_labels[args.method_used])
    fig.suptitle(plot_titles[args.experiment_name])
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    plt.savefig(args.output_dir+'test_plot.png', dpi=300)


def plot_scrambling(tasks, args):
    fig, axs = plt.subplots(ncols=len(tasks), sharey=False, figsize=(8.5,5))
    task_scores = dict()
    task_baselines = dict()
    # get top and bottom scores
    for i, task_name in enumerate(tasks):
        scores = []
        for seed in range(1, 11):
            res_file = args.data_dir+args.method_used+'/'+task_name+'*/'+str(seed)+'/eval_results.txt'
            for fn in glob.glob(res_file):
                with open(fn, 'r') as f:
                    score = float(f.readline().strip().split(' ')[-1])
                    scores.append(score)
        task_scores[task_name] = scores
        # obtaining full and scratch models
        full_score, scratch_score, full_sd, scratch_sd = get_baseline(args, task_name, '5000')
        task_baselines[task_name] = [scratch_score, full_score]
        # formatting individual violin
        parts = axs[i].violinplot(scores, showextrema=False)
        for pc in parts['bodies']:
            pc.set_facecolor(colors[i])
            pc.set_edgecolor('face')
            pc.set_alpha(0.25)
        axs[i].vlines(np.arange(1,2,1), min(scores), max(scores), color=colors[i], linestyle='-', lw=1.2)
        axs[i].set_ylabel(metrics[task_name], fontsize=12)
        axs[i].set_xticks(np.arange(1,2,1))
        xticklabel = task_name + ' ('+ metrics[task_name] + ')'
        axs[i].set_xticklabels([xticklabel], fontsize=12)
        axs[i].set_xlim(0.6, 1.4)
        # adding points for permutations
        axs[i].scatter([1]*10, scores, marker='o', color=colors[i], facecolors='none', 
                       s=40, label = 'permutations')
        # add full and scratch lines
        axs[i].axhspan(full_score-2*full_sd, full_score+2*full_sd, 
                       color='grey', label='_nolegend_', alpha=.25)
        axs[i].axhline(y=full_score, color=colors[i], label='full FT', ls='dashdot')
        axs[i].axhspan(scratch_score-2*scratch_sd, scratch_score+2*scratch_sd, 
                       color='grey', label='_nolegend_', alpha=.25)
        axs[i].axhline(y=scratch_score, color=colors[i], label='scratch FT', ls='dashed')
        # adding mean performance
        axs[i].axhline(y=np.mean(scores), xmin=.25, xmax=.75, color=colors[i], label='mean performance', ls='solid')
        # adding legend
        axs[i].legend(loc = 'upper right')
    # getting y axis limits
    acc_min = min(task_scores[tasks[0]] + task_scores[tasks[1]] + 
                  task_baselines[tasks[0]] + task_baselines[tasks[1]])
    acc_max = max(task_scores[tasks[0]] + task_scores[tasks[1]] +
                  task_baselines[tasks[0]] + task_baselines[tasks[1]])
    mcc_min = min(task_scores[tasks[2]] + task_baselines[tasks[2]])
    mcc_max = max(task_scores[tasks[2]] + task_baselines[tasks[2]]) 
    # space above and below plot
    m = .02
    # num ticks
    n_ticks = 10
    acc_yticks = np.linspace(acc_min,acc_max,n_ticks)
    mcc_yticks = np.linspace(mcc_min,mcc_max,n_ticks)
    # formatting for SST
    axs[0].set_ylim(acc_min-m,acc_max+m+0.035)
    axs[0].set_yticks(acc_yticks)
    axs[0].set_yticklabels(['%.2f'%round(i,2) for i in acc_yticks])
    # formatting for QNLI
    axs[1].set_ylim(acc_min-m,acc_max+m)
    axs[1].set_ylabel('')
    axs[1].set_yticks([])
    # formatting for CoLA
    axs[2].set_ylim(mcc_min-.07,mcc_max+.03+0.09)
    axs[2].set_yticks(mcc_yticks)
    axs[2].set_yticklabels(['%.2f'%round(i,2) for i in mcc_yticks])
    axs[2].yaxis.set_ticks_position('right')
    axs[2].yaxis.set_label_position('right')
    # formatting global plot
    # axs[2].set_ylim(0, 0.25)
    fig.suptitle('Layer permutation', fontsize=16)
    plt.subplots_adjust(top=0.9, wspace=0.1)
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    plt.savefig(args.output_dir+'_test_plot.png', bbox_inches='tight')


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
    args = parser.parse_args()

    tasks = args.task_name.split(",")
    sizes = args.subset_size.split(",")

    if args.do_scrambling:
        plot_scrambling(tasks, args)
    elif args.plot_vertical:
        plot_vertical(tasks, sizes, args)
    else:
        plot_subplots(tasks, sizes, args)

if __name__ == "__main__":
    main()
