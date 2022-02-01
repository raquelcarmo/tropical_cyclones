import matplotlib.pyplot as plt
import numpy as np
import os
import seaborn as sns

def plot_history(history, fold_var, save_dir):
    fig, (ax1, ax2) = plt.subplots(nrows = 2, ncols = 1, figsize=(10, 10))
    ax1.plot(history.history['accuracy'], label='Training Accuracy')
    ax1.plot(history.history['val_accuracy'], label='Validation Accuracy')
    ax1.grid(True)
    ax1.legend(loc='lower right')
    ax1.set(ylabel = "Accuracy",
            title = 'Training and Validation Accuracy')

    ax2.plot(history.history["loss"], label='Training Loss')
    ax2.plot(history.history["val_loss"], label='Validation Loss')
    ax2.plot(np.argmin(history.history["val_loss"]), np.min(history.history["val_loss"]), 
            marker="x", color="r", label = "Best model")
    ax2.grid(True)
    ax2.legend(loc='upper right')
    ax2.set(xlabel = 'Epoch', 
        ylabel = 'Cross Entropy',
        title = 'Training and Validation Loss')
    plt.show()
    fig.savefig(save_dir + "model_" + str(fold_var) + ".jpg", bbox_inches='tight')


def plot_history_ft(args, dict, fold_var, save_dir):
    ie = args['initial_epochs']
    fig, (ax1, ax2) = plt.subplots(nrows = 2, ncols = 1, figsize=(10, 10))
    ax1.plot(dict['acc'], label='Training Accuracy')
    ax1.plot(dict['validation_acc'], label='Validation Accuracy')
    #plt.ylim([min(plt.ylim()),1])
    ax1.plot([ie-1, ie-1], plt.ylim(), label='Start Fine Tuning')
    ax1.grid(True)
    ax1.legend(loc='lower right')
    ax1.set(ylabel = "Accuracy",
            title = 'Training and Validation Accuracy')

    ax2.plot(dict['loss'], label='Training Loss')
    ax2.plot(dict['validation_loss'], label='Validation Loss')
    #plt.ylim([0, 1.0])
    ax2.plot([ie-1, ie-1], plt.ylim(), label='Start Fine Tuning')
    ax2.plot(np.argmin(dict['validation_loss']), np.min(dict['validation_loss']), 
            marker="x", color="r", label = "Best model")
    ax2.grid(True)
    ax2.legend(loc='upper right')
    ax2.set(xlabel = 'Epoch',
            ylabel = 'Cross Entropy',
            title = 'Training and Validation Loss')
    plt.show()
    fig.savefig(save_dir + "model_" + str(fold_var) + ".jpg", bbox_inches='tight')


def plot_model_comparisons(nb_models, metrics, dir_dict, mean_dict, inter_dict, sub_dir, title, save_plot):
    # get colors
    colors = sns.color_palette("hls", nb_models)

    fig, ax = plt.subplots(figsize=(15,8))
    index = np.arange(len(metrics))
    bar_width = 0.1
    opacity = 0.8

    for i in range(1, nb_models+1):
        plt.bar(index + (i-1)*bar_width, mean_dict['model%d' %(i)], bar_width, yerr = inter_dict['model%d' %(i)],
                capsize = 5, alpha = opacity, color = colors[i-1],
                label = dir_dict[i].split(sep='/')[-2])

    font_size = 15
    font_size_ticks = 13
    plt.ylabel('Scores', fontsize = font_size)
    plt.title(title, fontsize = font_size)
    plt.xticks(index + (nb_models/2-0.5)*bar_width, tuple([name.capitalize() for name in metrics]), fontsize = font_size)
    plt.yticks(fontsize = font_size_ticks)
    #plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05), fancybox=True, shadow=True, ncol=2, prop={'size': font_size-3})
    plt.legend(loc = 'lower left', prop={'size': font_size})
    plt.grid(True)
    #ax.set_ylim(ymin=0.75)
    plt.tight_layout()
    plt.show()

    # SAVE PLOT
    if save_plot:
        dir = sub_dir + "model_comparisons/"
        os.makedirs(dir, exist_ok=True)
        fig.savefig("{}/{}.jpg".format(dir, title.replace(" ", "_")), bbox_inches='tight')


def plot_frozen_layers(nb_models, metrics, dir_dict, mean_dict, inter_dict, sub_dir, title, save_plot):
    # get colors
    colors = sns.color_palette("hls", nb_models)
    a = np.arange(nb_models)
    x = [0, 165, 170, 172, 174]
    assert nb_models == len(x)

    fig, ax = plt.subplots(figsize=(15,8))

    for j in range(len(metrics)):
        metric_mean = []
        metric_yerr = []
        for i in range(1, nb_models+1):
            #ax.plot(a, mean_dict['model%d' %(i)], color = colors[i-1], label = dir_dict[i].split(sep='/')[-2])
            metric_mean.append(mean_dict['model%d' %(i)][j])
            metric_yerr.append(inter_dict['model%d' %(i)][j])

        ax.errorbar(a, metric_mean, yerr = metric_yerr, capsize=5, capthick=2,
                    fmt='-o', color = colors[j], label = metrics[j].capitalize())
        bellow = np.asarray(metric_mean)-np.asarray(metric_yerr)
        up = np.asarray(metric_mean)+np.asarray(metric_yerr)
        ax.fill_between(a, bellow.tolist(), up.tolist(), color=colors[j], alpha=.1)
    
    font_size = 15
    font_size_ticks = 13
    plt.xlabel('Number of frozen layers in backbone', fontsize = font_size)
    plt.ylabel('Scores', fontsize = font_size)
    plt.title(title, fontsize = font_size)
    ax.xaxis.set_ticks(a) #set the ticks to be a
    ax.xaxis.set_ticklabels(x, fontsize = font_size_ticks) # change the ticks' names to x
    #ax.set_ylim(ymin=0.25)
    #plt.xticks(x, fontsize = font_size_ticks)
    start, end = ax.get_ylim()
    ax.yaxis.set_ticks(np.arange(np.round(start, 1), np.round(end, 1), 0.05))
    plt.yticks(fontsize = font_size_ticks)
    plt.legend(loc = 'lower left', prop={'size': font_size})
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # SAVE PLOT
    if save_plot:
        dir = sub_dir + "model_comparisons/"
        os.makedirs(dir, exist_ok=True)
        fig.savefig("{}/{}.jpg".format(dir, title.replace(" ", "_")), bbox_inches='tight')