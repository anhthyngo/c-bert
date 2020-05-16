"""
Module to define analysis methods.
"""

import torch
import matplotlib
import matplotlib.pyplot as plt
import numpy as np

colors = {
        "BERT TriviaQA-web"      : "mediumseagreen",
        "BERT SQuAD"             : "orange",
        "Meta-BERT TriviaQA-web" : "tab:blue",
        "Meta-BERT SQuAD"        : "tab:pink"
        }

markers = {
        "BERT TriviaQA-web"       : "o",
        "BERT SQuAD"              : "^",
        "Meta-BERT TriviaQA-web"  : "P",
        "Meta-BERT SQuAD"         : "X"
        }

names = {
    "BERT TriviaQA-web"       : "BERT Trivia",
    "BERT SQuAD"              : "BERT SQuAD",
    "Meta-BERT TriviaQA-web"  : "Meta-BERT Trivia",
    "Meta-BERT SQuAD"         : "Meta-BERT SQuAD"
    }

def plot_learning(
        data,                            # dictionary of dictionaries of data
        title = None,                    # title for plot
        iter_key = "iter",               # iteration key word
        val_key = "f1",                  # validation key word
        score_type = "F1",               # score type for label name
        linewidth = 1.0,                 # linewidth for plot
        offset = 10,                     # pixel offset for axes
        label_x_offset = 200,              # x direction offset for label
        label_y_offset = 1,              # y direction offset for label
        x_size = 10,                     # x size of figure
        y_size = 5,                      # y size of figure
        x_label = "training iteration",  # name for x-axis
        x_tick_int = 20000,              # interval for ticks on x-axis
        iterations = 100000,             # max number of training iterations
        y_label = "score",               # name for y-axis
        y_tick_int = 10,                 # interval for ticks on y-axis
        max_score = 100                  # max score
        ):
    """
    Generate plot like Yogatama for continual learning
    """
    
    # Create plot figure
    fig = plt.figure(figsize = (x_size, y_size))
    ax = fig.add_subplot(111)
    
    # Plot Style
# =============================================================================
#     plt.style.use('seaborn-deep')
# =============================================================================
    
    # Format x and y axes
    plt.xlabel(x_label)
    plt.xlim(0,iterations)
    plt.xticks(np.arange(0,iterations+1,x_tick_int))
    
    plt.ylabel(y_label)
    plt.ylim(0,max_score)
    plt.yticks(np.arange(0,max_score+1,y_tick_int))
    
    # Adjust plot borders
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.spines['left'].set_position(('outward', offset))
    ax.spines['bottom'].set_position(('outward', offset))
    
    # Iterate through keys
    for key in data.keys():
        
        assert key in colors.keys(), "{} is not a support key in: {}".format(key, colors.keys())
        
        # Get data set
        experiment = data.get(key)
        
        # Get x and y values for plotting
        temp_iter = experiment.get(iter_key)
        temp_val = experiment.get(val_key)
        
        # Plot data on figure
        ax.plot(temp_iter, temp_val, linestyle='-', marker=markers.get(key),
                color=colors.get(key), mec='w', ms=8, clip_on=False)
        
        # Add dataset labels
        plt.text(temp_iter[-1]+label_x_offset, temp_val[-1]-label_y_offset,
                 names.get(key)+" "+score_type, color=colors.get(key))
        
        # Add title
        if not title is None:
            plt.title(title)
        
        # Adjust layout
        fig.tight_layout()
    
    return fig
