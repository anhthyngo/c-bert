"""
Module to define analysis methods. Right now can only think of plots.
"""

import torch
import matplotlib
import matplotlib.pyplot as plt
import numpy as np

def plot_learning(
        trivia,                          # dictionary for TriviaQA validation F1 scores
        squad,                           # dictionary for SQuAD validation F1 scores
        model_name,                      # name of model used
        linewidth = 1.0,                 # linewidth for plot
        offset = 10,                     # pixel offset for axes
        label_x_offset = 0.2,            # x direction offset for label
        label_y_offset = 0.1,            # y direction offset for label
        x_size = 8,                      # x size of figure
        y_size = 5,                      # y size of figure
        x_label = "training iteration",  # name for x-axis
        x_tick_int = 20000,              # interval for ticks on x-axis
        iterations = 100000,             # max number of training iterations
        y_label = "score",               # name for y-axis
        y_tick_int = 20,                 # interval for ticks on y-axis
        max_score = 100,                 # max score
        trivia_name = "Trivia F1",       # name for TriviaQA label
        trivia_color = "mediumseagreen", # color for TriviaQA
        trivia_marker = "o",             # marker for TriviaQA
        squad_name = "SQuAD F1",         # name for SQuAD label
        squad_color = "orange",          # color for SQuAD
        squad_marker = "^"               # marker for SQuAD
        ):
    """
    Generate plot like Yogatama for continual learning
    """
    
    # Create plot figure
    fig = plt.figure(figsize = (x_size, y_size))
    ax = fig.gca()
    
    # Get x and y values for data
    trivia_iter = trivia.get("iter")
    trivia_val = trivia.get("val")
    squad_iter = squad.get("iter")
    squad_val = squad.get("val")
    
    # Plot data on figure
    ax.plot(trivia_iter, trivia_val, linestyle='-',
            marker=trivia_marker, color=trivia_color, fillstyle='none', 
            linewidth=linewidth, markerfacecolor = 'white',
            clip_on=False)
        
    ax.plot(squad_iter, squad_val, linestyle='-',
            marker=squad_marker, color=squad_color, fillstyle='none',
            linewidth=linewidth, markerfacecolor = 'white',
            clip_on=False)
    
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
    
    # Add dataset labels
    plt.text(trivia_iter[-1]+label_x_offset, trivia_val[-1]-label_y_offset,
             model_name+" "+trivia_name, color=trivia_color)
    plt.text(squad_iter[-1]+label_x_offset, squad_val[-1]-label_y_offset,
             model_name+" "+squad_name, color=squad_color)
    
    return fig