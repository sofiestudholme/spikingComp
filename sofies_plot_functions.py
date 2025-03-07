""" 

Helper functions for plotting with matplotlib.pyplot
Sets defalt settings for nice figs with seaborn

"""

##################################################
########## imports ###############################
##################################################
import math
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import matplotlib.gridspec as gridspec
try:
    import seaborn as sns
    pa =sns.color_palette() + sns.hls_palette(6,0.7,s=0.6)
    sns.set_palette(pa, n_colors=None, desat=None, color_codes=False)

    _new_black =  '#373737'      #'#373737'
    size=7
    sns.set_theme(style='ticks', font_scale=1, rc={
        # 'font.family': 'sans-serif',
        # 'font.sans-serif': ['Arial', 'DejaVu Sans'],
        # 'svg.fonttype': 'none',
        'text.usetex': False,
        'pdf.fonttype': 42,
        'ps.fonttype': 42,
        'font.size': size,
        'axes.labelsize': size,
        'axes.titlesize': 9,
        'axes.labelpad': 2,
        'axes.linewidth': 0.5,
        'axes.titlepad': 4,
        'lines.linewidth': 0.75,
        'legend.fontsize': size,
        'legend.title_fontsize': 7,
        'xtick.labelsize': size,
        'ytick.labelsize': size,
        'xtick.major.size': 2,
        'xtick.major.pad': 1,
        'xtick.major.width': 0.5,
        'ytick.major.size': 2,
        'ytick.major.pad': 1,
        'ytick.major.width': 0.5,
        'xtick.minor.size': 2,
        'xtick.minor.pad': 1,
        'xtick.minor.width': 0.5,
        'ytick.minor.size': 2,
        'ytick.minor.pad': 1,
        'ytick.minor.width': 0.5,

        # Avoid black unless necessary
        'text.color': _new_black,
        'patch.edgecolor': _new_black,
        'patch.force_edgecolor': False, # Seaborn turns on edgecolors for histograms by default and I don't like it
        'hatch.color': _new_black,
        'axes.edgecolor': _new_black,
        'axes.labelcolor': _new_black,
        'xtick.color': _new_black,
        'ytick.color': _new_black
    })
    plt.rcParams["xtick.direction"] = "in"
    plt.rcParams["ytick.direction"] = "in"

    COLOURS  = pa =sns.color_palette() + sns.hls_palette(6,0.7,s=0.6)
    COLOURS2 = sns.hls_palette(8,0.7,s=0.6)

except:
    print("Can't find seaborn :(\n default fig params not updated")

try:
    import pandas as pd  
    import csv
    from IPython.display import set_matplotlib_formats
    from collections import OrderedDict
    import matplotlib.gridspec as gridspec
except:
    print("Missing some imports :(")



#############################################################
######### plt settings ######################################
#############################################################

set_matplotlib_formats('svg','pdf')
plt.rcParams.update({
    "figure.facecolor":  (1.0, 1.0, 1.0, 1.0),  # red   with alpha = 30%
    "axes.facecolor":    (1.0, 1.0, 1.0, 1.0),  # green with alpha = 50%
    "savefig.facecolor": (1.0, 1.0, 1.0, 1.0),  # blue  with alpha = 20%
})


# Figures / plt stuff 
SINGLE_COLUMN_WIDTH=3.3
FULL_TWO_COLUMN_WIDTH=7
ONE_AND_HALD_COLUMN_WIDTH=5.3

BLACK = '#373737'

FIG_WIDTH=8.5
FIG_HEIGHT=4
PLOT = 0
TRANSPARENCY = 0.7
CMAP = plt.cm.get_cmap('plasma')
CMAP_WINTER = plt.cm.get_cmap('winter')


LINESTYLES = OrderedDict(
    [('solid',               (0, ())),
     ('loosely dotted',        (0, (1, 10))),
     ('dotted',                (0, (1, 1))),
     ('densely dotted',        (0, (1, 1))),
     ('long dash with offset', (5, (10, 3))),
     ('loosely dashed',        (0, (5, 10))),
     ('dashed',                (0, (5, 5))),
     ('densely dashed',        (0, (5, 1))),

     ('loosely dashdotted',    (0, (3, 10, 1, 10))),
     ('dashdotted',            (0, (3, 5, 1, 5))),
     ('densely dashdotted',    (0, (3, 1, 1, 1))),

     ('dashdotdotted',         (0, (3, 5, 1, 5, 1, 5))),
     ('loosely dashdotdotted', (0, (3, 10, 1, 10, 1, 10))),
     ('densely dashdotdotted', (0, (3, 1, 1, 1, 1, 1)))]
     )


plt.rcParams['axes.formatter.use_mathtext'] = True

#################################################################################################################### Plot functions #################################################################################

def save_(name):
    """ save as pdf in file figures / """
    plt.savefig(f'figures/{name}.pdf',bbox_inches='tight')

def save_png(name):
    """ save as png in file figures / """
    plt.savefig(f'figures/{name}.png',bbox_inches='tight')

def make_panel_square(ax):
    range_x = ax.get_xlim()[1] - ax.get_xlim()[0] 
    range_y = ax.get_ylim()[1] - ax.get_ylim()[0] 
    ax.set_aspect(range_x/range_y)

def turn_off_ticks_but_not_ticklabel(ax,y=True,x=True):
    if y:
        ax.yaxis.set_ticks_position('none')
    if x:
        ax.xaxis.set_ticks_position('none') 
 
def make_outline_white(ax):
    ax.spines['bottom'].set_color('white')
    ax.spines['top'].set_color('white') 
    ax.spines['right'].set_color('white')
    ax.spines['left'].set_color('white')

def make_panel_square(ax):
    range_x = ax.get_xlim()[1] - ax.get_xlim()[0] 
    range_y = ax.get_ylim()[1] - ax.get_ylim()[0] 
    ax.set_aspect(range_x/range_y)

def set_ticks_out(ax):
    ax.tick_params(axis='both', direction='out')

def plot_with_errorbars(samples,ax,colour,label='',linestyle="solid",x=[],show_legend=True,plot_solid=True,**kwargs):
    """ 
    takes array of shape (samples, epochs)
    plots mean and standard errror of mean 
    """
    if len(x) == 0:
        x = range(len(samples))
    samples = np.array(samples)
    average=np.mean(samples,axis=0)
    std = np.std(samples,axis=0) / np.sqrt(samples.shape[0])
    if plot_solid:
        ax.plot(x,average,linestyle=linestyle,label=label,color=colour,**kwargs)
    else:
        ax.scatter(x,average,label=label,color=colour,**kwargs)
    ax.fill_between(x,np.where(average-2*std > 0, average-2*std, 0),average+2*std,alpha=0.1,color=colour,edgecolor="none")
    if show_legend:
        ax.legend()


def plot_channels(x,title='x_channels',y=np.array([-1]),spacing = 'same'):
    ''' spacing same be same or relative
    old plot channels, doesnt use subplots'''
    max_=0
    f = plt.figure()
    f.set_size_inches(FIG_WIDTH, FIG_WIDTH*0.75)
    ''' x shape is (samples,channels)'''
    if spacing == 'relative':
        for i, sig in enumerate(x.T):
            max_+=abs(min(x[:,i]))
            plt.plot(x[:,i]+max_ )
            plt.axhline(max_,color='k')
            max_+=abs(max(x[:,i]))

    elif spacing == 'same':
        space=2*np.max(abs(np.max(x,axis=0)-np.min(x,axis=0)))
        for i, sig in enumerate(x.T):
            plt.plot((x[:,i]+i*space))
            plt.axhline(space*i,color='k')
    if y[0]!=-1:
        plt.plot((y-1)*max_/x.shape[1],'k')
    plt.yticks([], [])
    plt.title(title)
    plt.show()
 

def plot_channels_2(x,title='x_channels',labels=[''],link_axes=False,link=8,x_multiplier=1,fig_ax=[],xlabel='timesteps',width=FIG_WIDTH,step=False,linewidth=0.3,c=[],limits = []):
    ''' 
    uses subplots
    x shape is (samples,channels)
    if linking axes, give the number of axes to link from 0 -> link
    labels can be custom or ['C'] - labels channels according to number or ['N'] labels neurons arrording to number
    set x_multiplier to downsample rate to get timesteps on x axis
    pass in [figure, axes_array] if want to plot onto existing figure
     '''
    if x.shape[0] < x.shape[1]:
        x=x.T
    channels = x.shape[1]
    samples = x.shape[0]

    if len(fig_ax) == 0: # not defined, create
        if link_axes:
            fig, axes = plt.subplots(channels,sharey=True)
        else:
            fig, axes = plt.subplots(channels)
    else:
        fig = fig_ax[0]
        axes = fig_ax[1]
    fig.set_size_inches(7,10)

    colours  = np.array(COLOURS)
    if len(c) != 0:
        colours = colours[c]


    for i, sig in enumerate(x.T):
        x = i+1 if i> 3 else i
        if step:
            axes[i].step([x_multiplier*idx for idx in range(len(sig))],sig,color = colours[x]) 
        else:
            axes[i].plot([x_multiplier*idx for idx in range(len(sig))],sig,color = colours[x%16],linewidth=linewidth,alpha=1) 
        axes[i].set_xticks([], [])
        if len(limits)>1:
            if len(limits[i])==1:
                axes[i].set_ylim([0,limits[i]])
            else: 
                axes[i].set_ylim(limits[i])
        axes[i].set_xticks([], [])
        axes[i].set_yticklabels('')
    return [fig, axes]



