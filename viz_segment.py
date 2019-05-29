import os
import numpy as np
import pandas as pd

import seaborn as sns
# sns.set(style='white', context='talk')

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import matplotlib.patches as patches
from matplotlib.ticker import FormatStrFormatter

from my_dist import trip_length
import my_ftn


# plt.rcParams.update({'font.size': 8})
# from matplotlib.collections import LineCollection
# from matplotlib.colors import ListedColormap, BoundaryNorm
# matplotlib.use('TkAgg')

# https://matplotlib.org/users/dflt_style_changes.html
# https://stackoverflow.com/questions/22408237/named-colors-in-matplotlib
palette = ['C0', 'C1', 'C2', 'C3', 'C4', 
           'C5', 'C6', 'C7', 'C8', 'C9', 
           'b', 'r', 'c', 'm', 'y', 
           'g', 'crimson', 'steelblue', 'yellowgreen', 'mediumpurple']

__all__ = ['draw_path', 'plot_clustering_result', 'plot_path_by_segment']


def draw_path(df, vin, ign_on_time, 
              cluster_assignment=None, 
              save_to=None, show=True,
              verbose=False):
    path = my_ftn.filter_to_one_path(df, vin, ign_on_time)
    if verbose:
        print('len of lon, lat: ', len(path[0]), len(path[1]))
    title = my_ftn.get_str_of_trip(vin, ign_on_time)
    plot_path_by_segment(*path, cluster_assignment=cluster_assignment, 
                         title=title, save_to=save_to, figsize=(6, 6), show=show)


def plot_clustering_result(df, 
                           figsize=(9, 12), 
                           title=None,
                           save_to=None,
                           show=True):

    n_data, n_col = df.shape
    colnames = df.columns
    cluster_assignment = df.index
    cluster_ids = np.unique(cluster_assignment)
    n_cluster = cluster_ids.shape[0]
    # print('#time={}, #col={}, #cluster={}'
    #       .format(n_data, n_col, n_cluster))
    # print(cluster_ids)

    assert n_data == cluster_assignment.shape[0]
    assert len(colnames) == n_col

    # define subplot
    f, ax = plt.subplots(n_col + 1, 1, 
                         sharex=True, figsize=figsize)
    
    # x-axis (time)
    x = np.arange(n_data)
    # color by cluster for each timepoints
    colors = [palette[int(c)] for c in cluster_assignment]

    # set title
    if title is not None:
        f.suptitle(title)

    # calculate cluster intervals
    cluster_chunk_labels = []
    cluster_chunk_lengths = []

    c0 = cluster_assignment[0]
    length = 1
    for c in cluster_assignment[1:]:
        if c0 == c:
            length += 1
        else:
            cluster_chunk_labels.append(c0)
            cluster_chunk_lengths.append(length)
            c0 = c
            length = 1
    cluster_chunk_labels.append(c0)
    cluster_chunk_lengths.append(length)

    # draw bar plot according to the cluster interval info
    for i, (cid, length) in enumerate(
            zip(cluster_chunk_labels, cluster_chunk_lengths)):
        if i == 0:
            x0, y0 = 0, 0
        else:
            x0, y0 = sum(cluster_chunk_lengths[:i]), 0
        xlen, ylen = length, 1
        cluster_color = palette[int(cid)]
        rect = patches.Rectangle((x0, y0), xlen, ylen, 
                                 facecolor=cluster_color)
        ax[0].add_patch(rect)
        ax[0].text(x0 + xlen*2/5, 
                   y0 + ylen*2/5, 
                   str(int(cid)), 
                   color='w',
                   fontsize=20)
    plt.setp(ax[0].get_yticklabels(), visible=False)

    # set xlim, ylim
    ax[0].set_xlim(0, len(x))
    ax[0].set_ylim(0, 1)

    # add legend to the bar plot
    legend_lines = [
        Line2D([0], [0], color=palette[int(cid)], lw=4) 
        for cid in cluster_ids]
    legend_labels = [str(cid) for cid in cluster_ids]
    ax[0].legend(legend_lines, legend_labels, 
        loc="upper center", ncol=n_cluster, shadow=True)

    for i in range(n_col):
        y = df.iloc[:, i]
        xy = np.array([x, y]).T.reshape(-1, 2)

        for start, stop, color0, color in zip(xy[:-1], 
                                              xy[1:], 
                                              colors[:-1], 
                                              colors[1:]):
            # if color0 == color:
            #     seg_x, seg_y = zip(start, stop)
            #     ax[i+1].plot(seg_x, seg_y, color=color)
        
            seg_x, seg_y = zip(start, stop)
            ax[i+1].plot(seg_x, seg_y, color=color)
        if colnames is not None:
            ax[i+1].set_ylabel(colnames[i])
    
    plt.tight_layout()
    f.subplots_adjust(top=0.97)
    if save_to is not None:
        plt.savefig(save_to)
    if show:
        plt.show()
    plt.close('all')



def plot_path_by_segment(longitude, latitude, 
                         cluster_assignment=None,
                         figsize=(10, 10), 
                         title=None,
                         save_to=None,
                         show=True):
    """
    plot xy path by segment onto the plane
    args:
        - longitute, latitude: x, y
        - cluster_assignment: segment info for all (x, y) pairs
                              if None, assume one cluster
    """

    if cluster_assignment is not None:
        # print(len(longitude), len(latitude), len(cluster_assignment))
        assert len(longitude) == len(cluster_assignment)
        assert len(latitude) == len(cluster_assignment)
    else:
        cluster_assignment = [0] * len(longitude)

    # define subplot
    f, ax = plt.subplots(figsize=figsize)

    # xy pairs and its colors
    xy = np.array([longitude, latitude]).T.reshape(-1, 2)
    colors = [palette[int(c)] for c in cluster_assignment]

    for start, stop, color in zip(xy[:-1], 
                                          xy[1:], 
                                          colors[1:]):    
        seg_x, seg_y = zip(start, stop)
        ax.plot(seg_x, seg_y, color=color, marker='o', markersize=3)
    
    ax.annotate('Start', xy=(longitude[0], latitude[0]), fontsize=12)
    ax.annotate('End', xy=(longitude[-1], latitude[-1]), fontsize=12)

    len_trip = trip_length(longitude, latitude)
    ax.annotate('Trip summary:\n{:,.0f}min, {:.2f}km'
                .format(len(longitude)/60, len_trip), 
                xy=(longitude.mean(), latitude.mean()))

    ax.xaxis.set_major_formatter(FormatStrFormatter('%.3f'))
    ax.yaxis.set_major_formatter(FormatStrFormatter('%.3f'))

    # add legend to the bar plot
    cluster_ids = np.unique(cluster_assignment)
    n_cluster = cluster_ids.shape[0]
    if n_cluster > 1:
        legend_lines = [
            Line2D([0], [0], color=palette[int(cid)], lw=4) 
            for cid in cluster_ids]
        legend_labels = [str(cid) for cid in cluster_ids]
        ax.legend(legend_lines, legend_labels, 
            loc="upper center", ncol=1, shadow=True)
    
    # set title
    if title is not None:
        f.suptitle(title)

    plt.tight_layout()
    f.subplots_adjust(top=0.95)
    if save_to is not None:
        plt.savefig(save_to)
    if show:
        plt.show()
    plt.close('all')