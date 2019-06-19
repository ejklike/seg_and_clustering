import os
import numpy as np

import seaborn as sns
# sns.set(style='white', context='talk')

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import matplotlib.patches as patches
from matplotlib.ticker import FormatStrFormatter

import utils
from dataio import COLUMNS_FOR_MODELING_SHORT


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
hexadecimal_color_list = [
    "000099","ff00ff","00ff00","663300","996633",
    "66ffff","3333cc","660066","66ccff","cc0000",
    "0000ff","003300","33ff00","00ffcc","ffff00",
    "ff9900","ff00ff","cccc66","666666","ffccff",
    "660000","00ff00","ffffff","3399ff","006666",
    "330000","ff0000","cc99ff","b0800f","3bd9eb",
    "ef3e1b"]

__all__ = ['plot_clustering_result', 'plot_path_by_segment', 'draw_bic_plot']


def _make_patch_spines_invisible(ax):
    ax.set_frame_on(True)
    ax.patch.set_visible(False)
    for sp in ax.spines.values():
        sp.set_visible(False)


def draw_bic_plot(trn_score_dict, tst_score_dict, args):

    trn_bic_list = [sdict['bic'] for sdict in trn_score_dict]
    tst_bic_list = [sdict['bic'] for sdict in tst_score_dict]

    trn_dbic_list = [trn_bic_list[i+1] - bic
                     if trn_bic_list[i+1] is not None and bic is not None
                     else None
                     for i, bic in enumerate(trn_bic_list[:-1])]
    tst_dbic_list = [tst_bic_list[i+1] - bic
                     if tst_bic_list[i+1] is not None and bic is not None
                     else None
                     for i, bic in enumerate(tst_bic_list[:-1])]

    trn_ll_list = [-sdict['nll'] if sdict['nll'] is not None else sdict['nll'] 
                   for sdict in trn_score_dict]
    tst_ll_list = [-sdict['nll'] if sdict['nll'] is not None else sdict['nll']
                   for sdict in tst_score_dict]

    np_list = [sdict['n_params'] for sdict in trn_score_dict]

    # visualization bic chart
    f, (ax1, ax2) = plt.subplots(nrows=2, sharex=True, figsize=(8, 8))
    f.subplots_adjust(right=0.75)

    par1 = ax1.twinx()
    par2 = ax2.twinx()

    ax2.set_xlabel('n_cluster')
    x = np.arange(args.min_nc, args.max_nc+1)

    mkw = dict(markersize=3, marker='o')
    
    ax1.set_ylabel("number of nonzero params")
    p11, = ax1.plot(x, np_list, color='C2', label="number of nonzero params", **mkw)

    par1.set_ylabel('Log-Likelihood')
    p13, = par1.plot(x, trn_ll_list, color='C0', label="Log-Likelihood (trn)", **mkw)
    p14, = par1.plot(x, tst_ll_list, color='C0', linestyle='dashed', label="Log-Likelihood (tst)", **mkw)
    
    ax2.set_ylabel('BIC')
    p21, = ax2.plot(x, trn_bic_list, color='C3', label="BIC (trn)", **mkw)
    p22, = ax2.plot(x, tst_bic_list, color='C3', linestyle='dashed', label="BIC (tst)", **mkw)
    
    par2.set_ylabel('Gradient of BIC')
    p23, = par2.plot(x[:-1], trn_dbic_list, color='C1', label="Grad. BIC (trn)", **mkw)
    p24, = par2.plot(x[:-1], tst_dbic_list, color='C1', linestyle='dashed', label="Grad. BIC (tst)", **mkw)

    ax1.yaxis.label.set_color(p11.get_color())
    par1.yaxis.label.set_color(p13.get_color())
    ax2.yaxis.label.set_color(p21.get_color())
    par2.yaxis.label.set_color(p23.get_color())

    tkw = dict(size=4, width=1.5)
    ax1.tick_params(axis='x', **tkw)
    ax2.tick_params(axis='x', **tkw)
    ax1.tick_params(axis='y', colors=p11.get_color(), **tkw)
    ax2.tick_params(axis='y', colors=p21.get_color(), **tkw)
    par1.tick_params(axis='y', colors=p13.get_color(), **tkw)
    par2.tick_params(axis='y', colors=p23.get_color(), **tkw)

    lines1 = [p11, p13, p14]
    lines2 = [p21, p22, p23, p24]
    ax1.legend(lines1, [l.get_label() for l in lines1])
    ax2.legend(lines2, [l.get_label() for l in lines2])
    
    f.suptitle('Clustering scores (w={}, lambda={}, beta={})'
               .format(args.ws, args.ld, args.bt))
    plt.tight_layout()
    f.subplots_adjust(top=0.95)
    fname = (args.basedir + 'score_plot(nc_from_{}_to_{}).png'
             .format(args.min_nc, args.max_nc))
    plt.savefig(fname)
    plt.close()

    print('')
    print('Please check the score plot for choosing the best number of cluster:')
    print(fname)
    print('')


def plot_clustering_result(df, 
                           cluster_assignment,
                           nll_vector,
                           figsize=(9, 12), 
                           title=None,
                           save_to=None,
                           show=False):

    n_data, n_col = df.shape

    if cluster_assignment is None:
        cluster_assignment = np.zeros(n_data)
    cluster_ids = np.unique(cluster_assignment)
    n_cluster = cluster_ids.shape[0]
    assert n_data == len(cluster_assignment)
    assert len(COLUMNS_FOR_MODELING_SHORT) == n_col
    # if this assertion fails, then add more palette colors!
    assert np.max(cluster_assignment) + 1 <= len(palette) 

    # define subplot
    f, ax = plt.subplots(n_col + 2, 1, 
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
                   fontsize=12)
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

    # plot nll
    xy = np.array([(x,y) for x, y in zip(x, nll_vector)]).reshape(-1, 2)
    for start, stop, _, color in zip(xy[:-1], 
                                     xy[1:], 
                                     colors[:-1], 
                                     colors[1:]):    
        seg_x, seg_y = zip(start, stop)
        ax[1].plot(seg_x, seg_y, color=color)
        ax[1].set_ylabel('NLL')
    ax[1].text(n_data/20, 
               (np.max(nll_vector) + np.mean(nll_vector))/2, 
               'avg(NLL)={:.1f}'.format(np.mean(nll_vector)), 
               fontsize=12)
    # add legend to the bar plot
    avg_nll_dict = {
        cid: np.mean([nll for c, nll in zip(cluster_assignment, nll_vector) if c==cid]) 
        for cid in cluster_ids}
    legend_labels = ['{}: avg={:.1f}'.format(cid, avg_nll) 
                     for cid, avg_nll in avg_nll_dict.items()]
    ax[1].legend(legend_lines, legend_labels, 
        loc="upper center", ncol=n_cluster, shadow=True)

    # plot sensor signals
    for i in range(n_col):
        y = df.iloc[:, i]
        xy = np.array([x, y]).T.reshape(-1, 2)
        for start, stop, _, color in zip(xy[:-1], 
                                         xy[1:], 
                                         colors[:-1], 
                                         colors[1:]):
            seg_x, seg_y = zip(start, stop)
            ax[i+2].plot(seg_x, seg_y, color=color)
        ax[i+2].set_ylabel(COLUMNS_FOR_MODELING_SHORT[i])
    
    plt.tight_layout()
    f.subplots_adjust(top=0.97)
    if save_to is not None:
        plt.savefig(save_to)
    if show:
        plt.show()
    plt.close()


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

    # filter points with zero lng or lat
    zero_idx = [i for i in range(len(longitude))
                if longitude[i] == 0 and latitude[i] == 0]
    longitude = np.delete(longitude, zero_idx)
    latitude = np.delete(latitude, zero_idx)
    cluster_assignment = np.delete(cluster_assignment, zero_idx)

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

    len_trip = utils.trip_length(longitude, latitude)
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
        ax.legend(legend_lines, legend_labels, ncol=1, shadow=True) #loc="upper center", 
    
    # set title
    if title is not None:
        f.suptitle(title)

    plt.tight_layout()
    f.subplots_adjust(top=0.95)
    if save_to is not None:
        plt.savefig(save_to)
    if show:
        plt.show()
    plt.close()