"""
python ticc_with_bic.py [vin] [window_size]

For each trip of the input vin, 

1) Find the TICC solution
    - Find the best cluster size based on bic
    - cluster size in [min_cluster_size, max_cluster_size]
      
2) Draw png files corresponding to the optimal solution
    - signal
    - path
    - mrfs

Example
-------
python bc.py 5NMS33AD0KH034994 1
python bc.py 5NMS33AD6KH026656 1
-------
python bc.py 5NMS53AD9KH003365 1

"""

import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import seaborn as sns

import networkx as nx
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('TkAgg')

import utils

#####################
# PARAMETER SETTING
#####################

lambda_parameter = 5e-3 # sparsity in Toplitz matrix
beta = 200 # segmentation penalty
window_size = 1

threshold = 2e-5


columns_for_modeling = [
    'ems11_vs', 'mdps_sas11_fs_sas_angle', 
    'ems11_n', 'esp12_lat_accel', 'esp12_long_accel', 
    'ems12_pv_av_can', 'ems12_brake_act']


def add_edges_from_matrix(G, M, node_list):
    """
    G: NetworkX graph
    M: True or False
    """
    edge_list = []
    n = M.shape[0]
    for i in range(n):
        for j in range(i+1, n):
            if M[i, j]: 
                edge_list.append((node_list[i], node_list[j]))

    G.add_edges_from(edge_list)
    return G


def calculate_bc_and_draw_graph(M, 
                                labels=None, 
                                threshold=1, 
                                save_to=None):
    
    # the number of nodes
    n_nodes = M.shape[0]
    assert n_nodes == len(labels)

    G = nx.Graph()
    node_list = [str(i) for i in range(n_nodes)]

    # prepare nodes
    G.add_nodes_from(node_list)
    # prepare edges
    M_TF = np.abs(M) > threshold
    G = add_edges_from_matrix(G, M_TF, node_list)

    # betweenness_centrality
    bc_dict = nx.centrality.betweenness_centrality(G)
    bc_list = [bc_dict[n] for n in node_list]

    # draw graph
    if labels is not None:
        labels = {n:l for n, l in zip(node_list, labels)}
    else:
        labels = {n:n for n in node_list}
    nx.draw(G, labels=labels)
    plt.title('MRF visualization')
    if save_to is not None:
        plt.savefig(save_to)
    plt.close()

    return bc_list


if __name__ == '__main__':

    target_vin = sys.argv[1]

    print("lam_sparse", lambda_parameter)
    print("switch_penalty", beta)
    print("num stacked", window_size)
    print('')

    vin_folder = 'output_folder/' + target_vin + '/'
    fname_prefix_string = ("ld=" + str(lambda_parameter) +
                           "bt=" + str(beta) +
                           "ws=" + str(window_size))

    solution_fname = (fname_prefix_string + '_solution.pkl')
    solution_path_list = Path(vin_folder).glob('*/' + solution_fname)

    for solution_path in solution_path_list:
        ign_on_time = (str(solution_path).replace('\\', '/')
                                         .replace(vin_folder, '')
                                         .replace(solution_fname, '')
                                         .replace('/', ''))
                                         
        print('------', ign_on_time)

        # OUR TARGET FILES
        base_folder = str(solution_path.parents[0])
        bc_path = os.path.join(base_folder, # betweenness centrality csv
                               fname_prefix_string + 
                               '_bc_matrix(thres=' + str(threshold) +').csv')
        hm_path = os.path.join(base_folder, # betweenness centrality heatmap
                               fname_prefix_string + 
                               '_bc_heatmap(thres=' + str(threshold) +').png')

        # check if it is already saved.
        does_exist = os.path.exists(bc_path) and os.path.exists(hm_path)
        if not does_exist:
            # load solution
            print('> Load solution ...')
            number_of_clusters, bic, cluster_MRFs, cluster_assignment = \
                utils.load_solution(solution_path)
            if number_of_clusters > 0:
                print('> calculate betweenness centrality')
                # calculate betweenness_centrality
                bc_list_list = []
                for k, MRF in cluster_MRFs.items():
                    save_to = None
                    bc_list = calculate_bc_and_draw_graph(MRF, 
                                                        labels=columns_for_modeling, 
                                                        threshold=threshold, 
                                                        save_to=save_to)
                    bc_list_list.append(bc_list)
                BC = pd.DataFrame(bc_list_list, columns=columns_for_modeling)

                # SAVE betweenness centrality results
                BC.to_csv(bc_path)
                with sns.axes_style("white"):
                    f, ax = plt.subplots(figsize=(15, 15))
                    sns.heatmap(BC, square=True, annot=BC)
                    plt.savefig(hm_path)
                    plt.close()
            else:
                print('> No solution. pass.')
        else:
            print('>>> Already calculated. pass.')