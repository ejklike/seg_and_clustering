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
                                weighted=True,
                                threshold=2e-5,
                                title=None,
                                save_to=None):
    """
    - Calculate betweenness centrality
    - Plot a (weighted) graph
    """
     # the number of nodes
    n_nodes = M.shape[0]
    assert n_nodes == len(labels)

    #2. Add nodes
    G = nx.Graph() #Create a graph object called G
    node_list = [str(i) for i in range(n_nodes)]
    G.add_nodes_from(node_list)
 
    #Note: You can also try a spring_layout
    pos = nx.circular_layout(G) 
    nx.draw_networkx_nodes(G, pos, node_color='steelblue', node_size=700)
 
    #3. If you want, add labels to the nodes
    label_dict = {n:l for n, l in zip(node_list, labels)}
    nx.draw_networkx_labels(G, pos, label_dict, font_size=16)
 
    #4. Add the edges
    n = M.shape[0]
    absM = np.abs(M)
    for i in range(n):
        for j in range(i+1, n):
            if absM[i, j] > threshold: 
                G.add_edge(node_list[i], 
                           node_list[j], 
                           weight=absM[i, j])
    
    #4 a. Iterate through the graph nodes to gather all the weights
    #     we'll use this when determining edge thickness
    all_weights = [data['weight'] for (n1, n2, data) in G.edges(data=True)]
 
    for n1, n2, data in G.edges(data=True):
        width = data['weight'] * n_nodes / sum(all_weights)
        nx.draw_networkx_edges(G, pos, edgelist=[(n1, n2)], width=width)
    
    # #5. Add the edge labels
    # edge_labels = nx.get_edge_attributes(G, 'weight')
    # edge_labels = {n:('{:.2f}'.format(l) if l > threshold else None) for n, l in edge_labels.items()}
    # nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, alpha=0.7)

    #Plot the graph
    plt.axis('off')
    if title is not None:
        plt.title(title)
    plt.savefig(save_to)
    plt.close()

    # betweenness_centrality
    weight = 'weight' if weighted else None
    bc_dict = nx.centrality.betweenness_centrality(G, weight=weight)
    bc_list = [bc_dict[n] for n in node_list]
    return bc_list




# def calculate_bc_and_draw_graph(M, 
#                                 labels=None, 
#                                 threshold=1, 
#                                 save_to=None):
    
#     # the number of nodes
#     n_nodes = M.shape[0]
#     assert n_nodes == len(labels)

#     G = nx.Graph()
#     node_list = [str(i) for i in range(n_nodes)]

#     # prepare nodes
#     G.add_nodes_from(node_list)
#     # prepare edges
#     M_TF = np.abs(M) > threshold
#     G = add_edges_from_matrix(G, M_TF, node_list)

#     # betweenness_centrality
#     bc_dict = nx.centrality.betweenness_centrality(G)
#     bc_list = [bc_dict[n] for n in node_list]

#     # draw graph
#     if save_to is not None:
#         if labels is not None:
#             labels = {n:l for n, l in zip(node_list, labels)}
#         else:
#             labels = {n:n for n in node_list}
#         pos = nx.circular_layout(G)
#         nx.draw(G, pos, labels=labels)
#         plt.title('MRF visualization')
#         plt.savefig(save_to)
#     plt.close()

#     return bc_list