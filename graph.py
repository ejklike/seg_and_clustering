import os
from itertools import product

import numpy as np
import pandas as pd
import seaborn as sns

import networkx as nx
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('TkAgg')

import utils
from dataio import COLUMNS_FOR_MODELING_SHORT


def _calculate_bc_and_draw_graph(M,
                                 labels=None,
                                 weighted=True,
                                 threshold=2e-5,
                                 title=None,
                                 save_to=None):
    """
    Given argument M, a Markov Random Field (MRF),
        - Calculate betweenness centrality of all nodes
        - Plot a (weighted) graph
    """
     # the number of nodes
    n_nodes = M.shape[0]
    assert n_nodes == len(labels)

    f = plt.figure(figsize=(4, 4))

    #2. Add nodes
    G = nx.Graph() #Create a graph object called G
    node_list = [str(i) for i in range(n_nodes)]
    G.add_nodes_from(node_list)
 
    #Note: You can also try a spring_layout
    pos = nx.circular_layout(G) 
    nx.draw_networkx_nodes(G, pos, node_color='steelblue', node_size=2000)
 
    #3. If you want, add labels to the nodes
    label_dict = {n:l for n, l in zip(node_list, labels)}
    nx.draw_networkx_labels(G, pos, label_dict, font_size=12)
 
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
    plt.tight_layout()
    plt.savefig(save_to)
    plt.close()

    # betweenness_centrality
    weight = 'weight' if weighted else None
    bc_dict = nx.centrality.betweenness_centrality(G, weight=weight)
    bc_list = [bc_dict[n] for n in node_list]
    return bc_list


def _draw_cross_time_graph(M,
                           labels=None,
                           weighted=True,
                           threshold=2e-5,
                           title=None,
                           save_to=None):
    """
    Given argument M, a Markov Random Field (MRF),
        - Calculate betweenness centrality of all nodes
        - Plot a (weighted) graph
    """
     # the number of nodes
    n_nodes = M.shape[0]
    assert n_nodes * 2 == len(labels)

    f = plt.figure(figsize=(4, 4))

    #2. Add nodes
    G = nx.Graph() #Create a graph object called G
    node_list1 = labels[:n_nodes]
    node_list2 = labels[n_nodes:]
    G.add_nodes_from(node_list1 + node_list2)
 
    pos = dict()
    pos.update( (n, (1, i)) for i, n in enumerate(node_list1) ) # put nodes from X at x=1
    pos.update( (n, (2, i)) for i, n in enumerate(node_list2) ) # put nodes from Y at x=2
    nx.draw_networkx_nodes(G, pos, node_color='steelblue', node_size=1500)

    #3. If you want, add labels to the nodes
    label_dict = {n: n for n in node_list1 + node_list2}
    nx.draw_networkx_labels(G, pos, label_dict, font_size=10)
 
    #4. Add the edges
    n = M.shape[0]
    absM = np.abs(M)
    for i in range(n):
        for j in range(1, n):
            if absM[i, j] > threshold: 
                G.add_edge(node_list1[i], 
                           node_list2[j], 
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
    plt.tight_layout()
    plt.savefig(save_to)
    plt.close()

def visualize_cluster_MRFs(cluster_MRFs, directory, args):

    print('> Save sensor network result figures ... ') 

    bc_path = os.path.join( # betweenness centrality csv
        directory, 'bc_matrix(thres=' + str(args.threshold) +').csv')
    hm_path = os.path.join( # betweenness centrality heatmap
        directory, 'bc_heatmap(thres=' + str(args.threshold) +').png')

    ncol = len(COLUMNS_FOR_MODELING_SHORT)
    gp_labels = COLUMNS_FOR_MODELING_SHORT
    BC_array = np.zeros((args.nc, ncol))

    for k, MRF in cluster_MRFs.items():
        # average over diagonal blocks
        MRF_intra_time_list = []
        for i in range(args.ws):
            MRF_intra_time_list.append(
                MRF[ncol*i:ncol*(i+1), ncol*i:ncol*(i+1)])
        MRF_intra_time = np.mean(MRF_intra_time_list, axis=0)
        # save betweenness centrality scores and graphs
        gp_path = os.path.join( # networkx graph 
            directory, 'graph_k={}(thres={}).png'
            .format(k, args.threshold))
        bc_list = _calculate_bc_and_draw_graph(
            MRF_intra_time, 
            labels=gp_labels, 
            threshold=args.threshold, 
            weighted=True,
            title='Graph between Sensors (k={})'.format(k),
            save_to=gp_path)
        BC_array[k, :] = bc_list

    # SAVE betweenness centrality values
    BC_df = pd.DataFrame(BC_array, columns=gp_labels)
    BC_df.to_csv(bc_path)
    # SAVE betweenness centrality heatmap
    with sns.axes_style("white"):
        plt.subplots(figsize=(6, 6))
        sns.heatmap(BC_df, square=True, annot=BC_df)
        plt.savefig(hm_path)
        plt.close('all')

    if args.ws > 1:
        # for each cluster,
        # we will draw cross time graph with timedelta d
        for k, MRF in cluster_MRFs.items():
            for d in range(1, args.ws): # d: timedelta
                # average over diagonal blocks according to the timedelta
                MRF_cross_time_list = []
                for i in range(args.ws - d):
                    MRF_cross_time_list.append(
                        MRF[ncol*i:ncol*(i+1), ncol*(i+d):ncol*(i+d+1)])
                MRF_cross_time = np.mean(MRF_cross_time_list, axis=0)
                gp_crosstime_path = os.path.join( # networkx graph 
                    directory, 'graph_corss_timedelta={}_k={}(thres={}).png'
                    .format(d, k, args.threshold))
                gp_crosstime_labels = \
                    ['{}\n({})'.format(c, t) for t, c in 
                        product(['t', 't+{}'.format(d)], COLUMNS_FOR_MODELING_SHORT)]
                _draw_cross_time_graph(
                    MRF_cross_time, 
                    labels=gp_crosstime_labels, 
                    threshold=args.threshold, 
                    weighted=True,
                    title='Cross-time Graph (timedelta={}, k={})'.format(d, k),
                    save_to=gp_crosstime_path)


        
