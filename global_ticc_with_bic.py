"""
python global_ticc_with_bic.py 5NMS33AD0KH034994 --verbose --min_nc 3 --max_nc 10 --maxiter 10 --ws 2
python global_ticc_with_bic.py 5NMS5CAA5KH018550 --verbose --test_size 0 --min_nc 3 --max_nc 10 --maxiter 10 --ws 2
"""

import argparse
import os
import glob
import sys
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from TICC.global_TICC_solver import GlobalTICC

import utils
from viz import (plot_path_by_segment, 
                 draw_bic_plot,
                 plot_clustering_result)
from bc import calculate_bc_and_draw_graph

#####################
# DATA SETTING
#####################

tablename = 'a302_rawsmpl_new'
columns_for_modeling = [
    'ems11_vs', 'mdps_sas11_fs_sas_angle', 
    'ems11_n', 'esp12_lat_accel', 'esp12_long_accel', 
    'ems12_pv_av_can', 'ems12_brake_act']
columns_for_modeling_short = [
    'SPD', 'SAS_ANG', 'ENG_SPD', 'LAT_ACC', 'LONG_ACC', 
    'ACC_PD_V', 'BRAKE']


def run_ticc(data_for_modeling, number_of_clusters, prefix_string, args):
    ticc = GlobalTICC(window_size=args.ws, 
                      number_of_clusters=number_of_clusters, 
                      lambda_parameter=args.ld, 
                      beta=args.bt, 
                      maxIters=args.maxiter, 
                      threshold=args.threshold,
                      write_out_file=write_out_file, 
                      prefix_string=prefix_string, 
                      num_proc=args.num_proc,
                      compute_BIC=True, 
                      verbose=args.verbose)
    cluster_assignment_list, cluster_MRFs, bic, iters = \
        ticc.fit(data_for_modeling)

    return ticc, iters, bic, cluster_MRFs, cluster_assignment_list


def loop_ticc_modeling(data_list, tst_data_list, prefix_string, args):
    best_nc = 0
    best_ticc = None
    best_bic = 1e10
    best_cluster_assignment_list = None
    best_cluster_MRFs = None

    trn_bic_list = []
    tst_bic_list = []

    # find the best num of clusters based on bic score
    for number_of_clusters in range(args.min_nc, args.max_nc+1):

        # ./output_folder/vin/global/ld=#bt=#ws=#nc=#/solution.pkl
        this_prefix_string = prefix_string + "nC=" + str(number_of_clusters)
        utils.maybe_exist(this_prefix_string)
        this_solution_path = this_prefix_string + "/solution.pkl"

        ### if: solution is already obtained ==> load solution
        if os.path.exists(this_solution_path):
            # load solution
            _, ticc, iters, bic, cluster_MRFs, cluster_assignment_list = \
                utils.load_solution(this_solution_path)
            # dump solution
            solution_dict = {
                'ticc': ticc,
                'number_of_clusters': number_of_clusters,
                'bic': bic,
                'cluster_assignment': cluster_assignment_list,
                'cluster_MRFs': cluster_MRFs}
            utils.dump_solution(solution_dict, this_solution_path)
            

        ### else: no solution exist ==> run ticc and get solution
        else:
            # run ticc
            ticc, iters, bic, cluster_MRFs, cluster_assignment_list = \
                run_ticc(data_list, number_of_clusters, this_prefix_string, args)
            # dump solution
            solution_dict = {
                'ticc': ticc,
                'iters': iters,
                'number_of_clusters': number_of_clusters,
                'bic': bic,
                'cluster_assignment': cluster_assignment_list,
                'cluster_MRFs': cluster_MRFs}
            utils.dump_solution(solution_dict, this_solution_path)

        #---#
        _, bic = ticc.test(trn_data_list)
        # dump solution
        solution_dict = {
            'ticc': ticc,
            'iters': iters,
            'number_of_clusters': number_of_clusters,
            'bic': bic,
            'cluster_assignment': cluster_assignment_list,
            'cluster_MRFs': cluster_MRFs}
        utils.dump_solution(solution_dict, this_solution_path)
        #---#
        
        # test
        if bic < 1e10:
            _, bic_tst = ticc.test(tst_data_list)
        else:
            bic_tst = np.nan
        print('nC={} | iter={} | bic(trn)={:.1f}, bic(tst)={:.1f}'
              .format(number_of_clusters, iters, bic, bic_tst), end='')
        
        trn_bic_list.append(bic)
        tst_bic_list.append(bic_tst)

        if bic < best_bic:
            print(' --best up to now!')
            best_ticc = ticc
            best_nc = number_of_clusters
            best_bic = bic
            best_cluster_assignment_list = cluster_assignment_list
            best_cluster_MRFs = cluster_MRFs
        else:
            print('')
        
    cluster_assignment_list_tst, bic_tst = best_ticc.test(tst_data_list)
    if args.max_nc - args.min_nc > 0:
        draw_bic_plot(trn_bic_list, tst_bic_list, prefix_string, args)

    return (best_nc, best_bic, best_cluster_MRFs, best_cluster_assignment_list, 
            bic_tst, cluster_assignment_list_tst)


def save_viz_and_df(ign_on_time_list, 
                    data_list, 
                    path_list, 
                    cluster_assignment_list, 
                    directory, 
                    args, 
                    key='trn'):

    assert key in ['trn', 'tst']

    columns = ['ign_on_time', 't', 'longitude', 'latitude'] + \
        columns_for_modeling_short + ['cluster_assignment']
    df_result = pd.DataFrame(columns=columns)

    total_length = len(ign_on_time_list)
    counter = 1

    for ign_on_time, data, path, cluster_assignment in \
            zip(ign_on_time_list, 
                data_list, 
                path_list, 
                cluster_assignment_list):

        print('\r[{}] process: {} / {} ({:.1f}%)'
              .format(key, counter, total_length, 
              counter/total_length * 100), end='')
        counter += 1

        # adjust cluster_assignment (oniy if window_size > 1)
        adjustment = [cluster_assignment[0]] * (args.ws - 1)
        cluster_assignment = adjustment + list(cluster_assignment)
        
        # data
        ign_on_time_df = pd.Series([ign_on_time] * len(data))
        t_df = pd.Series(np.arange(len(data)))
        path_df = pd.DataFrame(path)
        data_df = pd.DataFrame(data)
        cluster_assignment_df = pd.Series(cluster_assignment)
        this_df = pd.concat(
            [ign_on_time_df, t_df, path_df, data_df, cluster_assignment_df], axis=1)
        this_df.columns = columns
        df_result = df_result.append(this_df)


        # WHERE THE OPTIMAL SOLUTION WILL BE SAVED
        data_path = directory + key + "_inputData_and_clusterID.csv"
        if not os.path.exists(data_path):
            df_result.to_csv(data_path, index=None)


        # draw signal plot
        sig_path = (directory + key + '_' + 
                    utils.strftime(ign_on_time) + '_signal.png')
        if not os.path.exists(sig_path):
            trip_str = utils.get_str_of_trip(args.target_vin, ign_on_time)
            data_df.columns = columns_for_modeling_short
            plot_clustering_result(data_df, 
                                   cluster_assignment,
                                   figsize=(24, 16),
                                   title=trip_str,
                                   save_to=sig_path, 
                                   show=False)

        # draw path plot
        xy_path = (directory + key + '_' + 
                   utils.strftime(ign_on_time) + '_path.png')
        if not os.path.exists(xy_path):
            longitude = this_df.longitude.values
            latitude = this_df.latitude.values
            zero_idx = [i for i in range(len(longitude))
                        if longitude[i] == 0 and latitude[i] == 0]
            longitude = np.delete(longitude, zero_idx)
            latitude = np.delete(latitude, zero_idx)
            cluster_assignment = np.delete(cluster_assignment, zero_idx)

            trip_str = utils.get_str_of_trip(args.target_vin, ign_on_time)
            plot_path_by_segment(longitude, 
                                latitude, 
                                cluster_assignment=cluster_assignment, 
                                title=trip_str, 
                                save_to=xy_path, 
                                show=False)



def betweenness_centrality(cluster_MRFs, 
                           number_of_clusters,
                           cluster_assignment_list, 
                           path_list,
                           directory, 
                           args):

    bc_path = os.path.join( # betweenness centrality csv
        directory, 'bc_matrix(thres=' + str(args.threshold) +').csv')
    hm_path = os.path.join( # betweenness centrality heatmap
        directory, 'bc_heatmap(thres=' + str(args.threshold) +').png')

    exist = os.path.exists(bc_path) and os.path.exists(hm_path) 
    if not exist:
        if number_of_clusters > 0:
            print('Calculate betweenness centrality.')
            ncol = len(columns_for_modeling_short)
            gp_labels = columns_for_modeling_short
            BC_array = np.zeros((number_of_clusters, ncol))
            
            if args.ws == 1:
                for k, MRF in cluster_MRFs.items():
                    gp_path = os.path.join( # networkx graph 
                        directory, 'graph_k={}(thres={}).png'
                        .format(k, args.threshold))
                    bc_list = calculate_bc_and_draw_graph(
                        MRF, 
                        labels=gp_labels, 
                        threshold=args.threshold, 
                        weighted=True,
                        title='Graph between Sensors (k={})'.format(k),
                        save_to=gp_path)
                    BC_array[k, :] = bc_list
            
            if args.ws == 2:
                for k, MRF in cluster_MRFs.items():
                    MRF_intra_time = (MRF[:ncol, :ncol] + MRF[ncol:, ncol:]) /2
                    gp_path = os.path.join( # networkx graph 
                        directory, 'graph_k={}(thres={}).png'
                        .format(k, args.threshold))
                    bc_list = calculate_bc_and_draw_graph(
                        MRF_intra_time, 
                        labels=gp_labels, 
                        threshold=args.threshold, 
                        weighted=True,
                        title='Graph between Sensors (k={})'.format(k),
                        save_to=gp_path)
                    BC_array[k, :] = bc_list

                    # for visualizing the whole nodes (w=1, 2)
                    gp_crosstime_path = os.path.join( # networkx graph 
                        directory, 'graph_corss_time_k={}(thres={}).png'
                        .format(k, args.threshold))
                    gp_crosstime_labels = \
                        [c + '(1)' for c in columns_for_modeling_short] +\
                            [c + '(2)' for c in columns_for_modeling_short]
                    _ = calculate_bc_and_draw_graph(
                        MRF, 
                        labels=gp_crosstime_labels, 
                        threshold=args.threshold, 
                        weighted=True,
                        title='Cross-time Graph (k={})'.format(k),
                        save_to=gp_crosstime_path)
                
            BC_df = pd.DataFrame(BC_array, columns=gp_labels)
            
            # SAVE heatmap
            with sns.axes_style("white"):
                plt.subplots(figsize=(15, 15))
                sns.heatmap(BC_df, square=True, annot=BC_df)
                plt.savefig(hm_path)
                plt.close('all')

            # calculate average distance (km/h) per cluster
            dist_by_cluster_dict = {k: [] for k in range(number_of_clusters)}
            for path, cluster_assignment in zip(path_list, 
                                                cluster_assignment_list):
                longitude, latitude = list(zip(*path))
                assert len(longitude) - args.ws + 1 == len(cluster_assignment)
                for i in range(1, len(cluster_assignment)):
                    c = cluster_assignment[i]
                    # (km / sec) * (3600 sec/hour) = km/hour
                    dist = utils.trip_length(longitude[i-1:i+1], 
                                             latitude[i-1:i+1]) * 3600
                    dist_by_cluster_dict[c].append(dist)
            BC_df['avg(km/h)'] = \
                [np.mean(dist_by_cluster_dict[k]) for k in range(number_of_clusters)]

            # SAVE dataframe
            BC_df.to_csv(bc_path)
        else:
            print('No solution. pass.')
    else:
        print('Already calculated. pass.')



if __name__ == '__main__':

    # argument parser
    parser = argparse.ArgumentParser()

    parser.add_argument(
        'target_vin', 
        type=str, 
        default=None,
        help='target vin')
    parser.add_argument(
        '--ws', 
        type=int, 
        default=1,
        help='window size')
    parser.add_argument(
        '--verbose', 
        type=bool, 
        nargs='?',
        default=False, #default
        const=True, #if the arg is given
        help='verbose for TICC')
    parser.add_argument(
        '--num_proc', 
        type=int, 
        default=4,
        help='the number of threads')

    parser.add_argument(
        '--ld', 
        type=float, 
        default=5e-3,
        help='lambda (sparsity in Toplitz matrix)')
    parser.add_argument(
        '--bt', 
        type=float, 
        default=200,
        help='beta (segmentation penalty)')
    parser.add_argument(
        '--maxiter', 
        type=int, 
        default=100,
        help='maxiter')
    parser.add_argument(
        '--threshold', 
        type=float, 
        default=2e-5,
        help='threshold')
    parser.add_argument(
        '--min_nc', 
        type=int, 
        default=3,
        help='min_num_cluster')
    parser.add_argument(
        '--max_nc', 
        type=int, 
        default=20,
        help='max_num_cluster')

    parser.add_argument(
        '--test-size', 
        type=float, 
        default=0.3,
        help='test data size')

    # Parse input arguments
    args, unparsed = parser.parse_known_args()
    write_out_file = True

    print('')
    print("lambda (sparsity penalty)", args.ld)
    print("beta (switching penalty)", args.bt)
    print("w (window size)", args.ws)
    print('')

    # Load rawdata (containing all trips)
    # (assume there is a target_vin data pre-dumped on local)
    rawdata_path = '../data/{}_{}.csv'.format(tablename, args.target_vin)
    if not os.path.exists(rawdata_path):
        df = utils.download_data(args.target_vin, save_to=rawdata_path)
    else:
        df = pd.read_csv(rawdata_path, low_memory=False)

    #######################################################################[:10]
    ign_on_time_list = list(df.ign_on_time.unique())[:100]
    num_trips = len(ign_on_time_list)
    if num_trips > 5:
        test_size = int(num_trips * args.test_size)
        trn_ign_on_time_list = ign_on_time_list[:-test_size]
        tst_ign_on_time_list = ign_on_time_list[-test_size:]
    else:
        trn_ign_on_time_list = ign_on_time_list
        tst_ign_on_time_list = ign_on_time_list
    
    print('target_vin: {}'.format(args.target_vin))
    print('...data loaded: n_trips={} (trn={} / tst= {})\n'
          .format(num_trips, len(trn_ign_on_time_list), len(tst_ign_on_time_list)))


    # prefix_string: ./output_folder/vin/global/
    basedir = 'output_folder/{}/global/'.format(args.target_vin)
    utils.maybe_exist(basedir)
    prefix_string = (basedir + 'ws={}ld={}bt={}'
                     .format(args.ws, args.ld, args.bt))

    #################
    # LOAD DATA
    #################
    trn_data_list = []
    tst_data_list = []
    trn_path_list = []
    tst_path_list = []
    
    for i, ign_on_time in enumerate(trn_ign_on_time_list):
        df_row_filtered = utils.filter_df_by(df, args.target_vin, ign_on_time)

        data_for_modeling = df_row_filtered[columns_for_modeling].values
        longitude = df_row_filtered.longitude.values
        latitude = df_row_filtered.latitude.values

        trn_data_list.append(data_for_modeling)
        trn_path_list.append(list(zip(longitude, latitude)))

    for i, ign_on_time in enumerate(tst_ign_on_time_list):
        df_row_filtered = utils.filter_df_by(df, args.target_vin, ign_on_time)

        data_for_modeling = df_row_filtered[columns_for_modeling].values
        longitude = df_row_filtered.longitude.values
        latitude = df_row_filtered.latitude.values

        tst_data_list.append(data_for_modeling)
        tst_path_list.append(list(zip(longitude, latitude)))

    #################
    # FIND SOLUTION
    #################

    print('> Find soultion ...')
    (number_of_clusters, bic, cluster_MRFs, cluster_assignment_list, 
        bic_tst, cluster_assignment_list_tst) = \
        loop_ticc_modeling(trn_data_list, tst_data_list, prefix_string, args)

    print('>>> Solution: nC={}, bic={:.1f}, bic(tst)={:.1f}'
    .format(number_of_clusters, bic, bic_tst))

    ##########################################
    # - SAVE DATAFRAME WITH CLUSTERING RESULT
    # - BETWEENNESS CENTRALITY
    # - VISUALIZE THE SOLUTION (SIGNAL, PATH)
    ##########################################

    # output_folder/vin/parameter###_result/
    directory = prefix_string + 'nC=' + str(number_of_clusters) + '_result/'
    utils.maybe_exist(directory)
    
    print('> Save betweenness centrality result ... ', end='') 
    # the result will be printed in the next function
    betweenness_centrality(cluster_MRFs, 
                           number_of_clusters,
                           cluster_assignment_list, 
                           trn_path_list,
                           directory, 
                           args)

    print('> Save dataframe with clustering result ...')
    print('> Save visualization ...')
    save_viz_and_df(trn_ign_on_time_list, 
                    trn_data_list, 
                    trn_path_list, 
                    cluster_assignment_list, 
                    directory, 
                    args, 
                    key='trn')
    save_viz_and_df(tst_ign_on_time_list, 
                    tst_data_list, 
                    tst_path_list, 
                    cluster_assignment_list_tst, 
                    directory, 
                    args, 
                    key='tst')