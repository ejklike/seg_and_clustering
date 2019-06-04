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
python ticc_with_bic.py 5NMS33AD0KH034994 
python ticc_with_bic.py 5NMS33AD6KH026656 
-------
python ticc_with_bic.py 5NMS53AD9KH003365 
python ticc_with_bic.py 5NMS5CAA5KH018550 --verbose --min_nc 3 --max_nc 20 --maxiter 10 --ws 1

-------

python ticc_with_bic.py 5NMS5CAA5KH018550 --verbose --min_nc 13 --max_nc 13 --maxiter 10 --ws 1
python ticc_with_bic.py 5NMS5CAA5KH018550 --verbose --min_nc 8 --max_nc 8 --maxiter 10 --ws 2
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

from TICC.TICC_solver import TICC

import utils
from viz_segment import (plot_path_by_segment, 
                         plot_path_sequence,
                         plot_clustering_result)
# from betweeness_centrality import betweeness_centrality_wrapper

#####################
# DATA SETTING
#####################

tablename = 'a302_rawsmpl_new'
# target_vin = '5NMS33AD0KH034994'
target_vin = '5NMS33AD6KH026656'

columns_for_modeling = [
    'ems11_vs', 'mdps_sas11_fs_sas_angle', 
    'ems11_n', 'esp12_lat_accel', 'esp12_long_accel', 
    'ems12_pv_av_can', 'ems12_brake_act']
    # 'acc_brake']
colname_dict = {
    'ems11_vs': 'SPD', 
    'mdps_sas11_fs_sas_angle': 'SAS_ANG', 
    'ems11_n': 'ENG_SPD', 
    'esp12_lat_accel': 'LAT_ACC', 
    'esp12_long_accel': 'LON_ACC', 
    'ems12_pv_av_can': 'ACC_PD_V',
    'ems12_brake_act': 'BRAKE'
    # 'acc_brake': 'ACC_BRAKE'
}
    # 'ems12_brake_on': 'BRK_ON', 
    # 'ems12_brake_off': 'BRK_OFF',
    # 'tcu12_cur_gr': 'GEAR', 
    # 'ems11_tqi_acor': 'ENG_TQ',
    # 'sas_sas11_fs_sas_angle': 'SAS11_ANG', 
    # 'sas11_fs_sas_speed': 'SAS11_SPD',


def shorten_colnames(df_):
    df = df_.copy()
    for colname_old in df.columns:
        if colname_old in colname_dict.keys():
            colname_new = colname_dict[colname_old]
            df[colname_new] = df[colname_old]
            df.drop(columns=[colname_old], inplace=True)
    return df


def run_ticc(data_for_modeling, number_of_clusters, prefix_string, args):

    # EXCEPTION criteria: 
    # n_sample is insufficient compared to n_component
    n_sample = data_for_modeling.shape[0]
    n_sample_is_insufficient = \
        n_sample - args.ws + 1 <= number_of_clusters
    if n_sample_is_insufficient:
        cluster_assignment = None
        cluster_MRFs = None
        bic = -1e10

    else:
        ticc = TICC(window_size=args.ws, 
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
        cluster_assignment, cluster_MRFs, bic = \
            ticc.fit(data_for_modeling)

    return bic, cluster_MRFs, cluster_assignment


def loop_ticc_modling(data_for_modeling, cmin, cmax, prefix_string):
    best_nc = 0
    best_bic = -1e10
    best_cluster_assignment = None
    best_cluster_MRFs = None

    # find the best num of clusters based on bic score
    for number_of_clusters in range(cmin, cmax+1):

        # ./output_folder/vin/ign_on_time/ld=#bt=#ws=#nc=#/solution.pkl
        this_prefix_string = prefix_string + "nC=" + str(number_of_clusters)
        utils.maybe_exist(this_prefix_string)
        this_solution_path = this_prefix_string + "/solution.pkl"

        ### if: solution is already obtained ==> load solution
        if os.path.exists(this_solution_path):
            # load solution
            _, bic, cluster_MRFs, cluster_assignment = \
                utils.load_solution(this_solution_path)
        ### else: no solution exist ==> run ticc and get solution
        else:
            # run ticc
            bic, cluster_MRFs, cluster_assignment = \
                run_ticc(data_for_modeling, number_of_clusters, this_prefix_string, args)
            # dump solution
            solution_dict = {
                'number_of_clusters': number_of_clusters,
                'bic': bic,
                'cluster_assignment': cluster_assignment,
                'cluster_MRFs': cluster_MRFs}
            utils.dump_solution(solution_dict, this_solution_path)

        print('n_cluster={} | bic={:.1f}'
              .format(number_of_clusters, bic), end='')

        if bic > best_bic:
            print(' --best up to now!')
            best_nc = number_of_clusters
            best_bic = bic
            best_cluster_assignment = cluster_assignment
            best_cluster_MRFs = cluster_MRFs
        else:
            print('')

    return best_nc, best_bic, best_cluster_MRFs, best_cluster_assignment


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

    # Parse input arguments
    args, unparsed = parser.parse_known_args()
    write_out_file = True

    # Load rawdata 
    # (assume there is a target_vin data pre-dumped on local)
    rawdata_path = '../data/{}_{}.csv'.format(tablename, args.target_vin)
    if not os.path.exists(rawdata_path):
        df = utils.download_data(args.target_vin, save_to=rawdata_path)
    else:
        df = pd.read_csv(rawdata_path, low_memory=False)
    print('...data loaded\n')

    print("lam_sparse", args.ld)
    print("switch_penalty", args.bt)
    print("num stacked", args.ws)
    print('')

    # loop modeling for all trip
    ign_on_time_list = list(df.ign_on_time.unique())# [240:]
    for i, ign_on_time in enumerate(ign_on_time_list):

        # trip_str: [VIN: ... | ign_on_time: ... ]
        trip_str = utils.get_str_of_trip(args.target_vin, ign_on_time)
        print('--- {} ({}/{})'.format(trip_str, i+1, len(ign_on_time_list)))

        # prefix_string: 
        # ./output_folder/vin/ign_on_time/
        base_folder = (
            'output_folder/' + args.target_vin + '/' 
            + utils.strftime(ign_on_time) + '/')
        utils.maybe_exist(base_folder)

        prefix_string = (base_folder
                         + "ld=" + str(args.ld) 
                         + "bt=" + str(args.bt)
                         + "ws=" + str(args.ws))
        
        # WHERE THE OPTIMAL SOLUTION WILL BE SAVED
        solution_path = prefix_string + "_solution.pkl"
        data_path = prefix_string + "_inputData_and_clusterID.csv"
        sig_path_wild = prefix_string + '*_signal.png'
        xy_path_wild = prefix_string + '*_path.png'

        # check if it is already modeled.
        is_solution_found = os.path.exists(solution_path)
        is_data_and_clusterID_saved = os.path.exists(data_path)
        is_signal_visualized = len(glob.glob(sig_path_wild)) > 0
        is_path_visualized = len(glob.glob(xy_path_wild)) > 0

        print('--- Check File Existence ...')
        print('--- Solution={} | DataFrame={} | SigViz={} | PathViz={}'
              .format(is_solution_found, 
                      is_data_and_clusterID_saved, 
                      is_signal_visualized, 
                      is_path_visualized))

        require_data_and_solution = not (
            is_solution_found and
            is_data_and_clusterID_saved and
            is_signal_visualized and
            is_path_visualized)

        if require_data_and_solution:

            #################
            # LOAD DATA
            #################

            ### prepare data
            print('> Load data ...', end='')
            if not os.path.exists(data_path):
                df_row_filtered = utils.filter_df_by(df, args.target_vin, ign_on_time)
            else:
                df_row_filtered = pd.read_csv(data_path, low_memory=False)

            df_for_modeling = shorten_colnames( # filter columns
                df_row_filtered[columns_for_modeling])
            data_for_modeling = df_for_modeling.values # nparray for modeling
            # data summary
            n_row = len(df_row_filtered)
            duration = n_row / 60
            dist = utils.trip_length(df_row_filtered.longitude, 
                                     df_row_filtered.latitude)
            print('Total {} rows ({:.1f}km for {:.0f}min)'
                .format(n_row, dist, duration))

            #################
            # FIND SOLUTION
            #################
            
            ### if: solution is already obtained ==> load solution
            if is_solution_found:
                print('> Load solution ...', end='')
                number_of_clusters, bic, cluster_MRFs, cluster_assignment = \
                    utils.load_solution(solution_path)
                print('num_cluster={}'.format(number_of_clusters))
            ### else: no solution exist ==> loop for all number of cluster sizes
            else:
                # ### prepare data
                # df_row_filtered = utils.filter_df_by(df, target_vin, ign_on_time)
                # df_for_modeling = shorten_colnames( # filter columns
                #     df_row_filtered[columns_for_modeling])
                # data_for_modeling = df_for_modeling.values # nparray for modeling

                ### loop modeling
                print('> Find soultion ...')
                number_of_clusters, bic, cluster_MRFs, cluster_assignment = \
                    loop_ticc_modling(data_for_modeling, 
                                      args.min_nc, 
                                      args.max_nc,
                                      prefix_string)
                ### dump solution
                solution_dict = {
                    'number_of_clusters': number_of_clusters,
                    'bic': bic,
                    'cluster_assignment': cluster_assignment,
                    'cluster_MRFs': cluster_MRFs}
                utils.dump_solution(solution_dict, solution_path)
            print('> Solution: nC={}, bic={}'.format(number_of_clusters, bic))

        ##########################################
        # SAVE DATAFRAME WITH CLUSTERING RESULT
        ##########################################

        if not is_data_and_clusterID_saved:
            if number_of_clusters > 0.:
                # adjust cluster_assignment (oniy if window_size > 1)
                adjustment = [cluster_assignment[0]] * (args.ws - 1)
                cluster_assignment = adjustment + list(cluster_assignment)

            print('> Save dataframe with clustering result  ...')
            df_row_filtered['cluster_assignment'] = cluster_assignment
            df_row_filtered.to_csv(data_path, index=None)

        ##########################################
        # VISUALIZE THE SOLUTION (SIGNAL, PATH)
        ##########################################

        if not is_signal_visualized or not is_path_visualized:
            # check it is already visualized or not
            prefix_string_for_viz = \
                prefix_string + 'nC=' + str(number_of_clusters)
            # prefix_string_for_viz_seq = \
            #     prefix_string + 'nC=' + str(number_of_clusters) + '_seq/'
            # utils.maybe_exist(prefix_string_for_viz_seq)

        # draw png files
        if not is_signal_visualized:
            print('> Generate signal visualization png file ...')
            if number_of_clusters > 0.:
                sig_path = prefix_string_for_viz + '_signal.png'
            else:
                sig_path = prefix_string_for_viz + '(no_solution)_signal.png'

            plot_clustering_result(df_for_modeling, 
                                   cluster_assignment,
                                   figsize=(24, 16),
                                   title=trip_str,
                                   save_to=sig_path, 
                                   show=False)

        if not is_path_visualized:
            print('> Generate path visualization png file ...')

            longitude = df_row_filtered.longitude.values
            latitude = df_row_filtered.latitude.values

            if number_of_clusters > 0.:
                xy_path = prefix_string_for_viz + '_path.png'
                # xy_t_path = prefix_string_for_viz_seq + 'path(t={}).png'
                
            else:
                xy_path = prefix_string_for_viz + '(no_solution)_path.png'
                # xy_t_path = prefix_string_for_viz_seq + 'path(t={}).png'
            
            # n_row = len(longitude)
            zero_idx = [i for i in range(n_row) if longitude[i] == 0 and latitude[i] == 0]
            longitude = np.delete(longitude, zero_idx)
            latitude = np.delete(latitude, zero_idx)
            cluster_assignment = np.delete(cluster_assignment, zero_idx)

            plot_path_by_segment(longitude, 
                                 latitude, 
                                 cluster_assignment=cluster_assignment, 
                                 title=trip_str, 
                                 save_to=xy_path, 
                                 show=False)
            # print('>>>>>> Generate path "sequence" visualization png files ...')
            # plot_path_sequence(longitude, 
            #                    latitude, 
            #                    cluster_assignment=cluster_assignment, 
            #                    title=trip_str, 
            #                    save_to=prefix_string_for_viz + '_path_animated.gif')


            # zero_idx = [i for i in range(n_row) if longitude[i] == 0 and latitude[i] == 0]
            # if len(zero_idx) > 0:
            #     print('>>>>>> Exist zero latitude and longitute points: #={}'.format(len(zero_idx)))
            #     if number_of_clusters > 0.:
            #         xy_nonzero_path = prefix_string_for_viz + '_(nonzero)path.png'
            #     else:
            #         xy_nonzero_path = prefix_string_for_viz + '(no_solution)_(nonzero)path.png'

            #     print('>>>>>> Generate path (nonzero) visualization png file ...')
            #     plot_path_by_segment(np.delete(longitude, zero_idx), 
            #                          np.delete(latitude, zero_idx), 
            #                          cluster_assignment=np.delete(cluster_assignment, zero_idx), 
            #                          title=trip_str, 
            #                          save_to=xy_nonzero_path, 
            #                          show=False)

        if require_data_and_solution:
            del df_row_filtered, df_for_modeling, data_for_modeling