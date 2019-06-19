import argparse
import os
import warnings
from itertools import product
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

import utils
import dataio

from viz import (plot_path_by_segment, 
                 plot_clustering_result)
from graph import visualize_cluster_MRFs


COLUMNS_FOR_MODELING_SHORT = dataio.COLUMNS_FOR_MODELING_SHORT


def save_inference_results(ign_on_time_list, 
                           data_list, 
                           path_list, 
                           output_dict,
                           directory, 
                           args, 
                           key='trn'):

    assert key in ['trn', 'tst']

    # we will save the whold input data & their corresponding 
    # cluster assignment and log-likelihood results
    df_result = pd.DataFrame()

    # output_folder/vin/parameter###/nC=#_result/path_and_signals(key)
    directory2 = directory + 'path_and_signals({})/'.format(key)
    utils.maybe_exist(directory2)

    # get output from output_dict
    cluster_assignment_list = output_dict['cluster_assignment']
    nll_vector_list = output_dict['nll_vector']

    # adjust output vector lengths according to the window size
    cluster_assignment_list = \
        utils.adjust_output_list(cluster_assignment_list, args.ws)
    nll_vector_list = \
        utils.adjust_output_list(nll_vector_list, args.ws)

    print('> Save summary statistics per cluster ... ({})'.format(key), end='')
    _save_summary_statistics_per_cluster(
        data_list, cluster_assignment_list, directory, args)

    print('> Save segmentation results for all trips ... ({})'.format(key))

    # we will iterate over these lists
    iterator = zip(ign_on_time_list, 
                   data_list, 
                   path_list, 
                   cluster_assignment_list, 
                   nll_vector_list)

    for counter, (ign_on_time, 
                  data_df, 
                  path_df, 
                  cluster_assignment, 
                  nll_vector) in enumerate(iterator):

        # trip key and figtitle
        prefix_string = directory2 + utils.strftime(ign_on_time)
        fig_title = 'ign_on_time: {}'.format(utils.strftime(ign_on_time))

        # dataframe
        this_df = _concat_data(ign_on_time, 
                               data_df, 
                               path_df, 
                               cluster_assignment, 
                               nll_vector)
        # save this df
        this_data_path = prefix_string + '_inputData_and_clusterID.csv'
        if not os.path.exists(this_data_path):
            this_df.to_csv(this_data_path, index=None)
        df_result = df_result.append(this_df)

        # draw signal plot
        sig_path = prefix_string + '_signal.png'
        if not os.path.exists(sig_path):
            plot_clustering_result(data_df, 
                                   cluster_assignment,
                                   nll_vector,
                                   figsize=(15, 10),
                                   title=fig_title,
                                   save_to=sig_path)

        # draw path plot
        xy_path = prefix_string + '_path.png'
        if not os.path.exists(xy_path):
            longitude = path_df.longitude.values
            latitude = path_df.latitude.values
            plot_path_by_segment(longitude, 
                                 latitude, 
                                 cluster_assignment=cluster_assignment, 
                                 title=fig_title, 
                                 save_to=xy_path, 
                                 show=False)

        print('\r[{}] process: {} / {} ({:.1f}%)'
              .format(key, 
                      counter + 1, 
                      len(ign_on_time_list), 
                      (counter + 1) / len(ign_on_time_list) * 100), end='')
    print('')

    # SAVE ALL DATA
    print('> Save Input Data and its cluster assignment results ...')
    data_path = directory + key + "_inputData_and_clusterID.csv"
    if not os.path.exists(data_path):
        df_result.to_csv(data_path, index=None)



def _save_summary_statistics_per_cluster(data_list, 
                                         cluster_assignment_list, 
                                         directory, 
                                         args,
                                         key='trn'):
    assert key in ['trn', 'tst']

    summary_path = os.path.join(
        directory, 'summary_statistics_per_cluster_{}.csv'.format(key))
    if not os.path.exists(summary_path):
        # concat all signals into one dataframe
        data_df = pd.concat(data_list, axis=0)

        data_df['cluster_assignment'] = \
            np.concatenate(cluster_assignment_list, axis=0)
        
        # calculate summary stats per cluster and columns
        df = pd.DataFrame()
        for colname in COLUMNS_FOR_MODELING_SHORT:
            colnames = ['{}({})'.format(s, colname) 
                        for s in ['avg', 'std', 'min', 'max']]
            for k in range(args.nc):
                series = data_df[data_df['cluster_assignment'] == k][colname]
                df.loc[k, colnames[0]] = series.mean()
                df.loc[k, colnames[1]] = series.std()
                df.loc[k, colnames[2]] = series.min()
                df.loc[k, colnames[3]] = series.max()
        df.to_csv(summary_path)
        print(' Calculated and saved.')

    else:
        print(' Already exists. pass.')


def _concat_data(ign_on_time, 
                 data_df, 
                 path_df, 
                 cluster_assignment, 
                 nll_vector):
    
    # prepare for concat
    ign_on_time_series = \
        pd.Series([ign_on_time] * len(data_df), name='ign_on_time')
    t_series = pd.Series(np.arange(len(data_df)), name='t')
    cluster_assignment_series = \
        pd.Series(cluster_assignment, name='cluster_assignment')
    nll_series = pd.Series(nll_vector, name='nll')
    
    # concat all data
    this_df = pd.concat([ign_on_time_series,
                            t_series,
                            path_df,
                            data_df,
                            cluster_assignment_series,
                            nll_series], axis=1)
    return this_df


if __name__ == '__main__':

    # argument parser
    parser = argparse.ArgumentParser()

    parser.add_argument(
        'nc', 
        type=int, 
        default=3,
        help='num_cluster to interpret')
    parser.add_argument(
        '--ld', 
        type=float, 
        default=5e-3,
        help='lambda (sparsity in Toplitz matrix)')
    parser.add_argument(
        '--bt', 
        type=int, 
        default=200,
        help='beta (segmentation penalty)')
    parser.add_argument(
        '--ws', 
        type=int, 
        default=1,
        help='window size')

    parser.add_argument(
        '--threshold', 
        type=float, 
        default=2e-5,
        help='threshold')
    parser.add_argument(
        '--test-size', 
        type=float, 
        default=0.3,
        help='test data size')

    # Parse input arguments
    args, unparsed = parser.parse_known_args()

    print('')
    print("lambda (sparsity penalty): ", args.ld)
    print("beta (switching penalty):  ", args.bt)
    print("w (window size):           ", args.ws)
    print("K (number of cluster):     ", args.nc)
    print('')

    # basedir: ./output_folder/vin/parameters/
    args.basedir = utils.get_basedir(args)
    utils.maybe_exist(args.basedir)

    #################
    # LOAD SOLUTION
    #################
    
    # ./output_folder/vin/global/ld=#bt=#ws=#/nc=#/solution.pkl
    solution_path = args.basedir + "nC=" + str(args.nc) + "/solution.pkl"
    print('load solution stored in local ...')
    try:
        ticc, output_dict_trn, _ = \
            utils.load_solution(solution_path)
    except FileNotFoundError as err:
        print('No solution in local. '
              'Please try after training the model.\n')
        raise

    ################################################
    # - Load rawdata (containing all trips)
    # - Parse the data into 'ign_on_time' unit
    # - Split the set of unit data into training/test sets
    ################################################
    # SAME AS 'main.py'
    ################################################

    dataloader = dataio.DataLoader()
    df = dataloader.load_rawdata_containing_all_trips()

    # In this data, each trip can be 
    # distinguished by 'ign_on_time' field.
    ign_on_time_list = list(df.ign_on_time.unique())
    
    # For test, we will choose only 100 trip samples.
    # If you do not want to test, please comment this line.
    ign_on_time_list = ign_on_time_list[:100] #<>#

    # split all trips into trn/tst sets
    trn_ign_on_time_list, tst_ign_on_time_list = \
        dataio.get_data_split_keys(ign_on_time_list, test_size=args.test_size)
    
    # get trn/tst trip data
    trn_data_list, tst_data_list, trn_path_list, tst_path_list = \
        dataio.get_data_and_path_list_from_split(df, 
                                                 trn_ign_on_time_list, 
                                                 tst_ign_on_time_list)

    #########################################
    # INFERENCE
    #########################################
    # print('> Infer from training data ...')
    # output_dict_trn, _ = ticc.test(trn_data_list)
    print('> Infer from test data ...')
    output_dict_tst, _ = ticc.test(tst_data_list)

    ####################################################
    # INTERPRET MODEL AND SAVE RESULTS
    # - BETWEENNESS CENTRALITY
    # - SAVE DATAFRAME WITH CLUSTERING ASSIGNMENT RESULT
    # - VISUALIZE THE SOLUTION (SIGNAL, PATH)
    #####################################################

    # output_folder/vin/parameter###/nC=#_result/
    directory = args.basedir + 'nC=' + str(args.nc) + '_result/'
    utils.maybe_exist(directory)
    
    print('')
    print('All about theta:')
    visualize_cluster_MRFs(ticc.theta_dict, directory, args)

    print('')
    print('All about cluster assignment:')
    save_inference_results(trn_ign_on_time_list, 
                           trn_data_list, 
                           trn_path_list, 
                           output_dict_trn,
                           directory, 
                           args, 
                           key='trn')
    save_inference_results(tst_ign_on_time_list, 
                           tst_data_list, 
                           tst_path_list, 
                           output_dict_tst,
                           directory, 
                           args, 
                           key='tst')