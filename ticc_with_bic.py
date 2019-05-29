"""
python ticc_with_bic.py [vin] [window_size]
---
python ticc_with_bic.py 5NMS33AD0KH034994 2
python ticc_with_bic.py 5NMS33AD6KH026656 2
"""

import os
import glob
import gc
import sys
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from TICC.TICC_solver import TICC

import my_ftn
from my_dist import trip_length

# import viz_font
from viz_segment import draw_path, plot_clustering_result, plot_path_by_segment
# from Visualize_function import visualize

#####################
# TICC SETTING
#####################

lambda_parameter = 5e-3 # sparsity in Toplitz matrix
beta = 200 # segmentation penalty
maxIters = 10
threshold = 2e-5
write_out_file = True
num_proc = 4

min_num_cluster = 3
max_num_cluster = 20

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


def download_data(target_vin, save_to=None):
    import pyodbc

    conn = pyodbc.connect('Dsn=Hive;uid=h1903174;pwd=h1903174!01', 
                      autocommit=True)
    cursor = conn.cursor()

    tablename = 'a302_rawsmpl_new_tma_2019_g'
    query = '''\
    SELECT vin, ign_on_time, trip_match_key, t, 
        latitude, longitude, 
        ems11_n, ems11_vs, ems12_brake_act, tcu12_cur_gr, ems11_tqi_acor, 
        mdps_sas11_fs_sas_angle, sas_sas11_fs_sas_angle, sas11_fs_sas_speed, 
        esp12_lat_accel, esp12_long_accel, ems12_pv_av_can
    FROM h1903174.%s
    WHERE vin in ("%s")''' % (tablename, target_vin)
    print(query)
    df = pd.read_sql(query, conn)
    if save_to is not None:
        df.to_csv(save_to, index=None)
    return df


def run_ticc(data_for_modeling, cmin=3, cmax=20):

    ticc_results = dict()
    best_n_cluster = 0
    best_bic = -1

    for number_of_clusters in range(cmin, cmax+1):
        
        # EXCEPTION criteria: n_sample vs. n_component
        if n_row - window_size + 1 <= number_of_clusters:
            print('n_cluster={} | n_sample={} is insufficient.'
                    .format(number_of_clusters, n_row))
            continue

        ticc = TICC(window_size=window_size, 
                    number_of_clusters=number_of_clusters, 
                    lambda_parameter=lambda_parameter, 
                    beta=beta, 
                    maxIters=maxIters, 
                    threshold=threshold,
                    write_out_file=write_out_file, 
                    prefix_string=prefix_string, 
                    num_proc=num_proc,
                    compute_BIC=True, 
                    verbose=False)
        cluster_assignment, cluster_MRFs, bic = ticc.fit(data_for_modeling)
        if cluster_assignment is not None:
            ticc_results[number_of_clusters] = {
                'cluster_assignment': cluster_assignment,
                'cluster_MRFs': cluster_MRFs,
                'bic': bic,
                'str_NULL': ticc.str_NULL}
        print('n_cluster={} | bic={:.1f}'.format(number_of_clusters, bic), end=' ')
        if bic > best_bic:
            best_n_cluster = number_of_clusters
            best_bic = bic
            print('--best up to now!')
        else:
            print('')
        del ticc
    
    # set model results
    number_of_clusters = best_n_cluster
    bic = best_bic

    if number_of_clusters == 0.:
        print(' > THE MODEL DOES NOT CONVERGE FOR ALL CLUSTER NUMBERS. < ')
        draw_path(df_row_filtered, target_vin, ign_on_time, 
            save_to=prefix_string + 'Result_xy_NOT_CONVERGE.png')

    else:
        # get model results
        cluster_assignment = ticc_results[number_of_clusters]['cluster_assignment']
        cluster_MRFs = ticc_results[number_of_clusters]['cluster_MRFs']
        str_NULL = ticc_results[number_of_clusters]['str_NULL']
        bic = ticc_results[number_of_clusters]['bic']
        # out to the ./output_folder/vin/ign_on_time
        str_NULL_list = str_NULL.split('/')
        str_NULL_prefix = str_NULL_list.pop(-2)
        str_NULL = '/'.join(str_NULL_list) + str_NULL_prefix + '_bic={:.1f}'.format(bic)


        # adjust cluster_assignment (oniy if window_size > 1)
        diff = len(df_for_modeling) - len(cluster_assignment)
        if diff > 0:
            cluster_assignment = \
                [cluster_assignment[0]] * diff + list(cluster_assignment)
        elif diff < 0:
            cluster_assignment = cluster_assignment[-diff:]
        # assign clustering results
        df_for_modeling.index =  cluster_assignment
        df_row_filtered.index = cluster_assignment
        # print(len(df_for_modeling), len(cluster_assignment))
        # assert len(df_for_modeling) == len(cluster_assignment)

        plot_clustering_result(
            df_for_modeling, figsize=(24, 16),
            title='TICC_clustering result | {}'.format(trip_str),
            save_to=str_NULL + '.png', show=False)

        draw_path(df_row_filtered, target_vin, ign_on_time, 
                cluster_assignment=cluster_assignment,
                save_to=str_NULL + '_xy.png', show=False)

        # CHECK MRF: dependencies btw variables
        str_NULL_mrf = str_NULL + '_mrf/'
        if not os.path.exists(os.path.dirname(str_NULL_mrf)):
            try:
                os.makedirs(os.path.dirname(str_NULL_mrf))
            except OSError as exc:  # Guard against race condition of path already existing
                raise
        vmax = np.max(list(cluster_MRFs.values()))
        vmin = np.min(list(cluster_MRFs.values()))
        for k, mrf in cluster_MRFs.items():
            with sns.axes_style("white"):
                f, ax = plt.subplots()
                sns.heatmap(mrf, square=True, vmin=vmin, vmax=vmax)
                plt.savefig(str_NULL_mrf + '/k={}.png'.format(k))
                plt.close('all')

        # ##With the inverses do some sort of thresholding
        # for cluster in range(number_of_clusters):
        #     out = (np.abs(cluster_MRFs[cluster]) > threshold).astype(np.int)
        #     file_name = str_NULL_mrf + "Cross time graphs.jpg"
        #     num_stacked = window_size - 1
            
        #     n = data_for_modeling.shape[1]
        #     assert n == out.shape[1] // window_size
        #     out2 = out[(num_stacked-1)*n:num_stacked*n, ]
        #     names = [colname_dict[c] for c in columns_for_modeling]
        #     if write_out_file:
        #         visualize(out2, -1, num_stacked, names, file_name)

    return number_of_clusters, bic


if __name__ == '__main__':

    target_vin = sys.argv[1]
    window_size = int(sys.argv[2])

    # Load data 
    # (assume there is a target_vin data pre-dumped on local)
    data_path = '../data/{}_{}.csv'.format(tablename, target_vin)
    if not os.path.exists(data_path):
        df = download_data(target_vin, save_to=data_path)
    else:
        df = pd.read_csv(data_path, low_memory=False)
    print('...data loaded\n')

    print("lam_sparse", lambda_parameter)
    print("switch_penalty", beta)
    print("num stacked", window_size)
    print('')

    # recorder object for saving meta data
    # vin,ign_on_time,ign_on_time_str,window_size,lambda,beta,n_cluster,bic
    recorder = my_ftn.Recorder('result_table_vin_{}.csv'.format(target_vin))

    # loop modeling for all trip
    ign_on_time_list = list(df.ign_on_time.unique())
    for ign_on_time in ign_on_time_list:

        # define some strirng
        trip_str = my_ftn.get_str_of_trip(target_vin, ign_on_time)
        prefix_string = (
            'output_folder/' + target_vin + '/' 
            + my_ftn.strftime(ign_on_time) + '/')
        my_ftn.maybe_exist(prefix_string)
        print('---', trip_str)

        # check if it is already modeled.
        fname_with_wildcard = (prefix_string + 
            "ld=" + str(lambda_parameter) + 
            "bt=" + str(beta) +
            "ws=" + str(window_size) + "*.png")
        result_figfiles = glob.glob(fname_with_wildcard)
        if len(result_figfiles) > 0:
            print('    Already modeled!')
            print('    (Check "{}")'.format(result_figfiles[0].split('/')[-1]))
            continue

        # filter to target vin and ign_on_time
        df_row_filtered = my_ftn.filter_df_by(df, target_vin, ign_on_time)
        # shorten column names
        # 1) basic
        df_for_modeling = shorten_colnames(df_row_filtered[columns_for_modeling])
        # 2) custom column (combining accel and brake)
        # df_row_filtered['acc_brake'] = \
        #     df_row_filtered['ems12_pv_av_can'] \
        #         - (df_row_filtered['ems12_brake_act'] > 1).astype(np.float32) * 10
        # df_for_modeling = shorten_colnames(df_row_filtered[columns_for_modeling])

        # np array for modeling
        # (option) standardize
        # mean = df_for_modeling.mean(axis=0).values
        # std = df_for_modeling.std(axis=0).values
        # data_for_modeling = (df_for_modeling.values - mean) / std
        data_for_modeling = df_for_modeling.values

        # trip summary
        n_row = len(df_row_filtered)
        dist = trip_length(df_row_filtered.longitude, 
                           df_row_filtered.latitude)
        duration = n_row / 60
        print('    Total {} rows ({:.1f}km for {:.0f}min)'
              .format(n_row, dist, duration))

        # loop for all number_of_clusters
        number_of_clusters, bic = run_ticc(data_for_modeling, 
                                           min_num_cluster, 
                                           max_num_cluster)
        
        # save meta data of results
        # vin,ign_on_time_str,ign_on_time,window_size,lambda,beta,n_cluster,bic
        recorder.append_values([
            target_vin, my_ftn.strftime(ign_on_time), ign_on_time, 
            n_row, dist, duration,
            window_size, lambda_parameter, beta, 
            number_of_clusters, bic, 
        ])
        recorder.next_line()

        del df_row_filtered, df_for_modeling, data_for_modeling
        gc.collect()