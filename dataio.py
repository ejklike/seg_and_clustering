"""
ALL ABOUT DATA I/O.
"""

import os

import numpy as np
import pandas as pd

from utils import strftime

#####################
# DATA SETTING
#####################

COLUMNS_FOR_MODELING = [
    'ems11_vs', 'mdps_sas11_fs_sas_angle', 
    'ems11_n', 'esp12_lat_accel', 'esp12_long_accel', 
    'ems12_pv_av_can', 'ems12_brake_act']
COLUMNS_FOR_MODELING_SHORT = [
    # (caution) same order as "columnss_for_modeling"!
    'SPD', 'SAS_ANG', 'ENG_SPD', 'LAT_ACC', 'LNG_ACC', 
    'ACC_PD_V', 'BRAKE']


__all__ = ['DataLoader', 
           'get_data_split_keys', 
           'get_data_and_path_list_from_split']


def _filter_df_by(df_, ign_on_time):
    df = df_.copy()
    df_filtered = df[(df['ign_on_time'] == ign_on_time)]
    df_filtered_sorted = df_filtered.sort_values(by=['t'])
    return df_filtered_sorted


def _filter_df_by_ign_on_time(df, ign_on_time):
    df_row_filtered = _filter_df_by(df, ign_on_time).reset_index()
    df_for_modeling = df_row_filtered[COLUMNS_FOR_MODELING]
    df_for_modeling.columns = COLUMNS_FOR_MODELING_SHORT
    df_lng_lat = df_row_filtered[['longitude', 'latitude']]
    return df_for_modeling, df_lng_lat


def get_data_split_keys(ign_on_time_list, test_size=0.3):    
    num_trips = len(ign_on_time_list)
    
    if num_trips > 5:
        test_size = int(num_trips * test_size)
        trn_ign_on_time_list = ign_on_time_list[:-test_size]
        tst_ign_on_time_list = ign_on_time_list[-test_size:]
    else:
        trn_ign_on_time_list = ign_on_time_list
        tst_ign_on_time_list = ign_on_time_list

    print('...data loaded: n_trips={} (trn={} / tst= {})\n'
        .format(num_trips, 
                len(trn_ign_on_time_list), 
                len(tst_ign_on_time_list)))
    
    return trn_ign_on_time_list, tst_ign_on_time_list


def get_data_and_path_list_from_split(df, trn_ign_on_time_list, tst_ign_on_time_list):
    """
    return lists of pd.DataFrames
    """
    trn_data_list, tst_data_list = [], []
    trn_path_list, tst_path_list = [], []
    
    for ign_on_time in trn_ign_on_time_list:
        df_for_modeling, df_lng_lat = \
            _filter_df_by_ign_on_time(df, ign_on_time)
        trn_data_list.append(df_for_modeling)
        trn_path_list.append(df_lng_lat)

    for ign_on_time in tst_ign_on_time_list:
        df_for_modeling, df_lng_lat = \
            _filter_df_by_ign_on_time(df, ign_on_time)
        tst_data_list.append(df_for_modeling)
        tst_path_list.append(df_lng_lat)

    return trn_data_list, tst_data_list, trn_path_list, tst_path_list


class DataLoader:

    def __init__(self):
        
        # HIVE SETTING FOR LOADING DATA
        #  - MODIFY AS YOU WANT
        self.connection_setting = 'Dsn=Hive;uid=h1903174;pwd=h1903174!01'
        tablename = 'a302_rawsmpl_new'
        target_vin = '5NMS33AD0KH034994'
        # target_vin = '5NMS5CAA5KH018550'
        self.query = '''\
            SELECT vin, ign_on_time, trip_match_key, t, 
                latitude, longitude, 
                ems11_n, ems11_vs, ems12_brake_act, tcu12_cur_gr, ems11_tqi_acor, 
                mdps_sas11_fs_sas_angle, sas_sas11_fs_sas_angle, sas11_fs_sas_speed, 
                esp12_lat_accel, esp12_long_accel, ems12_pv_av_can
            FROM h1903174.%s
            WHERE vin in ("%s")''' % (tablename, target_vin)

        # DOWNLOADED DATA WILL BE STORED IN THIS PATH
        self.rawdata_path = \
            '../data/{}_{}.csv'.format(tablename, target_vin)

    def _download_data(self):
        import pyodbc
        
        conn = pyodbc.connect(self.connection_setting, autocommit=True)

        print('--- Load data from query below:')
        print(self.query)
        
        df = pd.read_sql(self.query, conn)
        df.to_csv(self.rawdata_path, index=None)
        return df

    def load_rawdata_containing_all_trips(self):
        """
        Download the target_vin data from HIVE.
        
        Once the data has been download, 
        it will be stored on the local for efficiency.
        
        (Next time, we will use the pre-dumped data.)
        """
        if not os.path.exists(self.rawdata_path):
            df = self._download_data()
        else:
            df = pd.read_csv(self.rawdata_path, low_memory=False)

        return df