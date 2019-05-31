import os
import pickle
from geopy.distance import vincenty as dist

import numpy as np
import pandas as pd

protocol = pickle.HIGHEST_PROTOCOL

def get_solution_path(prefix_string, 
                      lambda_parameter, 
                      switch_penalty, 
                      window_size):
    return (prefix_string + 
            "ld=" + str(lambda_parameter) + 
            "bt=" + str(switch_penalty) +
            "ws=" + str(window_size) + 
            "_solution.pkl")

def dump_solution(solution_dict, soultion_path):
    with open(soultion_path, 'wb') as handle:
        pickle.dump(solution_dict, handle, protocol=protocol)

def load_solution(soultion_path):
    with open(soultion_path, 'rb') as handle:
        solution_dict = pickle.load(handle)
        number_of_clusters = solution_dict['number_of_clusters']
        bic = solution_dict['bic']
        cluster_MRFs = solution_dict['cluster_MRFs']
        cluster_assignment = solution_dict['cluster_assignment']
    return number_of_clusters, bic, cluster_MRFs, cluster_assignment

def maybe_exist(directory):
    """make sure the existence of given directory"""
    if not os.path.exists(directory):
        os.makedirs(directory)


def download_data(target_vin, save_to=None):
    import pyodbc

    conn = pyodbc.connect('Dsn=Hive;uid=h1903174;pwd=h1903174!01', 
                          autocommit=True)

    tablename = 'a302_rawsmpl_new_tma_2019_g'
    query = '''\
    SELECT vin, ign_on_time, trip_match_key, t, 
        latitude, longitude, 
        ems11_n, ems11_vs, ems12_brake_act, tcu12_cur_gr, ems11_tqi_acor, 
        mdps_sas11_fs_sas_angle, sas_sas11_fs_sas_angle, sas11_fs_sas_speed, 
        esp12_lat_accel, esp12_long_accel, ems12_pv_av_can
    FROM h1903174.%s
    WHERE vin in ("%s")''' % (tablename, target_vin)
    print('--- Load data from query below:')
    print(query)
    
    df = pd.read_sql(query, conn)
    if save_to is not None:
        df.to_csv(save_to, index=None)
    return df


def strftime(ign_on_time):
    ign_on_time = pd.Timestamp(ign_on_time).to_pydatetime()
    ign_on_time_str = ign_on_time.strftime("%Y%m%d %Hh%Mm%Ss")
    return ign_on_time_str


def get_str_of_trip(vin, ign_on_time):
    return ('VIN: {} | ign_on_time: {}'
            .format(vin, strftime(ign_on_time)))


def filter_df_by(df_, vin, ign_on_time):
    df = df_.copy()
    df_filtered = \
        df[(df['vin'] == vin) & (df['ign_on_time'] == ign_on_time)]
    df_filtered_sorted = df_filtered.sort_values(by=['t'])
    return df_filtered_sorted


def filter_to_one_path(df_, vin, ign_on_time):
    df = filter_df_by(df_, vin, ign_on_time)
    x, y = df.longitude.values, df.latitude.values
    return x, y


def trip_length(longitude, latitude):
    # vincenty(dist) function get (latitude, longitude) pair!
    xy = np.array([latitude, longitude]).T.reshape(-1, 2)
    
    total_length = 0.
    for start, stop, in zip(xy[:-1], xy[1:]):
        total_length += dist(start, stop).km
    
    return total_length


class Recorder(object):

    def __init__(self, fname):
        self.fname = fname
        self.str_to_record = ''

    def append_values(self, values):
        base_str = '{},' * len(values)
        self.str_to_record += base_str.format(*values)

    def next_line(self):
        with open(self.fname, 'a') as fout:
            self.str_to_record += '\n'
            fout.write(self.str_to_record)
            self.str_to_record = ''