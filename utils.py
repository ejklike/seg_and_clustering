import os
from geopy.distance import vincenty as dist

import numpy as np
import pandas as pd


def maybe_exist(directory):
    """make sure the existence of given directory"""
    if not os.path.exists(directory):
        os.makedirs(directory)


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