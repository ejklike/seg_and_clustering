import os
import sys
import inspect

import pandas as pd


# https://stackoverflow.com/questions/47908281/
# python-memory-consumption-of-objects-and-process
# def get_size(obj, seen=None):
#     """Recursively finds size of objects in bytes"""
#     size = sys.getsizeof(obj)
#     if seen is None:
#         seen = set()
#     obj_id = id(obj)
#     if obj_id in seen:
#         return 0
#     # Important mark as seen *before* entering recursion to gracefully handle
#     # self-referential objects
#     seen.add(obj_id)
#     if hasattr(obj, '__dict__'):
#         for cls in obj.__class__.__mro__:
#             if '__dict__' in cls.__dict__:
#                 d = cls.__dict__['__dict__']
#                 if inspect.isgetsetdescriptor(d) or inspect.ismemberdescriptor(d):
#                     size += get_size(obj.__dict__, seen)
#                 break
#     if isinstance(obj, dict):
#         size += sum((get_size(v, seen) for v in obj.values()))
#         size += sum((get_size(k, seen) for k in obj.keys()))
#     elif hasattr(obj, '__iter__') and not isinstance(obj, (str, bytes, bytearray)):
#         size += sum((get_size(i, seen) for i in obj))
#     return size


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

    # x_normal = [a for a, b in zip(x, y) if a != 0 and b != 0]
    # y_normal = [b for a, b in zip(x, y) if a != 0 and b != 0]
    # return x_normal, y_normal


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