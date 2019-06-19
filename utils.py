import os
import pickle
from geopy.distance import vincenty as dist

import numpy as np
import pandas as pd


protocol = pickle.HIGHEST_PROTOCOL

def get_basedir(args):
    return ('output_folder/ws={}ld={}bt={}/'
            .format(args.ws, args.ld, args.bt))


def dump_solution(ticc, output_dict, score_dict, soultion_path):
    with open(soultion_path, 'wb') as handle:
        solution_dict = {'ticc': ticc,
                         'output_dict': output_dict,
                         'score_dict': score_dict}
        pickle.dump(solution_dict, handle, protocol=protocol)


def load_solution(soultion_path):
    with open(soultion_path, 'rb') as handle:
        solution_dict = pickle.load(handle)
        ticc = solution_dict['ticc']
        output_dict = solution_dict['output_dict']
        score_dict = solution_dict['score_dict']
        return ticc, output_dict, score_dict
        

def maybe_exist(directory):
    """make sure the existence of given directory"""
    if not os.path.exists(directory):
        os.makedirs(directory)


def strftime(ign_on_time):
    ign_on_time = pd.Timestamp(ign_on_time).to_pydatetime()
    ign_on_time_str = ign_on_time.strftime("%Y%m%d %Hh%Mm%Ss")
    return ign_on_time_str


def trip_length(longitude, latitude):
    # caution: vincenty(dist) function takes a (latitude, longitude) pair!
    xy = np.array([latitude, longitude]).T.reshape(-1, 2)
    
    total_length = 0.
    for start, stop, in zip(xy[:-1], xy[1:]):
        total_length += dist(start, stop).km
    
    return total_length


def adjust_output_list(output_list, ws):
    if ws == 1: 
        return output_list
    elif ws > 1:
        new_output_list = []
        for output in output_list:
            # adjust cluster_assignment and nll_vector lengths
            adjustment = [output[0]] * (ws - 1)
            new_output = np.array(adjustment + list(output))
            new_output_list.append(new_output)
        return new_output_list
    else:
        raise ValueError('Window size must be greater or equal to 1.')