
from geopy.distance import vincenty as dist

import numpy as np


def trip_length(longitude, latitude):
    # vincenty(dist) function get (latitude, longitude) pair!
    xy = np.array([latitude, longitude]).T.reshape(-1, 2)
    
    total_length = 0.
    for start, stop, in zip(xy[:-1], xy[1:]):
        total_length += dist(start, stop).km
    
    return total_length