
from math import sin, cos, sqrt, atan2, radians
from geopy.distance import vincenty as dist

import numpy as np

# def dist(xy1, xy2):
#     # https://stackoverflow.com/questions/19412462/
#     # getting-distance-between-two-points-based-on-latitude-longitude

#     # approximate radius of earth in km
#     R = 6373.0

#     lat1, lon1 = xy1
#     lat2, lon2 = xy2

#     dlon = lon2 - lon1
#     dlat = lat2 - lat1

#     a = sin(dlat / 2)**2 + cos(lat1) * cos(lat2) * sin(dlon / 2)**2
#     c = 2 * atan2(sqrt(a), sqrt(1 - a))

#     distance = R * c
#     return distance


def trip_length(longitude, latitude):
    xy = np.array([latitude, longitude]).T.reshape(-1, 2)
    
    total_length = 0.
    for start, stop, in zip(xy[:-1], xy[1:]):
        # vincenty(dist) function get (latitude, longitude) pair!
        total_length += dist(start, stop).km
    
    return total_length