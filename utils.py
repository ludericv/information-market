from math import atan2, pi, radians

import numpy as np


def norm(vector):
    return (sum(x ** 2 for x in vector)) ** 0.5


def get_orientation_from_vector(vector):
    angle = atan2(vector[1], vector[0])
    return (360 * angle / (2 * pi)) % 360


def rotation_matrix(angle):
    theta = radians(angle)
    return np.array(((np.cos(theta), -np.sin(theta)),
                     (np.sin(theta), np.cos(theta))))