from enum import Enum
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


def rotate(vector, angle):
    rot_mat = rotation_matrix(angle)
    rotated_vector = rot_mat.dot(vector)
    return rotated_vector


def distance_between(robot1, robot2):
    return norm(robot2.pos - robot1.pos)


class InsufficientFundsException(Exception):
    pass


class NoInformationSoldException(Exception):
    pass


class NoLocationSensedException(Exception):
    pass


class CommunicationState(Enum):
    OPEN = 1
    PROCESSING = 2
    ON_COOLDOWN = 3
