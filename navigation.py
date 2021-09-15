from enum import Enum
from utils import rotation_matrix
import numpy as np


class Location(Enum):
    FOOD = 1
    NEST = 2


class Target:
    def __init__(self, location):
        self.location = location
        self.relative_distance = np.array([0, 0]).astype('float64')
        self.age = 0
        self.known = False

    def is_known(self):
        return self.known

    def set_known(self, known):
        self.known = known

    def get_distance(self):
        return self.relative_distance

    def set_distance(self, distance):
        self.relative_distance = distance

    def update(self, dr):
        self.age += 1
        self.relative_distance -= dr

    def rotate(self, angle):
        rot_mat = rotation_matrix(angle)
        self.relative_distance = rot_mat.dot(self.relative_distance)


class NavigationTable:
    def __init__(self):
        self.targets = dict()
        for location in Location:
            self.targets[location] = Target(location)

    def replace_location_information(self, location, relative_distance, age, known):
        self.targets[location].location = location
        self.targets[location].set_distance(relative_distance)
        self.targets[location].age = age
        self.targets[location].set_known(known)

    def get_location_information(self, location):
        self.targets[location].get_distance(), self.targets[location].age, self.targets[location].is_known()

    def is_location_known(self, location):
        return self.targets[location].is_known()

    def set_location_known(self, location, known):
        self.targets[location].known = known

    def get_location_vector(self, location):
        return self.targets[location].get_distance()

    def set_location_vector(self, location, distance):
        self.targets[location].set_distance(distance)

    def update_from_movement(self, dr):
        for location in self.targets:
            self.targets[location].update(dr)

    def rotate_from_angle(self, angle):
        for location in self.targets:
            self.targets[location].rotate(angle)

    def get_age(self, location):
        return self.targets[location].age

    def set_location_age(self, location, age):
        self.targets[location].age = age
