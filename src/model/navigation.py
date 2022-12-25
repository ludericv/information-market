from enum import Enum
from helpers.utils import rotation_matrix
import numpy as np


class Location(Enum):
    FOOD = 1
    NEST = 2


class Target:
    def __init__(self, location):
        self.location = location
        self.relative_distance = np.array([0, 0]).astype('float64')
        self.age = 0
        self.valid = False

    def is_valid(self):
        return self.valid

    def get_age(self):
        return self.age

    def set_age(self, age):
        self.age = age

    def set_valid(self, valid):
        self.valid = valid

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
        self.entries = dict()
        for location in Location:
            self.entries[location] = Target(location)

    def is_information_valid_for_location(self, location):
        return self.entries[location].is_valid()

    def set_information_valid_for_location(self, location, known):
        self.entries[location].valid = known

    def get_relative_position_for_location(self, location):
        return self.entries[location].get_distance()

    def set_relative_position_for_location(self, location, distance):
        self.entries[location].set_distance(distance)

    def update_from_movement(self, dr):
        for location in self.entries:
            self.entries[location].update(dr)

    def rotate_from_angle(self, angle):
        for location in self.entries:
            self.entries[location].rotate(angle)

    def get_age_for_location(self, location):
        return self.entries[location].age

    def set_age_for_location(self, location, age):
        self.entries[location].age = age

    def get_information_entry(self, location):
        return self.entries[location]

    def replace_information_entry(self, location, new_target):
        self.entries[location] = new_target
