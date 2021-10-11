from enum import Enum
from utils import rotation_matrix
import numpy as np


class Location(Enum):
    FOOD = 1
    NEST = 2


class Target:
    def __init__(self, location, quality):
        self.location = location
        self.relative_distance = np.array([0, 0]).astype('float64')
        self.age = 0
        self.quality = quality
        self.decaying_quality = 1
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

    def decay_quality(self, decay_rate):
        self.decaying_quality *= decay_rate
        # res = self.decaying_quality - decay_rate
        # self.decaying_quality = res if res > 0 else 0


class NavigationTable:
    def __init__(self, quality):
        self.targets = dict()
        for location in Location:
            self.targets[location] = Target(location, quality)

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

    def get_target(self, location):
        return self.targets[location]

    def replace_target(self, location, new_target):
        self.targets[location] = new_target

    def reset_quality(self, location, quality):
        self.targets[location].quality = quality
        self.targets[location].decaying_quality = 1

    def decay_qualities(self, decay_rate):
        for location in self.targets:
            self.targets[location].decay_quality(decay_rate)


