from enum import Enum
import numpy as np


class Location(Enum):
    FOOD = 1
    NEST = 2


class Target:
    def __init__(self, location):
        self.location = location
        self.relative_distance = np.array([0, 0]).astype('float64')
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
        self.relative_distance += dr


class NavigationTable:
    def __init__(self):
        self.targets = dict()
        for location in Location:
            self.targets[location] = Target(location)

    def is_location_known(self, location):
        return self.targets[location].is_known()

    def set_location_known(self, location, known):
        self.targets[location].known = known

    def get_location_vector(self, location):
        return self.targets[location].get_distance()

    def set_location_vector(self, location, distance):
        self.targets[location].set_distance(distance)

    def update_from_movement(self, dr):
        for target in self.targets:
            target.update(dr)
