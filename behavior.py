from abc import ABC, abstractmethod
from math import cos, radians, sin
from random import choices

import numpy as np

from agent import State
from navigation import Location, NavigationTable
from utils import get_orientation_from_vector, norm


class Behavior(ABC):

    def __init__(self, body):
        self.body = body
        self.dr = np.array([0, 0]).astype('float64')
        self.state = State.SEEKING_FOOD
        self.navigation_table = NavigationTable(quality=1)

    @abstractmethod
    def communicate(self, neighbors):
        pass

    @abstractmethod
    def step(self, sensors):
        """Simulates 1 step of behavior (= 1 movement)"""

    def get_dr(self):
        return self.dr


class HonestBehavior(Behavior):

    def __init__(self, body):
        super().__init__(body)

    def communicate(self, neighbors):
        pass

    def step(self, sensors):
        self.dr[0], self.dr[1] = 0, 0
        self.update_behavior(sensors)
        self.update_movement_based_on_state()
        self.check_movement_with_sensors(sensors)

    def update_behavior(self, sensors):
        sensing_food = sensors["FOOD"]
        sensing_nest = sensors["NEST"]
        if sensing_food:
            self.body.set_food_vector()
            self.navigation_table.set_location_known(Location.FOOD, True)
            self.navigation_table.set_location_age(Location.FOOD, 0)
            self.navigation_table.reset_quality(Location.FOOD, 1)
        if sensing_nest:
            self.body.set_nest_vector()
            self.navigation_table.set_location_known(Location.NEST, True)
            self.navigation_table.set_location_age(Location.NEST, 0)
            self.navigation_table.reset_quality(Location.NEST, 1)

        if self.state == State.EXPLORING:
            if self.navigation_table.is_location_known(Location.FOOD) and not self.body.carries_food:
                self.state = State.SEEKING_FOOD
            if self.navigation_table.is_location_known(Location.NEST) and self.body.carries_food:
                self.state = State.SEEKING_NEST

        elif self.state == State.SEEKING_FOOD and sensing_food:
            if self.navigation_table.is_location_known(Location.NEST):
                self.state = State.SEEKING_NEST
            else:
                self.state = State.EXPLORING

        elif self.state == State.SEEKING_NEST and sensing_nest:
            if self.navigation_table.is_location_known(Location.FOOD):
                self.state = State.SEEKING_FOOD
            else:
                self.state = State.EXPLORING

        if sensors["FRONT"]:
            if self.state == State.SEEKING_NEST:
                self.navigation_table.set_location_known(Location.NEST, False)
                self.state = State.EXPLORING
            elif self.state == State.SEEKING_FOOD:
                self.navigation_table.set_location_known(Location.FOOD, False)
                self.state = State.EXPLORING

    def update_movement_based_on_state(self):
        if self.state == State.EXPLORING:
            turn_angle = self.body.get_levi_turn_angle()
            self.dr = self.body.speed * np.array([cos(radians(turn_angle)), sin(radians(turn_angle))])
        elif self.state == State.SEEKING_FOOD:
            self.dr = self.body.speed * self.navigation_table.get_location_vector(Location.FOOD) / norm(self.navigation_table.get_location_vector(Location.FOOD))
        elif self.state == State.SEEKING_NEST:
            self.dr = self.body.speed * self.navigation_table.get_location_vector(Location.NEST) / norm(self.navigation_table.get_location_vector(Location.NEST))

    def check_movement_with_sensors(self, sensors):
        if (sensors["FRONT"] and self.dr[1] > 0) or (sensors["BACK"] and self.dr[1] < 0):
            self.dr[1] = -self.dr[1]
        if (sensors["RIGHT"] and self.dr[0] > 0) (sensors["LEFT"] and self.dr[0] < 0):
            self.dr[0] = -self.dr[0]



