import copy
from abc import ABC, abstractmethod
from enum import Enum
from math import cos, radians, sin
from random import choices

import numpy as np

from navigation import Location, NavigationTable
from strategy import BetterAgeStrategy
from utils import get_orientation_from_vector, norm


class State(Enum):
    EXPLORING = 1
    SEEKING_FOOD = 2
    SEEKING_NEST = 3


class Behavior(ABC):

    def __init__(self, body):
        self.body = body
        self.dr = np.array([0, 0]).astype('float64')
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
        self.state = State.EXPLORING
        self.new_information = copy.deepcopy(self.navigation_table)
        self.strategy = BetterAgeStrategy()

    def communicate(self, neighbors):
        pass
        # self.new_information = copy.deepcopy(self.navigation_table)
        #
        # for neighbor in neighbors:
        #     for location in Location:
        #         if self.strategy.should_combine(self.new_information.get_target(location),
        #                                         neighbor.get_nav_target(location)):
        #             new_target = self.strategy.combine(self.new_information.get_target(location),
        #                                                neighbor.get_nav_target(location), neighbor.pos - self.body.pos)
        #             self.new_information.replace_target(location, new_target)

    def step(self, sensors):
        self.navigation_table = self.new_information
        self.dr[0], self.dr[1] = 0, 0
        self.update_behavior(sensors)
        self.update_movement_based_on_state()
        self.check_movement_with_sensors(sensors)

    def update_behavior(self, sensors):
        if sensors["FOOD"]:
            self.body.set_food_vector()
            self.navigation_table.set_location_known(Location.FOOD, True)
            self.navigation_table.set_location_age(Location.FOOD, 0)
            self.navigation_table.reset_quality(Location.FOOD, 1)
        if sensors["NEST"]:
            self.body.set_nest_vector()
            self.navigation_table.set_location_known(Location.NEST, True)
            self.navigation_table.set_location_age(Location.NEST, 0)
            self.navigation_table.reset_quality(Location.NEST, 1)

        if self.state == State.EXPLORING:
            if self.navigation_table.is_location_known(Location.FOOD) and not self.body.carries_food:
                self.state = State.SEEKING_FOOD
            if self.navigation_table.is_location_known(Location.NEST) and self.body.carries_food:
                self.state = State.SEEKING_NEST

        elif self.state == State.SEEKING_FOOD:
            if sensors["FOOD"]:
                if self.navigation_table.is_location_known(Location.NEST):
                    self.state = State.SEEKING_NEST
                else:
                    self.state = State.EXPLORING
            elif norm(self.navigation_table.get_location_vector(Location.FOOD)) < self.body.speed:
                self.state = State.EXPLORING

        elif self.state == State.SEEKING_NEST:
            if sensors["NEST"]:
                if self.navigation_table.is_location_known(Location.FOOD):
                    self.state = State.SEEKING_FOOD
                else:
                    self.state = State.EXPLORING
            elif norm(self.navigation_table.get_location_vector(Location.NEST)) < self.body.speed:
                self.state = State.EXPLORING

        if sensors["FRONT"]:
            if self.state == State.SEEKING_NEST:
                self.navigation_table.set_location_known(Location.NEST, False)
                self.state = State.EXPLORING
            elif self.state == State.SEEKING_FOOD:
                self.navigation_table.set_location_known(Location.FOOD, False)
                self.state = State.EXPLORING

    def update_movement_based_on_state(self):
        if self.state == State.SEEKING_FOOD:
            self.dr = self.body.speed * self.navigation_table.get_location_vector(Location.FOOD)
            food_norm = norm(self.navigation_table.get_location_vector(Location.FOOD))
            if food_norm > self.body.speed:
                self.dr = self.dr/food_norm

        elif self.state == State.SEEKING_NEST:
            self.dr = self.body.speed * self.navigation_table.get_location_vector(Location.NEST)
            nest_norm = norm(self.navigation_table.get_location_vector(Location.NEST))
            if nest_norm > self.body.speed:
                self.dr = self.dr/nest_norm

        else:
            turn_angle = self.body.get_levi_turn_angle()
            self.dr = self.body.speed * np.array([cos(radians(turn_angle)), sin(radians(turn_angle))])

    def check_movement_with_sensors(self, sensors):
        if (sensors["FRONT"] and self.dr[1] > 0) or (sensors["BACK"] and self.dr[1] < 0):
            self.dr[1] = -self.dr[1]
        if (sensors["RIGHT"] and self.dr[0] > 0) or (sensors["LEFT"] and self.dr[0] < 0):
            self.dr[0] = -self.dr[0]
