import copy
from abc import ABC, abstractmethod
from enum import Enum
from math import cos, radians, sin

import numpy as np

from src.model.communication import CommunicationSession
from src.model.navigation import Location, NavigationTable
from model.strategy import WeightedAverageAgeStrategy
from helpers.utils import get_orientation_from_vector, norm, InsufficientFundsException


class State(Enum):
    EXPLORING = 1
    SEEKING_FOOD = 2
    SEEKING_NEST = 3


class Behavior(ABC):

    def __init__(self):
        self.dr = np.array([0, 0]).astype('float64')
        self.navigation_table = NavigationTable(quality=1)
        self.color = "blue"

    @abstractmethod
    def communicate(self, neighbors):
        pass

    @abstractmethod
    def step(self, sensors, api):
        """Simulates 1 step of behavior (= 1 movement)"""

    def get_dr(self):
        return self.dr

    def get_target(self, location):
        return self.navigation_table.get_target(location)

    def debug_text(self):
        return ""


class HonestBehavior(Behavior):

    def __init__(self):
        super().__init__()
        self.state = State.EXPLORING
        self.strategy = WeightedAverageAgeStrategy()

    def communicate(self, session: CommunicationSession):
        for location in Location:
            metadata = session.get_metadata(location)
            metadata_sorted_by_age = sorted(metadata.items(), key=lambda item: item[1]["age"])
            for bot_id, data in metadata_sorted_by_age:
                if data["age"] < self.navigation_table.get_age(location) - 10:
                    try:
                        other_target = session.make_transaction(neighbor_id=bot_id, location=location)
                        new_target = self.strategy.combine(self.navigation_table.get_target(location),
                                                           other_target,
                                                           session.get_distance_from(bot_id))
                        self.navigation_table.replace_target(location, new_target)
                        break
                    except InsufficientFundsException:
                        pass

    def step(self, sensors, api):
        self.dr[0], self.dr[1] = 0, 0
        self.update_behavior(sensors, api)
        self.update_movement_based_on_state(api)
        self.check_movement_with_sensors(sensors, api)
        self.update_nav_table_based_on_dr(api)

    def update_behavior(self, sensors, api):
        for location in Location:
            if sensors[location]:
                try:
                    self.navigation_table.set_location_vector(location, api.get_vector(location))
                    self.navigation_table.set_location_known(location, True)
                    self.navigation_table.set_location_age(location, 0)
                    self.navigation_table.reset_quality(location, 1)
                except:
                    print(f"Sensors do not sense {location}")

        if self.state == State.EXPLORING:
            if self.navigation_table.is_location_known(Location.FOOD) and not api.carries_food():
                self.state = State.SEEKING_FOOD
            if self.navigation_table.is_location_known(Location.NEST) and api.carries_food():
                self.state = State.SEEKING_NEST

        elif self.state == State.SEEKING_FOOD:
            if api.carries_food():
                if self.navigation_table.is_location_known(Location.NEST):
                    self.state = State.SEEKING_NEST
                else:
                    self.state = State.EXPLORING
            elif norm(self.navigation_table.get_location_vector(Location.FOOD)) < api.radius():
                self.navigation_table.set_location_known(Location.FOOD, False)
                self.state = State.EXPLORING

        elif self.state == State.SEEKING_NEST:
            if not api.carries_food():
                if self.navigation_table.is_location_known(Location.FOOD):
                    self.state = State.SEEKING_FOOD
                else:
                    self.state = State.EXPLORING
            elif norm(self.navigation_table.get_location_vector(Location.NEST)) < api.radius():
                self.navigation_table.set_location_known(Location.NEST, False)
                self.state = State.EXPLORING

        if sensors["FRONT"]:
            if self.state == State.SEEKING_NEST:
                self.navigation_table.set_location_known(Location.NEST, False)
                self.state = State.EXPLORING
            elif self.state == State.SEEKING_FOOD:
                self.navigation_table.set_location_known(Location.FOOD, False)
                self.state = State.EXPLORING

    def update_movement_based_on_state(self, api):
        if self.state == State.SEEKING_FOOD:
            self.dr = self.navigation_table.get_location_vector(Location.FOOD)
            food_norm = norm(self.navigation_table.get_location_vector(Location.FOOD))
            if food_norm > api.speed():
                self.dr = self.dr * api.speed() / food_norm

        elif self.state == State.SEEKING_NEST:
            self.dr = self.navigation_table.get_location_vector(Location.NEST)
            nest_norm = norm(self.navigation_table.get_location_vector(Location.NEST))
            if nest_norm > api.speed():
                self.dr = self.dr * api.speed() / nest_norm

        else:
            turn_angle = api.get_levi_turn_angle()
            self.dr = api.speed() * np.array([cos(radians(turn_angle)), sin(radians(turn_angle))])

    def check_movement_with_sensors(self, sensors, api):
        if (sensors["FRONT"] and self.dr[0] >= 0) or (sensors["BACK"] and self.dr[0] <= 0):
            self.dr[0] = -self.dr[0]
        if (sensors["RIGHT"] and self.dr[1] <= 0) or (sensors["LEFT"] and self.dr[1] >= 0):
            self.dr[1] = -self.dr[1]

    def update_nav_table_based_on_dr(self, api):
        self.navigation_table.update_from_movement(self.dr)
        self.navigation_table.rotate_from_angle(-get_orientation_from_vector(self.dr))
        self.navigation_table.decay_qualities(1-0.01*abs(api.get_mu))


class CarefulBehavior(HonestBehavior):
    def __init__(self, security_level=3):
        super(CarefulBehavior, self).__init__()
        self.color = "deep sky blue"
        self.security_level = security_level
        self.pending_information = {location: {} for location in Location}

    def communicate(self, session: CommunicationSession):
        for location in Location:
            metadata = session.get_metadata(location)
            metadata_sorted_by_age = sorted(metadata.items(), key=lambda item: item[1]["age"])
            for bot_id, data in metadata_sorted_by_age:
                if data["age"] < self.navigation_table.get_age(location) and bot_id not in self.pending_information[location]:
                    try:
                        other_target = session.make_transaction(neighbor_id=bot_id, location=location)
                        other_target.set_distance(other_target.get_distance() + session.get_distance_from(bot_id))
                        if not self.navigation_table.is_location_known(location):
                            self.navigation_table.replace_target(location, other_target)
                        else:
                            self.pending_information[location][bot_id] = other_target
                            if len(self.pending_information[location]) >= self.security_level:
                                self.combine_pending_information(location)
                    except InsufficientFundsException:
                        pass

    def combine_pending_information(self, location):
        distances = [t.get_distance() for t in self.pending_information[location].values()]
        mean_distance = np.mean(distances, axis=0)
        best_target = min(self.pending_information[location].values(), key=lambda t: norm(t.get_distance()-mean_distance))
        self.navigation_table.replace_target(location, best_target)
        self.pending_information[location].clear()

    def step(self, sensors, api):
        super().step(sensors, api)
        self.update_pending_information()

    def update_pending_information(self):
        for location in Location:
            for target in self.pending_information[location].values():
                target.update(self.dr)
                target.rotate(-get_orientation_from_vector(self.dr))

    def debug_text(self):
        return f"size of pending: {[len(self.pending_information[l]) for l in Location]}\n" \
               f"{self.pending_information[Location.FOOD]}\n" \
               f"{self.pending_information[Location.NEST]}"


class SmartBehavior(HonestBehavior):
    def __init__(self, threshold=0.1):
        super(SmartBehavior, self).__init__()
        self.pending_information = {location: {} for location in Location}
        self.threshold = threshold

    def communicate(self, session: CommunicationSession):
        for location in Location:
            metadata = session.get_metadata(location)
            metadata_sorted_by_age = sorted(metadata.items(), key=lambda item: item[1]["age"])
            for bot_id, data in metadata_sorted_by_age:
                if data["age"] < self.navigation_table.get_age(location) and bot_id not in self.pending_information[location]:
                    try:
                        other_target = session.make_transaction(neighbor_id=bot_id, location=location)
                        other_target.set_distance(other_target.get_distance() + session.get_distance_from(bot_id))  # TODO: refactor strategy and communication session so everything in own reference frame
                        if not self.navigation_table.is_location_known(location) or \
                                self.difference_score(self.navigation_table.get_location_vector(location), other_target.get_distance()) < self.threshold:
                            self.navigation_table.replace_target(location, other_target)
                            self.pending_information[location].clear()
                        else:
                            for target in self.pending_information[location].values():
                                if self.difference_score(target.get_distance(), other_target.get_distance()) < self.threshold:
                                    self.navigation_table.replace_target(location, other_target)
                                    self.pending_information[location].clear()
                                    break
                            else:
                                self.pending_information[location][bot_id] = other_target
                    except InsufficientFundsException:
                        pass

    @staticmethod
    def difference_score(current_vector, bought_vector):
        return norm(current_vector-bought_vector)/norm(current_vector)

    def step(self, sensors, api):
        super().step(sensors, api)
        self.update_pending_information()

    def update_pending_information(self):
        for location in Location:
            for target in self.pending_information[location].values():
                target.update(self.dr)
                target.rotate(-get_orientation_from_vector(self.dr))


class SaboteurBehavior(HonestBehavior):
    def __init__(self):
        super().__init__()
        self.color = "red"

    def get_target(self, location):
        t = copy.deepcopy(self.navigation_table.get_target(location))
        t.rotate(90)
        return t


class GreedyBehavior(HonestBehavior):
    def __init__(self):
        super().__init__()
        self.color = "green"

    def get_target(self, location):
        t = copy.deepcopy(self.navigation_table.get_target(location))
        t.age = 1
        return t