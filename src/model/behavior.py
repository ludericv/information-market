import copy
from abc import ABC, abstractmethod
from enum import Enum
from math import cos, radians, sin

import numpy as np

from model.communication import CommunicationSession
from model.navigation import Location, NavigationTable
from model.strategy import WeightedAverageAgeStrategy
from helpers.utils import get_orientation_from_vector, norm, InsufficientFundsException, NoInformationSoldException, \
    NoLocationSensedException


class State(Enum):
    EXPLORING = 1
    SEEKING_FOOD = 2
    SEEKING_NEST = 3


def behavior_factory(behavior_params):
    behavior = eval(behavior_params['class'])(**behavior_params['parameters'])
    return behavior


class Behavior(ABC):

    def __init__(self):
        self.color = "blue"
        self.navigation_table = NavigationTable()

    @abstractmethod
    def buy_info(self, neighbors):
        pass

    @abstractmethod
    def step(self, api):
        """Simulates 1 step of behavior (= 1 movement)"""

    @abstractmethod
    def sell_info(self, location):
        pass

    def debug_text(self):
        return ""


class NaiveBehavior(Behavior):
    def __init__(self):
        super().__init__()
        self.state = State.EXPLORING
        self.strategy = WeightedAverageAgeStrategy()
        self.dr = np.array([0, 0]).astype('float64')
        self.id = -1

    def buy_info(self, session: CommunicationSession):
        for location in Location:
            metadata = session.get_metadata(location)
            metadata_sorted_by_age = sorted(metadata.items(), key=lambda item: item[1]["age"])
            for bot_id, data in metadata_sorted_by_age:
                if data["age"] < self.navigation_table.get_age_for_location(location):
                    try:
                        other_target = session.make_transaction(neighbor_id=bot_id, location=location)
                        new_target = self.strategy.combine(self.navigation_table.get_information_entry(location),
                                                           other_target,
                                                           session.get_distance_from(bot_id))
                        self.navigation_table.replace_information_entry(location, new_target)
                        break
                    except InsufficientFundsException:
                        pass
                    except NoInformationSoldException:
                        pass

    def step(self, api):
        self.dr[0], self.dr[1] = 0, 0
        self.id = api.get_id()
        sensors = api.get_sensors()
        self.update_behavior(sensors, api)
        self.update_movement_based_on_state(api)
        self.check_movement_with_sensors(sensors)
        self.update_nav_table_based_on_dr()

    def sell_info(self, location):
        return self.navigation_table.get_information_entry(location)

    def update_behavior(self, sensors, api):
        for location in Location:
            if sensors[location]:
                try:
                    self.navigation_table.set_relative_position_for_location(location,
                                                                             api.get_relative_position_to_location(
                                                                                 location))
                    self.navigation_table.set_information_valid_for_location(location, True)
                    self.navigation_table.set_age_for_location(location, 0)
                except NoLocationSensedException:
                    print(f"Sensors do not sense {location}")

        if self.state == State.EXPLORING:
            if self.navigation_table.is_information_valid_for_location(Location.FOOD) and not api.carries_food():
                self.state = State.SEEKING_FOOD
            if self.navigation_table.is_information_valid_for_location(Location.NEST) and api.carries_food():
                self.state = State.SEEKING_NEST

        elif self.state == State.SEEKING_FOOD:
            if api.carries_food():
                if self.navigation_table.is_information_valid_for_location(Location.NEST):
                    self.state = State.SEEKING_NEST
                else:
                    self.state = State.EXPLORING
            elif norm(self.navigation_table.get_relative_position_for_location(Location.FOOD)) < api.radius():
                self.navigation_table.set_information_valid_for_location(Location.FOOD, False)
                self.state = State.EXPLORING

        elif self.state == State.SEEKING_NEST:
            if not api.carries_food():
                if self.navigation_table.is_information_valid_for_location(Location.FOOD):
                    self.state = State.SEEKING_FOOD
                else:
                    self.state = State.EXPLORING
            elif norm(self.navigation_table.get_relative_position_for_location(Location.NEST)) < api.radius():
                self.navigation_table.set_information_valid_for_location(Location.NEST, False)
                self.state = State.EXPLORING

        if sensors["FRONT"]:
            if self.state == State.SEEKING_NEST:
                self.navigation_table.set_information_valid_for_location(Location.NEST, False)
                self.state = State.EXPLORING
            elif self.state == State.SEEKING_FOOD:
                self.navigation_table.set_information_valid_for_location(Location.FOOD, False)
                self.state = State.EXPLORING

    def update_movement_based_on_state(self, api):
        if self.state == State.SEEKING_FOOD:
            self.dr = self.navigation_table.get_relative_position_for_location(Location.FOOD)
            food_norm = norm(self.navigation_table.get_relative_position_for_location(Location.FOOD))
            if food_norm > api.speed():
                self.dr = self.dr * api.speed() / food_norm

        elif self.state == State.SEEKING_NEST:
            self.dr = self.navigation_table.get_relative_position_for_location(Location.NEST)
            nest_norm = norm(self.navigation_table.get_relative_position_for_location(Location.NEST))
            if nest_norm > api.speed():
                self.dr = self.dr * api.speed() / nest_norm

        else:
            turn_angle = api.get_levi_turn_angle()
            self.dr = api.speed() * np.array([cos(radians(turn_angle)), sin(radians(turn_angle))])

        api.set_desired_movement(self.dr)

    def check_movement_with_sensors(self, sensors):
        if (sensors["FRONT"] and self.dr[0] >= 0) or (sensors["BACK"] and self.dr[0] <= 0):
            self.dr[0] = -self.dr[0]
        if (sensors["RIGHT"] and self.dr[1] <= 0) or (sensors["LEFT"] and self.dr[1] >= 0):
            self.dr[1] = -self.dr[1]

    def update_nav_table_based_on_dr(self):
        self.navigation_table.update_from_movement(self.dr)
        self.navigation_table.rotate_from_angle(-get_orientation_from_vector(self.dr))


class CarefulBehavior(NaiveBehavior):
    def __init__(self, security_level=3):
        super(CarefulBehavior, self).__init__()
        self.color = "deep sky blue"
        self.security_level = security_level
        self.pending_information = {location: {} for location in Location}

    def buy_info(self, session: CommunicationSession):
        for location in Location:
            metadata = session.get_metadata(location)
            metadata_sorted_by_age = sorted(metadata.items(), key=lambda item: item[1]["age"])
            for bot_id, data in metadata_sorted_by_age:
                if data["age"] < self.navigation_table.get_age_for_location(location) and bot_id not in \
                        self.pending_information[
                            location]:
                    try:
                        other_target = session.make_transaction(neighbor_id=bot_id, location=location)
                        other_target.set_distance(other_target.get_distance() + session.get_distance_from(bot_id))
                        if not self.navigation_table.is_information_valid_for_location(location):
                            self.navigation_table.replace_information_entry(location, other_target)
                        else:
                            self.pending_information[location][bot_id] = other_target
                            if len(self.pending_information[location]) >= self.security_level:
                                self.combine_pending_information(location)
                    except InsufficientFundsException:
                        pass
                    except NoInformationSoldException:
                        pass

    def combine_pending_information(self, location):
        distances = [t.get_distance() for t in self.pending_information[location].values()]
        mean_distance = np.mean(distances, axis=0)
        best_target = min(self.pending_information[location].values(),
                          key=lambda t: norm(t.get_distance() - mean_distance))
        self.navigation_table.replace_information_entry(location, best_target)
        self.pending_information[location].clear()

    def step(self, api):
        super().step(api)
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


class ScepticalBehavior(NaiveBehavior):
    def __init__(self, threshold=0.25):
        super(ScepticalBehavior, self).__init__()
        self.pending_information = {location: {} for location in Location}
        self.threshold = threshold

    def buy_info(self, session: CommunicationSession):
        for location in Location:
            metadata = session.get_metadata(location)
            metadata_sorted_by_age = sorted(metadata.items(), key=lambda item: item[1]["age"])
            for bot_id, data in metadata_sorted_by_age:
                if data["age"] < self.navigation_table.get_age_for_location(location) and bot_id not in \
                        self.pending_information[
                            location]:
                    try:
                        other_target = session.make_transaction(neighbor_id=bot_id, location=location)
                        other_target.set_distance(other_target.get_distance() + session.get_distance_from(
                            bot_id))

                        if not self.navigation_table.is_information_valid_for_location(location) or \
                                self.difference_score(
                                    self.navigation_table.get_relative_position_for_location(location),
                                    other_target.get_distance()) < self.threshold:
                            new_target = self.strategy.combine(self.navigation_table.get_information_entry(location),
                                                               other_target,
                                                               np.array([0, 0]))
                            self.navigation_table.replace_information_entry(location, new_target)
                            self.pending_information[location].clear()
                        else:
                            for target in self.pending_information[location].values():
                                if self.difference_score(target.get_distance(),
                                                         other_target.get_distance()) < self.threshold:
                                    new_target = self.strategy.combine(target,
                                                                       other_target,
                                                                       np.array([0, 0]))
                                    self.navigation_table.replace_information_entry(location, new_target)
                                    self.pending_information[location].clear()
                                    break
                            else:
                                self.pending_information[location][bot_id] = other_target
                    except InsufficientFundsException:
                        pass
                    except NoInformationSoldException:
                        pass

    @staticmethod
    def difference_score(current_vector, bought_vector):
        v_norm = norm(current_vector)
        score = norm(current_vector - bought_vector) / v_norm if v_norm > 0 else 1000
        return score

    def step(self, api):
        super().step(api)
        self.update_pending_information()

    def update_pending_information(self):
        for location in Location:
            for target in self.pending_information[location].values():
                target.update(self.dr)
                target.rotate(-get_orientation_from_vector(self.dr))


class SaboteurBehavior(NaiveBehavior):
    def __init__(self, rotation_angle=90):
        super().__init__()
        self.color = "red"
        self.rotation_angle = rotation_angle

    def sell_info(self, location):
        t = copy.deepcopy(self.navigation_table.get_information_entry(location))
        t.rotate(self.rotation_angle)
        return t


class GreedyBehavior(NaiveBehavior):
    def __init__(self):
        super().__init__()
        self.color = "green"

    def sell_info(self, location):
        t = copy.deepcopy(self.navigation_table.get_information_entry(location))
        t.age = 1
        return t


class FreeRiderBehavior(ScepticalBehavior):
    def __init__(self):
        super().__init__()
        self.color = "pink"

    def sell_info(self, location):
        return None


class ScaboteurBehavior(ScepticalBehavior):
    def __init__(self, rotation_angle=90, threshold=0.25):
        super().__init__()
        self.color = "red"
        self.rotation_angle = rotation_angle
        self.threshold = threshold

    def sell_info(self, location):
        t = copy.deepcopy(self.navigation_table.get_information_entry(location))
        t.rotate(self.rotation_angle)
        return t


class ScepticalGreedyBehavior(ScepticalBehavior):
    def __init__(self):
        super().__init__()
        self.color = "green"

    def sell_info(self, location):
        t = copy.deepcopy(self.navigation_table.get_information_entry(location))
        t.age = 1
        return t
