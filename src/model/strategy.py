import copy
from abc import ABC, abstractmethod

from model.navigation import Target


class InformationStrategy(ABC):
    @abstractmethod
    def should_combine(self, my_target: Target, other_target: Target):
        """Whether a robot's target information should be replaced/combined with another"""

    @abstractmethod
    def combine(self, my_target: Target, other_target: Target, bots_distance) -> Target:
        """Combine a robot's target information with other information"""


class BetterAgeStrategy(InformationStrategy):
    def should_combine(self, my_target: Target, other_target: Target):
        """Whether a robot's target information should be replaced/combined with another"""
        return my_target.age > other_target.age

    def combine(self, my_target: Target, other_target: Target, bots_distance) -> Target:
        """Combine a robot's target information with other information"""
        new_target = copy.deepcopy(other_target)
        new_target.set_distance(new_target.get_distance() + bots_distance)
        return new_target


class WeightedAverageAgeStrategy(InformationStrategy):
    def should_combine(self, my_target: Target, other_target: Target):
        """Whether a robot's target information should be replaced/combined with another"""
        if other_target.valid and my_target.age > other_target.age:
            return True
        return False

    def combine(self, my_target: Target, other_target: Target, bots_distance) -> Target:
        """Combine a robot's target information with other information"""
        new_target = copy.deepcopy(other_target)
        ages_sum = my_target.age + other_target.age
        new_distance = (my_target.age / ages_sum) * (other_target.get_distance() + bots_distance) + (
                other_target.age / ages_sum) * my_target.get_distance()  # older = lower weight
        if not my_target.is_valid():
            new_distance = other_target.get_distance() + bots_distance
        new_target.set_distance(new_distance)
        new_target.age = ages_sum // 2
        return new_target


class DecayingQualityStrategy(InformationStrategy):

    def should_combine(self, my_target: Target, other_target: Target):
        """Whether a robot's target information should be replaced/combined with another"""
        if other_target.is_valid() and other_target.decaying_quality > my_target.decaying_quality:
            return True
        return False

    def combine(self, my_target: Target, other_target: Target, bots_distance) -> Target:
        """Combine a robot's target information with other information"""
        new_target = copy.deepcopy(other_target)
        new_target.set_distance(new_target.get_distance() + bots_distance)
        return new_target


class WeightedDecayingQualityStrategy(DecayingQualityStrategy):
    def combine(self, my_target: Target, other_target: Target, bots_distance) -> Target:
        """Combine a robot's target information with other information"""
        new_target = copy.deepcopy(other_target)
        qualities_sum = my_target.decaying_quality + other_target.decaying_quality
        new_distance = (my_target.decaying_quality / qualities_sum) * my_target.get_distance() + (
                other_target.decaying_quality / qualities_sum) * (other_target.get_distance() + bots_distance)
        if not my_target.is_valid():
            new_distance = other_target.get_distance() + bots_distance
        new_target.set_distance(new_distance)
        new_target.decaying_quality = other_target.decaying_quality
        return new_target
