from collections import Counter

import numpy as np


def exponential_model(max_price, supply, demand):
    return max_price ** (1 - supply / demand)


def logistics_model(max_price, supply, demand):
    if max_price == 1:
        return 1
    # k = steepness assumed to be 0.1
    k = 0.1
    x0 = np.log(max_price - 1) / k
    x = demand - supply
    return max_price / (1 + np.exp(k * (x0 - x)))


def market_factory(market_params):
    return eval(market_params['class'])(**market_params['parameters'])


class Market:
    max_history = 1000

    def __init__(self, demand, max_price):
        self.demand = demand * self.max_history
        self.max_price = max_price
        self.supply_during_current_step = 0
        self.history_index = 0
        self.supply_history = np.zeros(self.max_history)
        self._price = self.compute_price()

    def compute_price(self):
        average_supply = np.sum(self.supply_history)
        return logistics_model(self.max_price, average_supply, self.demand)

    def sell_strawberry(self, robot_id):
        self.supply_during_current_step += 1
        # print(f"Supply = {np.sum(self.supply_history)}, demand = {self.demand}")
        return self._price

    def get_supply(self):
        return np.sum(self.supply_history)

    def step(self):
        self.supply_history[self.history_index] = self.supply_during_current_step
        self.supply_during_current_step = 0
        self.history_index = (self.history_index + 1) % self.max_history
        self._price = self.compute_price()


class RoundTripPriceMarket:
    def __init__(self, min_time, max_price):
        self.demand = 10
        self.robot_times = Counter()
        self.min_time = min_time
        self.max_price = max_price

    def sell_strawberry(self, robot_id):
        if robot_id not in self.robot_times:
            self.robot_times[robot_id] = 0
            return self.max_price

        price = self.max_price * np.exp((self.min_time - self.robot_times[robot_id]) / self.min_time)
        # print(f"{price} for {self.robot_times[robot_id]} (min of {self.min_time})")
        self.robot_times[robot_id] = 0
        return price

    def step(self):
        for bot_id in self.robot_times:
            self.robot_times[bot_id] += 1

    def get_supply(self):
        return 0


class FixedPriceMarket:
    def __init__(self, reward):
        self.demand = 10
        self.reward = reward

    def sell_strawberry(self, robot_id):
        return self.reward

    def step(self):
        pass

    def get_supply(self):
        return 0
