import numpy as np


def exponential_model(max_price, supply, demand):
    return max_price ** (1 - supply / demand)


def logistics_model(max_price, supply, demand):
    # k = steepness assumed to be 1
    k = 0.1
    x0 = (max_price-1)/k
    x = demand - supply
    return max_price/(1 + np.exp(k*(x0-x)))


class Market:
    max_history = 1000

    def __init__(self, demand, max_price):
        self.demand = demand
        self.max_price = max_price
        self.supply_during_current_step = 0
        self.history_index = 0
        self.supply_history = np.zeros(self.max_history)
        self._price = self.compute_price()

    def compute_price(self):
        average_supply = np.sum(self.supply_history)
        return logistics_model(self.max_price, average_supply, self.demand)

    def sell_strawberry(self):
        self.supply_during_current_step += 1
        print(f"Supply = {np.sum(self.supply_history)}")
        return self._price

    def step(self):
        self.supply_history[self.history_index] = self.supply_during_current_step
        self.supply_during_current_step = 0
        self.history_index = (self.history_index + 1) % self.max_history
        self._price = self.compute_price()
