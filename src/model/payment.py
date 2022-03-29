from abc import ABC, abstractmethod
from collections import Counter

from helpers.utils import InsufficientFundsException


class PaymentSystem(ABC):
    def __init__(self):
        self._creditors = set()

    @abstractmethod
    def get_shares_mapping(self, total_amount):
        pass

    @abstractmethod
    def add_creditor(self, creditor_id):
        pass

    def step(self):
        pass

    def reset_creditors(self):
        self._creditors.clear()


class FixedSharePaymentSystem(PaymentSystem):
    def __init__(self):
        super().__init__()
        self._creditors = Counter()

    def get_shares_mapping(self, total_amount):
        nb_of_creditors = sum(self._creditors.values())
        shares_mapping = {creditor_id: 0 for creditor_id in self._creditors}
        if nb_of_creditors > 0:
            share = total_amount / nb_of_creditors
            for creditor_id in self._creditors.elements():
                shares_mapping[creditor_id] += share * self._creditors[creditor_id]
        return shares_mapping

    def add_creditor(self, creditor_id):
        self._creditors[creditor_id] += 1


class TimeVaryingSharePaymentSystem(PaymentSystem):
    def __init__(self):
        super().__init__()
        self._creditors = Counter()
        self._creditor_times = {}

    @staticmethod
    def get_share(creditor_time, total_time):
        return total_time-creditor_time

    def get_shares_mapping(self, total_amount):
        total_time = sum(self._creditor_times.values())
        total_shares = sum(self.get_share(self._creditor_times[c], total_time) for c in self._creditor_times)
        shares_mapping = {creditor_id: 0 for creditor_id in self._creditor_times}
        if total_shares > 0:
            for creditor_id in self._creditor_times:
                share = self.get_share(self._creditor_times[creditor_id], total_time) * total_amount / total_shares
                shares_mapping[creditor_id] += share
        return shares_mapping

    def add_creditor(self, creditor_id):
        self._creditors[creditor_id] += 1
        if creditor_id not in self._creditor_times:
            self._creditor_times[creditor_id] = 1

    def step(self):
        for creditor_id in self._creditors:
            self._creditor_times[creditor_id] += self._creditors[creditor_id]

    def reset_creditors(self):
        self._creditors.clear()
        self._creditor_times.clear()


class PaymentDB:
    def __init__(self, population_ids, initial_reward, info_share):
        self.database = {}
        self.info_share = info_share
        for robot_id in population_ids:
            self.database[robot_id] = {"reward": initial_reward,
                                       "payment_system": FixedSharePaymentSystem()}

    def step(self):
        for robot_id in self.database:
            self.database[robot_id]["payment_system"].step()

    def pay_reward(self, robot_id, reward=1):
        self.database[robot_id]["reward"] += reward

    def add_creditor(self, debitor_id, creditor_id):
        self.database[debitor_id]["payment_system"].add_creditor(creditor_id)

    def pay_creditors(self, debitor_id, total_reward=1):
        mapping = self.database[debitor_id]["payment_system"].get_shares_mapping(self.info_share)
        for creditor_id, share in mapping.items():
            self.database[debitor_id]["reward"] -= share
            self.database[creditor_id]["reward"] += share
        self.database[debitor_id]["payment_system"].reset_creditors()

    def get_reward(self, robot_id):
        return self.database[robot_id]["reward"]

    def apply_cost(self, robot_id, cost):
        self.database[robot_id]["reward"] -= cost
        if self.database[robot_id]["reward"] < 0:
            self.database[robot_id]["reward"] = 0
            raise InsufficientFundsException





