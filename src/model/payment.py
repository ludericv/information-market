from abc import ABC, abstractmethod
from collections import Counter, deque

import numpy as np
import pandas as pd

from helpers.utils import InsufficientFundsException
from model.navigation import Location


class Transaction:
    def __init__(self, buyer_id, seller_id, location, info_relative_angle, timestep):
        self.buyer_id = buyer_id
        self.seller_id = seller_id
        self.location = location
        self.relative_angle = info_relative_angle
        self.timestep = timestep


class PaymentSystem(ABC):
    def __init__(self):
        self._creditors = set()

    @abstractmethod
    def get_shares_mapping(self, total_amount):
        pass

    @abstractmethod
    def record_transaction(self, transaction: Transaction):
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
                shares_mapping[creditor_id] += share
        return shares_mapping

    def record_transaction(self, transaction: Transaction):
        self._creditors[transaction.seller_id] += 1


class LastKPaymentSystem(PaymentSystem):

    def __init__(self, k=5):
        super().__init__()
        self._creditors = deque(maxlen=k)

    def get_shares_mapping(self, total_amount):
        if len(self._creditors) == 0:
            return {}
        shares_mapping = {creditor_id: 0 for creditor_id in self._creditors}
        share = total_amount / len(self._creditors)
        for creditor_id in self._creditors:
            shares_mapping[creditor_id] += share
        return shares_mapping

    def record_transaction(self, transaction: Transaction):
        self._creditors.append(transaction.seller_id)


class TimeVaryingSharePaymentSystem(PaymentSystem):
    def __init__(self):
        super().__init__()
        self._creditors = Counter()
        self._creditor_times = {}

    @staticmethod
    def get_share(creditor_time, total_time):
        return total_time - creditor_time

    def get_shares_mapping(self, total_amount):
        total_time = sum(self._creditor_times.values())
        total_shares = sum(self.get_share(self._creditor_times[c], total_time) for c in self._creditor_times)
        shares_mapping = {creditor_id: 0 for creditor_id in self._creditor_times}
        if total_shares > 0:
            for creditor_id in self._creditor_times:
                share = self.get_share(self._creditor_times[creditor_id], total_time) * total_amount / total_shares
                shares_mapping[creditor_id] += share
        return shares_mapping

    def record_transaction(self, transaction: Transaction):
        self._creditors[transaction.seller_id] += 1
        if transaction.seller_id not in self._creditor_times:
            self._creditor_times[transaction.seller_id] = 1

    def step(self):
        for creditor_id in self._creditors:
            self._creditor_times[creditor_id] += self._creditors[creditor_id]

    def reset_creditors(self):
        self._creditors.clear()
        self._creditor_times.clear()


class TransactionPaymentSystem(PaymentSystem):

    def __init__(self):
        super().__init__()
        self.transactions = set()
        self.timestep = 0

    def step(self):
        self.timestep += 1

    def get_shares_mapping(self, total_amount):
        df = pd.DataFrame([[t.seller_id, t.relative_angle, 0] for t in self.transactions],
                          columns=["seller", "angle", "alike"])
        angle_window = 30
        df["alike"] = df.apply(func=lambda row: (((df.iloc[:, 1] - row[1]) % 360) < angle_window).sum()
                                                         + (((row[1] - df.iloc[:, 1]) % 360) < angle_window).sum() - 1,
                                        axis=1)
        sellers_to_alike = df.groupby("seller").sum().to_dict()["alike"]
        # print(df)
        total_shares = sum(sellers_to_alike.values())
        mapping = {seller: total_amount * sellers_to_alike[seller]/total_shares for seller in sellers_to_alike}
        return mapping

    def record_transaction(self, transaction: Transaction):
        self.transactions.add(transaction)

    def reset_creditors(self):
        self.transactions.clear()


class SmallDeviationPaymentSystem(TransactionPaymentSystem):
    def get_shares_mapping(self, total_amount):
        df = pd.DataFrame([[t.seller_id, t.relative_angle, 0] for t in self.transactions],
                          columns=["seller", "angle", "alike"])
        df["alike"] = df.apply(func=lambda row: pd.DataFrame(
            [((df.iloc[:, 1] - row[1]) % 360)**2,
             ((row[1] - df.iloc[:, 1]) % 360)**2]).
                               apply(min, axis=0).
                               sum(),
                               axis=1)
        sellers_to_alike = df.groupby("seller").sum()
        sellers_to_alike["alike"] = sellers_to_alike["alike"].sum() - sellers_to_alike["alike"]
        sellers_to_alike = sellers_to_alike.to_dict()["alike"]
        total_shares = sum(sellers_to_alike.values())
        mapping = {seller: total_amount * sellers_to_alike[seller] / total_shares for seller in sellers_to_alike}
        return mapping


class PaymentDB:
    def __init__(self, population_ids, initial_reward, info_share):
        self.database = {}
        self.info_share = info_share
        for robot_id in population_ids:
            self.database[robot_id] = {"reward": initial_reward,
                                       "payment_system": SmallDeviationPaymentSystem()}

    def step(self):
        for robot_id in self.database:
            self.database[robot_id]["payment_system"].step()

    def pay_reward(self, robot_id, reward=1):
        self.database[robot_id]["reward"] += reward

    def record_transaction(self, transaction: Transaction):
        self.database[transaction.buyer_id]["payment_system"].record_transaction(transaction)

    def pay_creditors(self, debitor_id, total_reward=1):
        mapping = self.database[debitor_id]["payment_system"].get_shares_mapping(self.info_share)
        # print(mapping)
        for creditor_id, share in mapping.items():
            self.database[debitor_id]["reward"] -= share * total_reward
            self.database[creditor_id]["reward"] += share * total_reward
        self.database[debitor_id]["payment_system"].reset_creditors()

    def get_reward(self, robot_id):
        return self.database[robot_id]["reward"]

    def apply_cost(self, robot_id, cost):
        self.database[robot_id]["reward"] -= cost
        if self.database[robot_id]["reward"] < 0:
            self.database[robot_id]["reward"] = 0
            raise InsufficientFundsException
