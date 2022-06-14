from abc import ABC, abstractmethod
from collections import Counter, deque

import numpy as np

from helpers.utils import InsufficientFundsException
from model.navigation import Location
import pandas as pd
pd.options.mode.chained_assignment = None


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
    def get_shares_mapping(self, total_amount, database=None, debitor_id=None):
        pass

    @abstractmethod
    def record_transaction(self, transaction: Transaction, database):
        pass

    def step(self):
        pass

    def reset_creditors(self):
        self._creditors.clear()


class FixedSharePaymentSystem(PaymentSystem):
    def __init__(self):
        super().__init__()
        self._creditors = Counter()

    def get_shares_mapping(self, total_amount, database=None, debitor_id=None):
        nb_of_creditors = sum(self._creditors.values())
        shares_mapping = {creditor_id: 0 for creditor_id in self._creditors}
        if nb_of_creditors > 0:
            share = total_amount / nb_of_creditors
            for creditor_id in self._creditors.elements():
                shares_mapping[creditor_id] += share
        return shares_mapping

    def record_transaction(self, transaction: Transaction, database):
        self._creditors[transaction.seller_id] += 1


class LastKPaymentSystem(PaymentSystem):

    def __init__(self, k=5):
        super().__init__()
        self._creditors = deque(maxlen=k)

    def get_shares_mapping(self, total_amount, database=None, debitor_id=None):
        if len(self._creditors) == 0:
            return {}
        shares_mapping = {creditor_id: 0 for creditor_id in self._creditors}
        share = total_amount / len(self._creditors)
        for creditor_id in self._creditors:
            shares_mapping[creditor_id] += share
        return shares_mapping

    def record_transaction(self, transaction: Transaction, database):
        self._creditors.append(transaction.seller_id)


class TimeVaryingSharePaymentSystem(PaymentSystem):
    def __init__(self):
        super().__init__()
        self._creditors = Counter()
        self._creditor_times = {}

    @staticmethod
    def get_share(creditor_time, total_time):
        return total_time - creditor_time

    def get_shares_mapping(self, total_amount, database=None, debitor_id=None):
        total_time = sum(self._creditor_times.values())
        total_shares = sum(self.get_share(self._creditor_times[c], total_time) for c in self._creditor_times)
        shares_mapping = {creditor_id: 0 for creditor_id in self._creditor_times}
        if total_shares > 0:
            for creditor_id in self._creditor_times:
                share = self.get_share(self._creditor_times[creditor_id], total_time) * total_amount / total_shares
                shares_mapping[creditor_id] += share
        return shares_mapping

    def record_transaction(self, transaction: Transaction, database):
        self._creditors[transaction.seller_id] += 1
        if transaction.seller_id not in self._creditor_times:
            self._creditor_times[transaction.seller_id] = 1

    def step(self):
        for creditor_id in self._creditors:
            self._creditor_times[creditor_id] += self._creditors[creditor_id]

    def reset_creditors(self):
        self._creditors.clear()
        self._creditor_times.clear()


class WindowTransactionPaymentSystem(PaymentSystem):

    def __init__(self):
        super().__init__()
        self.transactions = set()
        self.pot_amount = 0
        self.timestep = 0

    def step(self):
        self.timestep += 1

    def get_shares_mapping(self, total_amount, database=None, debitor_id=None):
        database.pay_reward(debitor_id, self.pot_amount)
        if len(self.transactions) == 0:
            return {}
        df_all = pd.DataFrame([[t.seller_id, t.relative_angle, t.location, 0] for t in self.transactions],
                          columns=["seller", "angle", "location", "alike"])
        angle_window = 30
        final_mapping = {}
        for location in Location:
            df = df_all[df_all["location"] == location]
            if df.shape[0] == 0:
                continue
            df["alike"] = df.apply(func=lambda row: (((df.iloc[:, 1] - row[1]) % 360) < angle_window).sum()
                                                             + (((row[1] - df.iloc[:, 1]) % 360) < angle_window).sum() - 1,
                                            axis=1)
            sellers_to_alike = df.groupby("seller").sum().to_dict()["alike"]
            mapping = {seller: sellers_to_alike[seller] for seller in sellers_to_alike}
            for seller in mapping:
                if seller in final_mapping:
                    final_mapping[seller] += mapping[seller]
                else:
                    final_mapping[seller] = mapping[seller]

        total_shares = sum(final_mapping.values())
        for seller in final_mapping:
            final_mapping[seller] = final_mapping[seller] * (total_amount + self.pot_amount) / total_shares
        return final_mapping

    def record_transaction(self, transaction: Transaction, database):
        vouch_amount = (1 / 25) * 0.5
        database.apply_cost(transaction.seller_id, vouch_amount)
        self.pot_amount += vouch_amount
        self.transactions.add(transaction)

    def reset_creditors(self):
        self.transactions.clear()
        self.pot_amount = 0


class NumpyWindowTransactionPaymentSystem(PaymentSystem):

    def __init__(self):
        super().__init__()
        self.transactions = set()
        self.pot_amount = 0
        self.timestep = 0

    def step(self):
        self.timestep += 1

    def get_shares_mapping(self, total_amount, database=None, debitor_id=None):
        database.pay_reward(debitor_id, self.pot_amount)
        if len(self.transactions) == 0:
            return {}
        df_all = np.array([[t.seller_id, t.relative_angle, t.location, 0] for t in self.transactions])
        angle_window = 30
        final_mapping = {}
        for location in Location:
            df = df_all[df_all[:, 2] == location]
            if df.shape[0] == 0:
                continue
            df[:, 3] = np.apply_along_axis(func1d=lambda row: (((df[:, 1] - row[1]) % 360) < angle_window).sum()
                                                             + (((row[1] - df[:, 1]) % 360) < angle_window).sum() - 1,
                                            axis=1, arr=df)
            mapping = {seller: 0 for seller in df[:, 0]}
            for row in df:
                mapping[row[0]] += row[3]

            for seller in mapping:
                if seller in final_mapping:
                    final_mapping[seller] += mapping[seller]
                else:
                    final_mapping[seller] = mapping[seller]

        total_shares = sum(final_mapping.values())
        for seller in final_mapping:
            final_mapping[seller] = final_mapping[seller] * (total_amount + self.pot_amount) / total_shares
        return final_mapping

    def record_transaction(self, transaction: Transaction, database):
        vouch_amount = 1 / 25
        database.apply_cost(transaction.seller_id, vouch_amount)
        self.pot_amount += vouch_amount
        self.transactions.add(transaction)

    def reset_creditors(self):
        self.transactions.clear()
        self.pot_amount = 0


class DeltaTimePaymentSystem(WindowTransactionPaymentSystem):
    def get_shares_mapping(self, total_amount, database=None, debitor_id=None):
        sorted_transactions = sorted([[t.seller_id, t.timestep] for t in self.transactions],
                                     key=lambda elem: elem[1])
        sorted_transactions.append([-1, self.timestep])
        df = pd.DataFrame(sorted_transactions, columns=["seller", "time"])
        df["dt"] = df["time"].diff().shift(-1) + 1
        df = df[:-1]
        df["score"] = 1 / df["dt"]
        df["share"] = total_amount * df["score"] / df["score"].sum()
        mapping = df.groupby("seller").sum().to_dict()["share"]
        return mapping


class SmallDeviationPaymentSystem(WindowTransactionPaymentSystem):
    def get_shares_mapping(self, total_amount, database=None, debitor_id=None):
        if len(self.transactions) <= 1:
            return {t.seller_id: total_amount for t in self.transactions}
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
        if total_shares == 0:
            return {seller: total_amount/len(sellers_to_alike) for seller in sellers_to_alike}
        mapping = {seller: total_amount * sellers_to_alike[seller] / total_shares for seller in sellers_to_alike}
        return mapping


class PaymentDB:
    def __init__(self, population_ids, initial_reward, info_share):
        self.database = {}
        self.info_share = info_share
        for robot_id in population_ids:
            self.database[robot_id] = {"reward": initial_reward,
                                       "payment_system": NumpyWindowTransactionPaymentSystem()}

    def step(self):
        for robot_id in self.database:
            self.database[robot_id]["payment_system"].step()

    def pay_reward(self, robot_id, reward=1):
        self.database[robot_id]["reward"] += reward

    def record_transaction(self, transaction: Transaction):
        self.database[transaction.buyer_id]["payment_system"].record_transaction(transaction, self)

    def pay_creditors(self, debitor_id, total_reward=1):
        mapping = self.database[debitor_id]["payment_system"].get_shares_mapping(self.info_share * total_reward, self, debitor_id)
        for creditor_id, share in mapping.items():
            self.database[debitor_id]["reward"] -= share
            self.database[creditor_id]["reward"] += share
        self.database[debitor_id]["payment_system"].reset_creditors()

    def get_reward(self, robot_id):
        return self.database[robot_id]["reward"]

    def apply_cost(self, robot_id, cost):
        if self.database[robot_id]["reward"] < cost:
            raise InsufficientFundsException
        else:
            self.database[robot_id]["reward"] -= cost
