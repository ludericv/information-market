from abc import ABC, abstractmethod
from collections import Counter


class PaymentSystem(ABC):
    def __init__(self):
        self._creditors = set()

    @abstractmethod
    def pay_creditors(self, total_amount):
        pass

    @abstractmethod
    def add_creditor(self, creditor):
        pass

    def step(self):
        pass

    def reset_creditors(self):
        self._creditors.clear()


class FixedSharePaymentSystem(PaymentSystem):
    def __init__(self):
        super().__init__()
        self._creditors = Counter()

    def pay_creditors(self, total_amount):
        nb_of_creditors = sum(self._creditors.values())
        if nb_of_creditors > 0:
            share = total_amount / nb_of_creditors
            for creditor in self._creditors.elements():
                creditor.modify_reward(share)
        self.reset_creditors()

    def add_creditor(self, creditor):
        self._creditors[creditor] += 1


class TimeVaryingSharePaymentSystem(PaymentSystem):
    def __init__(self):
        super().__init__()
        self._creditors = Counter()
        self._creditor_times = {}

    @staticmethod
    def get_share(creditor_time, total_time):
        return total_time-creditor_time

    def pay_creditors(self, total_amount):
        total_time = sum(self._creditor_times.values())
        total_shares = sum(self.get_share(self._creditor_times[c], total_time) for c in self._creditor_times)
        for creditor in self._creditor_times:
            share = self.get_share(self._creditor_times[creditor], total_time) * total_amount / total_shares
            creditor.modify_reward(share)

    def add_creditor(self, creditor):
        self._creditors[creditor] += 1
        if creditor not in self._creditor_times:
            self._creditor_times[creditor] = 1

    def step(self):
        for creditor in self._creditors:
            self._creditor_times[creditor] += self._creditors[creditor]

    def reset_creditors(self):
        self._creditors.clear()
        self._creditor_times.clear()


