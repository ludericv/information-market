import copy

from navigation import Location, Target
from utils import distance_between, rotate


class CommunicationSession:
    def __init__(self, client, neighbors, info_cost):
        self._client = client
        self._neighbors = neighbors
        self._info_cost = info_cost

    def get_ages(self, location: Location):
        return [n.get_nav_location_age(location) for n in self._neighbors]

    def are_locations_known(self, location: Location):
        return [n.knows_location(location) for n in self._neighbors]

    def get_target_price(self, neighbor_index: int, location: Location) -> float:
        return self._info_cost

    def make_transaction(self, neighbor_index, location) -> Target:
        price = self.get_target_price(neighbor_index, location)
        self._client.modify_reward(-price)
        self._neighbors[neighbor_index].modify_reward(price)
        target = copy.deepcopy(self._neighbors[neighbor_index].get_target_from_behavior(location))
        target.rotate(self._neighbors[neighbor_index].orientation-self._client.orientation)
        return target

    def get_distance_from(self, neighbor_index):
        distance = self._neighbors[neighbor_index].pos - self._client.pos
        return rotate(distance, -self._client.orientation)

    def get_own_reward(self):
        return self._client.reward()
