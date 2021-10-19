import copy

from navigation import Location, Target
from utils import distance_between, rotate


class CommunicationSession:
    def __init__(self, client, neighbors):
        self._client = client
        self._neighbors = neighbors

    def get_ages(self, location: Location):
        return [n.get_nav_location_age(location) for n in self._neighbors]

    def are_locations_known(self, location: Location):
        return [n.knows_location(location) for n in self._neighbors]

    def get_target_price(self, neighbor_index: int, location: Location) -> float:
        return self._neighbors[neighbor_index].get_target_price(location)

    def make_transaction(self, neighbor_index, location) -> Target:
        # increment neighbor balance
        # decrease buyer balance
        target = copy.deepcopy(self._neighbors[neighbor_index].get_target(location))
        target.rotate(self._neighbors[neighbor_index].orientation-self._client.orientation)
        return target

    def get_distance_from(self, neighbor_index):
        distance = self._neighbors[neighbor_index].pos - self._client.pos
        return rotate(distance, -self._client.orientation)
