import copy

from model.navigation import Location, Target
from helpers.utils import rotate


class CommunicationSession:
    def __init__(self, client, neighbors, info_cost):
        self._client = client
        self._neighbors = {n.id: n for n in neighbors}
        self._info_cost = info_cost

    def get_ages(self, location: Location):
        return [n.get_target_from_behavior(location).get_age() for n in self._neighbors]

    def are_locations_known(self, location: Location):
        return [n.get_target_from_behavior(location).is_known() for n in self._neighbors]

    def get_metadata(self, location):
        metadata = {n_id: {
            "age": n.get_target_from_behavior(location).get_age(),
            "known": n.get_target_from_behavior(location).is_known()
        } for n_id, n in self._neighbors.items()}
        return metadata

    def make_transaction(self, neighbor_id, location) -> Target:
        self._client.add_creditor(neighbor_id)
        target = copy.deepcopy(self._neighbors[neighbor_id].get_target_from_behavior(location))
        target.rotate(self._neighbors[neighbor_id].orientation-self._client.orientation)
        return target

    def get_distance_from(self, neighbor_id):
        distance = self._neighbors[neighbor_id].pos - self._client.pos
        return rotate(distance, -self._client.orientation)

    def get_own_reward(self):
        return self._client.reward()
