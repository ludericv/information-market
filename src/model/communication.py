import copy

from model.navigation import Location, Target
from helpers.utils import rotate, CommunicationState, NoInformationSoldException, get_orientation_from_vector
from model.payment import Transaction


class CommunicationSession:
    def __init__(self, client, neighbors, info_cost):
        self._client = client
        self._neighbors = {n.id: n for n in neighbors if n.comm_state == CommunicationState.OPEN}
        self._info_cost = info_cost

    def get_metadata(self, location):
        metadata = {n_id: {
            "age": n.get_target_from_behavior(location).get_age(),
            "known": n.get_target_from_behavior(location).is_known()
        } for n_id, n in self._neighbors.items() if
            n.get_target_from_behavior(location) is not None}
        return metadata

    def make_transaction(self, neighbor_id, location) -> Target:
        target = copy.deepcopy(self._neighbors[neighbor_id].get_target_from_behavior(location))
        if target is None:
            raise NoInformationSoldException
        target.rotate(self._neighbors[neighbor_id].orientation - self._client.orientation)
        transaction = Transaction(self._client.id,
                                  neighbor_id,
                                  location,
                                  get_orientation_from_vector(target.get_distance()),
                                  None)
        self._client.record_transaction(transaction)
        self._client.communication_happened()
        self._neighbors[neighbor_id].communication_happened()
        return target

    def get_distance_from(self, neighbor_id):
        distance = self._neighbors[neighbor_id].pos - self._client.pos
        return rotate(distance, -self._client.orientation)

    def get_own_reward(self):
        return self._client.reward()
