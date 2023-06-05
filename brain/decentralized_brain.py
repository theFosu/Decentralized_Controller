from typing import List
import numpy as np
import numpy.typing as npt
import copy

from revolve2.actor_controller import ActorController
from revolve2.core.modular_robot import Body, Brain
from brain.decentralized_controller import DecentralizedController
from brain.ModularPolicy import JointPolicy


class DecentralizedBrain(Brain):
    """
    Decentralized brain that controls each limb through two modules with weights and topology defined by NEAT algorithm
    """

    _policy: JointPolicy
    _dof_ranges: npt.NDArray[np.float_]
    _full_message_length: int
    _single_message_length: int

    def __init__(self, network: JointPolicy, dof_ranges: npt.NDArray[np.float_], full_message_length, single_message_length):
        self._policy = network
        self._dof_ranges = dof_ranges
        self.full_message_length = full_message_length
        self.single_message_length = single_message_length

    def make_controller(self, body: Body, dof_ids: List[int]) -> ActorController:
        """
        Create a controller for the provided body.

        :param body: The body to make the brain for.
        :param dof_ids: Map from actor joint index to module id.
        :returns: The created controller.
        """

        actor, _ = body.to_actor()

        dof_ids = remove_brick_ids(dof_ids)

        models = []
        for joint in actor.joints:
            models.append([joint, copy.deepcopy(self._policy)])

        return DecentralizedController(
            self._dof_ranges, dof_ids, models, actor,
            self.full_message_length, self.single_message_length
        )


def remove_brick_ids(dof_ids):

    # Create a dictionary to map each unique integer to its corresponding smallest possible value
    d = {val: i for i, val in enumerate(sorted(set(dof_ids)))}

    # Iterate through the list and replace each integer with its corresponding smallest possible value from the dictionary
    new_ids = [d[val] for val in dof_ids]

    return new_ids
