from typing import List, Tuple
import numpy as np
import numpy.typing as npt
import neat
import math

from revolve2.actor_controller import ActorController
from revolve2.core.modular_robot import Body, Brain
from brain.decentralized_controller import DecentralizedController


class DecentralizedBrain(Brain):
    """
    Decentralized brain that controls each limb through two modules with weights and topology defined by NEAT algorithm
    """

    _genotype_bu: neat.genome
    _genotype_td: neat.genome
    _dof_ranges: npt.NDArray[np.float_]
    _sensory_length: int
    _single_message_length: int
    _full_message_length: int
    _config_bu: neat.Config
    _config_td: neat.Config

    def __init__(self, genotype_bu: neat.genome, genotype_td: neat.genome, dof_ranges: npt.NDArray[np.float_], sensory_length: int, single_message_length: int, full_message_length: int, config_bu: neat.Config, config_td: neat.Config):
        self._genotype_bu = genotype_bu
        self._genotype_td = genotype_td
        self._dof_ranges = dof_ranges
        self._sensory_length = sensory_length
        self._single_message_length = single_message_length
        self._full_message_length = full_message_length
        self._config_bu = config_bu
        self._config_td = config_td

    def make_controller(self, body: Body, dof_ids: List[int]) -> ActorController:
        """
        Create a controller for the provided body.

        :param body: The body to make the brain for.
        :param dof_ids: Map from actor joint index to module id.
        :returns: The created controller.
        """

        actor, _ = body.to_actor()

        # Add the origin body as the root module
        models = [(actor.bodies[0], neat.nn.recurrent.RecurrentNetwork.create(self._genotype_bu, self._config_bu),
                   neat.nn.recurrent.RecurrentNetwork.create(self._genotype_td, self._config_td))]
        for joint in actor.joints:

            models.append((joint, neat.nn.recurrent.RecurrentNetwork.create(self._genotype_bu, self._config_bu),
                           neat.nn.recurrent.RecurrentNetwork.create(self._genotype_td, self._config_td)))

        dof_ids = remove_brick_ids(dof_ids)

        return DecentralizedController(
            self._dof_ranges, dof_ids, self._sensory_length, self._single_message_length, self._full_message_length, models, actor
        )


def remove_brick_ids(dof_ids):

    # Create a dictionary to map each unique integer to its corresponding smallest possible value
    d = {val: i for i, val in enumerate(sorted(set(dof_ids)))}

    # Iterate through the list and replace each integer with its corresponding smallest possible value from the dictionary
    new_ids = [d[val] for val in dof_ids]

    return new_ids
