from typing import List, Tuple
import numpy as np
import numpy.typing as npt
import torch

from revolve2.actor_controller import ActorController
from revolve2.core.modular_robot import ActiveHinge, Body, Brain
from revolve2.core.physics.actor import Actor
from brain.reinforcement_controller import ReinforcementController
from brain.reinforcement_network import ReinforcementLearner


class ReinforcementBrain(Brain):
    """
    Centralized (distributed) brain that controls every limb through a NN whose weights are evolved through GA
    """

    _weight_matrix: npt.NDArray[np.float_]
    _dof_ranges: npt.NDArray[np.float_]
    _num_input_neurons: int

    def __init__(self, weight_matrix: npt.NDArray[np.float_], dof_ranges: npt.NDArray[np.float_], num_input_neurons: int):
        self._weight_matrix = weight_matrix
        self._dof_ranges = dof_ranges
        self._num_input_neurons = num_input_neurons

    def make_controller(self, body: Body, dof_ids: List[int]) -> ActorController:
        """
        Create a controller for the provided body.

        :param body: The body to make the brain for.
        :param dof_ids: Map from actor joint index to module id.
        :returns: The created controller.
        """

        actor, _ = body.to_actor()

        torch.set_default_dtype(torch.double)

        model = ReinforcementLearner(self._num_input_neurons, len(self._dof_ranges))

        model.make_weights(self._weight_matrix)

        return ReinforcementController(
            self._dof_ranges, model, actor
        )
