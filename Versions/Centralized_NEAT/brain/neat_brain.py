from typing import List, Tuple
import numpy as np
import numpy.typing as npt
import neat

from revolve2.actor_controller import ActorController
from revolve2.core.modular_robot import ActiveHinge, Body, Brain
from revolve2.core.physics.actor import Actor
from brain.neat_controller import NEATController


class NEATBrain(Brain):
    """
    Centralized (distributed) brain that controls every limb through a NN whose weights are evolved through GA
    """

    _genotype: neat.genome
    _dof_ranges: npt.NDArray[np.float_]
    _config: neat.Config

    def __init__(self, genotype: neat.genome, dof_ranges: npt.NDArray[np.float_], config: neat.Config):
        self._genotype = genotype
        self._dof_ranges = dof_ranges
        self._config = config

    def make_controller(self, body: Body, dof_ids: List[int]) -> ActorController:
        """
        Create a controller for the provided body.

        :param body: The body to make the brain for.
        :param dof_ids: Map from actor joint index to module id.
        :returns: The created controller.
        """

        actor, _ = body.to_actor()

        model = neat.ctrnn.CTRNN.create(self._genotype, self._config, 1/60)

        return NEATController(
            self._dof_ranges, model, actor
        )
