from __future__ import annotations

import math

from state_measures import *

from typing import List
import numpy as np
import numpy.typing as npt
import neat
import neat.ctrnn as ctrnn
from revolve2.actor_controller import ActorController
from revolve2.core.physics.actor import Actor
from revolve2.serialization import SerializeError, StaticData


class DecentralizedController(ActorController):
    """A controller that manipulates all limbs given a weight matrix evolved through GA."""

    _dof_ranges: npt.NDArray[np.float_]
    _models: List[ctrnn.CTRNN]
    _actor: Actor
    _target: npt.NDArray[np.float_]

    def __init__(
        self,
        dof_ranges: npt.NDArray[np.float_],
        models: List[ctrnn.CTRNN],
        actor: Actor
    ) -> None:
        """
        Initialize this object.

        :param models: Neural network instances.
        :param dof_ranges: Maximum range (half the complete range) of the output of degrees of freedom.
        """

        self._dof_ranges = dof_ranges
        self._models = models
        self._actor = actor

        self._target = np.full(len(self._dof_ranges), 0.5 * math.pi / 2.0)

    def step(self, dt: float) -> None:
        """
        Step the controller dt seconds forward.

        :param dt: The number of seconds to step forward.
        """
        input_data = retrieve_decentralized_info_from_actor(self._actor)

        for i, single_target in enumerate(self._target):

            single_target = self._models[i].advance(input_data[i], dt, dt)

            self._target[i] = single_target[0]

    def get_dof_targets(self) -> List[float]:
        """
        Get the goal position of the limb.

        :returns: The dof targets.
        """

        return list(
            np.clip(
                self._target,
                a_min=-self._dof_ranges,
                a_max=self._dof_ranges,
            )
        )

    @property
    def actor(self):
        return self._actor

    # TODO: these do not work
    def serialize(self) -> StaticData:
        """
        Serialize this object.

        :returns: The serialized object.
        """

        return {
            "dof_ranges": self._dof_ranges.tolist(),
            "input_neurons": self._target.tolist(),
        }

    @classmethod
    def deserialize(cls, data: StaticData) -> DecentralizedController:
        """
        Deserialize an instance of this class from `StaticData`.

        :param data: The data to deserialize from.
        :returns: The deserialized instance.
        :raises SerializeError: If this object cannot be deserialized from the given data.
        """
        if (
            not type(data) == dict
            or not "input_neurons" in data
            or not type(data["input_neurons"]) == list
            or not all(type(s) == float for s in data["input_neurons"])
            or not "weight_array" in data
            or not type(data["weight_array"]) == list
            or not all(type(r) == float for r in data["weight_array"])
            or not "dof_ranges" in data
            or not type(data["dof_ranges"]) == list
            or not all(type(r) == float for r in data["dof_ranges"])
        ):
            raise SerializeError()

        network = DecentralizedController(data["input_neurons"], len(data["dof_ranges"]))

        return DecentralizedController(
            np.array(data["dof_ranges"]),
            network,
            cls._actor,
        )
