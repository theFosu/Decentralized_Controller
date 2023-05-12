from __future__ import annotations

import math

from brain.reinforcement_network import ReinforcementLearner
from state_measures import *

import numpy as np
import numpy.typing as npt
import torch
from revolve2.actor_controller import ActorController
from revolve2.core.physics.actor import Actor
from revolve2.serialization import SerializeError, StaticData


class ReinforcementController(ActorController):
    """A controller that manipulates all limbs given a weight matrix evolved through GA."""

    _dof_ranges: npt.NDArray[np.float_]
    _model: ReinforcementLearner
    _actor: Actor
    _target: npt.NDArray[np.float_]

    def __init__(
        self,
        dof_ranges: npt.NDArray[np.float_],
        model: ReinforcementLearner,
        actor: Actor
    ) -> None:
        """
        Initialize this object.

        :param model: Neural network class.
        :param dof_ranges: Maximum range (half the complete range) of the output of degrees of freedom.
        """

        self._dof_ranges = dof_ranges
        self._model = model  # Already "trained" model
        self._actor = actor

        self._target = np.full(model.noutput, 0.5 * math.pi / 2.0)

    def step(self, dt: float) -> None:
        """
        Step the controller dt seconds forward.

        :param dt: The number of seconds to step forward.
        """

        input_data = retrieve_joint_info_from_actor(self._actor) + retrieve_body_info_from_actor(self._actor) + [dt]
        input_tensor = torch.tensor(input_data)

        target_torch = self._model(input_tensor)

        self._target = target_torch.detach().numpy()

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

    def serialize(self) -> StaticData:
        """
        Serialize this object.

        :returns: The serialized object.
        """

        return {
            "dof_ranges": self._dof_ranges.tolist(),
            "input_neurons": self._target.tolist(),
            "weight_array": self._model.get_weights().tolist()
        }

    @classmethod
    def deserialize(cls, data: StaticData) -> ReinforcementController:
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

        network = ReinforcementLearner(data["input_neurons"], len(data["dof_ranges"]))
        network.make_weights(data["weight_array"])

        return ReinforcementController(
            np.array(data["dof_ranges"]),
            network,
            cls._actor,
        )
