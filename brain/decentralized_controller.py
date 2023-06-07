from __future__ import annotations

import math

from state_measures import *
from brain.ModularPolicy import JointPolicy

from typing import List
import torch
import numpy as np
import numpy.typing as npt
from revolve2.actor_controller import ActorController
from revolve2.core.physics.actor import Actor, Joint
from revolve2.serialization import SerializeError, StaticData


class DecentralizedController(ActorController):
    """A controller that manipulates all limbs given a weight matrix evolved through GA."""

    _dof_ranges: npt.NDArray[np.float_]
    _dof_ids: List[int]
    _models: List[List[Joint, JointPolicy]]
    _actor: Actor
    _target: npt.NDArray[np.float_]
    _full_message_length: int
    _single_message_length: int

    def __init__(
        self,
        dof_ranges: npt.NDArray[np.float_], dof_ids: List[int],
        models: List[List[Joint, JointPolicy]],
        actor: Actor,
        full_length: int,
        single_length: int
    ) -> None:
        """
        Initialize this object.

        :param models: Neural network instances.
        :param dof_ranges: Maximum range (half the complete range) of the output of degrees of freedom.
        """

        self._dof_ranges = dof_ranges
        self._dof_ids = dof_ids
        self._models = models
        self._actor = actor

        self._target = np.full(len(self._dof_ranges), 0.5 * math.pi / 2.0)
        self._full_message_length = full_length
        self._single_message_length = single_length

    def map_output(self, unordered_targets: List[float]) -> None:
        """Maps the arbitrary output of the down step to their corresponding dof_id index"""
        for i, target in enumerate(unordered_targets):
            self._target[self._dof_ids[i]] = target

    def up_step(self, dt: float) -> (torch.Tensor, int):

        full_message = torch.tensor([0.0 for _ in range(self._full_message_length)])

        filled = 0
        for module, network in reversed(self._models):

            input_data = retrieve_extended_joint_info(module)
            _, modular_message = network(torch.tensor(input_data), True)

            full_message[filled:filled + self._single_message_length] = modular_message
            filled += self._single_message_length

        return full_message, filled

    def down_step(self, message: torch.Tensor, filled_index: int):

        output = []
        for module, network in self._models:
            dof, new_message = network(message, False)
            message[filled_index-self._single_message_length:filled_index] = new_message

            filled_index -= self._single_message_length

            output.append(dof[0])

        return output

    def step(self, dt: float) -> None:
        """
        Step the controller dt seconds forward.

        :param dt: The number of seconds to step forward.
        """
        # Total message (without DOF output) is retrieved from Bottom-Up modules
        message, filled_index = self.up_step(dt)

        message = message.to(dtype=torch.double)
        unordered_targets = self.down_step(message, filled_index)

        self.map_output(unordered_targets)

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
