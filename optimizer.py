import math
import asyncio

import numpy as np
from typing import List
from revolve2.core.modular_robot import Body, ModularRobot
from revolve2.core.physics.running import EnvironmentState, Runner, Batch, Environment, PosedActor
from revolve2.core.physics.environment_actor_controller import EnvironmentActorController
from revolve2.runners.mujoco import LocalRunner
from revolve2.standard_resources import terrains
from pyrr import Vector3, Quaternion
from brain.decentralized_brain import DecentralizedBrain
from brain.ModularPolicy import JointPolicy

from evotorch.algorithms import SNES
from evotorch.logging import StdOutLogger, PicklingLogger
from evotorch.neuroevolution import NEProblem
import torch


class DecentralizedOptimizer:
    """
    Evolutionary NEAT optimizer. Does not need to implement much, just a Revolve2 gimmick
    """
    solver: SNES
    Logger: StdOutLogger
    Pickler: PicklingLogger

    _robot_bodies: List[Body]

    _runner: Runner
    _TERRAIN = terrains.flat()

    _simulation_time: int
    _sampling_frequency: float
    _control_frequency: float

    _full_message_length: int
    _single_message_length: int

    def __init__(self, robot_bodies: List[Body],
                 population_size: int, simulation_time: int,
                 sampling_frequency: float, control_frequency: float,
                 sensory_length: int, batch_size: int, single_message_length: int, biggest_body: int):

        policy_args = {'state_dim': sensory_length, 'action_dim': 1,
                       'msg_dim': single_message_length, 'batch_size': batch_size,
                       'max_action': 1, 'max_children': biggest_body}

        torch.set_default_dtype(torch.double)

        problem = NEProblem(objective_sense="max",
                            network=JointPolicy, network_eval_func=self.evaluate_network,
                            network_args=policy_args, initial_bounds=(-1, 1),
                            device="cuda:0" if torch.cuda.is_available() else "cpu")

        self.solver = SNES(problem, popsize=population_size, stdev_init=5)
        self.Logger = StdOutLogger(self.solver)
        # self.Pickler = PicklingLogger(self.solver, interval=1, directory='Checkpoints/')

        self._robot_bodies = robot_bodies

        self._simulation_time = simulation_time
        self._sampling_frequency = sampling_frequency
        self._control_frequency = control_frequency

        self._full_message_length = single_message_length * biggest_body
        self._single_message_length = single_message_length

        self._runner = LocalRunner(headless=True, num_simulators=4)

    def run(self, num_generations):

        self.solver.run(num_generations)

    def evaluate_network(self, network):

        batch = Batch(
            simulation_time=self._simulation_time,
            sampling_frequency=self._sampling_frequency,
            control_frequency=self._control_frequency,
        )

        for body in self._robot_bodies:

            _, controller = self.develop(network, body, self._full_message_length, self._single_message_length).make_actor_and_controller()
            actor = controller.actor
            bounding_box = actor.calc_aabb()
            env = Environment(EnvironmentActorController(controller))
            env.static_geometries.extend(self._TERRAIN.static_geometry)
            env.actors.append(
                PosedActor(
                    actor,
                    Vector3(
                        [
                            0.0,
                            0.0,
                            bounding_box.size.z / 2.0 - bounding_box.offset.z,
                        ]
                    ),
                    Quaternion(),
                    [0.0 for _ in controller.get_dof_targets()],
                )
            )
            batch.environments.append(env)

        # loop = asyncio.get_event_loop()
        # batch_results = loop.run_until_complete(self._runner.run_batch(batch))
        batch_results = self._runner.run_batch(batch)

        fitness_sum = 0
        for environment_result in batch_results.environment_results:
            fitness_sum += self.get_fitness(environment_result.environment_states)

        return fitness_sum / len(self._robot_bodies)

    @staticmethod
    def get_fitness(states: List[EnvironmentState]) -> float:

        return math.sqrt(
            (states[0].actor_states[0].position[0] - states[-1].actor_states[0].position[0]) ** 2
            + ((states[0].actor_states[0].position[1] - states[-1].actor_states[0].position[1]) ** 2))

    @staticmethod
    def develop(network, robot_body, full_message_length, single_message_length) -> ModularRobot:

        dof_ranges = np.full(len(robot_body.find_active_hinges()), 1.0)
        brain = DecentralizedBrain(network, dof_ranges, full_message_length, single_message_length)

        return ModularRobot(robot_body, brain)
