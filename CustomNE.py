from typing import Callable, List, Optional, Union

import torch
from torch import nn
import numpy as np
from pyrr import Vector3, Quaternion

from evotorch import SolutionBatch
from evotorch.core import BoundsPairLike, ObjectiveSense
from evotorch.tools.misc import Device
from evotorch.neuroevolution import NEProblem
from revolve2.core.modular_robot import Body, ModularRobot
from revolve2.core.physics.running import EnvironmentState, Runner, Batch, Environment, PosedActor
from revolve2.core.physics.environment_actor_controller import EnvironmentActorController
from revolve2.runners.mujoco import LocalRunner
from standard_resources import terrains
from brain.decentralized_brain import DecentralizedBrain


class CustomNE(NEProblem):

    def __init__(self,
                 bodies: List[Body],
                 num_neighbors: int,
                 state_length: int,
                 num_simulators: int,
                 objective_sense: ObjectiveSense,
                 network: Union[str, nn.Module, Callable[[], nn.Module]],
                 network_args: Optional[dict] = None,
                 simulation_time: int = 10,
                 sampling_frequency: float = 60,
                 control_frequency: float = 60,
                 initial_bounds: Optional[BoundsPairLike] = (-0.00001, 0.00001),
                 num_actors: Optional[Union[int, str]] = None,
                 actor_config: Optional[dict] = None,
                 num_gpus_per_actor: Optional[Union[int, float, str]] = None,
                 num_subbatches: Optional[int] = None,
                 subbatch_size: Optional[int] = None,
                 device: Optional[Device] = None,
                 ):
        super().__init__(objective_sense=objective_sense,
                         network=network,
                         network_args=network_args,
                         initial_bounds=initial_bounds,
                         num_actors=num_actors,
                         actor_config=actor_config,
                         subbatch_size=subbatch_size,
                         device=device,
                         num_subbatches=num_subbatches,
                         num_gpus_per_actor=num_gpus_per_actor)

        self._robot_bodies = bodies
        self._simulation_time = simulation_time
        self._sampling_frequency = sampling_frequency
        self._control_frequency = control_frequency
        self._num_neighbors = num_neighbors
        self._state_length = state_length
        self._TERRAIN = terrains.flat()
        self.num_simulators = num_simulators
        self._runner = LocalRunner(headless=True, num_simulators=num_simulators)

    def _evaluate_batch(self, solutions: SolutionBatch):
        """N.B. Both evotorch and revolve define a concept of batch: their use is similar but not the same"""

        networks = [self.make_net(solution) for solution in solutions.values]
        fitnesses = torch.full((len(solutions),), 99999.0)

        batch = Batch(
            simulation_time=self._simulation_time,
            sampling_frequency=self._sampling_frequency,
            control_frequency=self._control_frequency,
            simulation_timestep=0.001
        )
        for body in self._robot_bodies:
            for i, network in enumerate(networks):

                _, controller = self.develop(network, body, self._num_neighbors,
                                             self._state_length).make_actor_and_controller()
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

                if (i + 1) % self.num_simulators == 0:

                    # batch_results = asyncio.get_event_loop().run_until_complete(self._runner.run_batch(batch))
                    batch_results = self._runner.run_batch(batch)

                    for j, environment_result in enumerate(batch_results.environment_results):
                        index = i - (self.num_simulators - j) + 1
                        fitness = self.get_fitness(environment_result.environment_states)
                        if fitnesses[index] > fitness:
                            fitnesses[index] = fitness

                        fitnesses[index] = self.get_fitness(environment_result.environment_states)

                    batch = Batch(
                        simulation_time=self._simulation_time,
                        sampling_frequency=self._sampling_frequency,
                        control_frequency=self._control_frequency,
                        simulation_timestep=0.001
                    )
            if len(batch.environments) > 0:
                batch_results = self._runner.run_batch(batch)

                for j, environment_result in enumerate(batch_results.environment_results):
                    fitness = self.get_fitness(environment_result.environment_states)
                    if fitnesses[-j - 1] > fitness:
                        fitnesses[-j - 1] = fitness

        solutions.set_evals(fitnesses)

    @staticmethod
    def get_fitness(states: List[EnvironmentState]) -> float:
        """
        Fitness function.
        The fitness is the distance traveled minus the sum of squared actions (to penalize large movements)"""

        actions = 0
        asymmetry = 0
        num_limbs = len(states[0].actor_states[0].dof_state)
        for i in range(1, len(states), 2):

            action = (np.square(states[i - 1].actor_states[0].dof_state - states[i].actor_states[0].dof_state).sum()) / num_limbs

            asymmetry += np.std(np.square(states[i - 1].actor_states[0].dof_state)) + np.std(np.square(states[i].actor_states[0].dof_state))

            if action == 0:
                action = 0.5  # Penalize no movement

            actions += action

        distance = 3 * ((states[0].actor_states[0].position[0] - states[-1].actor_states[0].position[0]) ** 2)
                   # + ((states[0].actor_states[0].position[1] - states[-1].actor_states[0].position[1]) ** 2)

        return distance - actions

    @staticmethod
    def develop(network, robot_body, num_neighbors, state_length) -> ModularRobot:

        dof_ranges = np.full(len(robot_body.find_active_hinges()), 1.0)
        brain = DecentralizedBrain(network, dof_ranges, num_neighbors, state_length)

        return ModularRobot(robot_body, brain)


'''    def _evaluate_batch(self, network: nn.Module) -> Union[float, torch.Tensor, tuple]:

        batch = Batch(
            simulation_time=self._simulation_time,
            sampling_frequency=self._sampling_frequency,
            control_frequency=self._control_frequency,
        )

        for body in self._robot_bodies:
            _, controller = self.develop(network, body, self._full_message_length,
                                         self._single_message_length).make_actor_and_controller()
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

        # batch_results = asyncio.get_event_loop().run_until_complete(self._runner.run_batch(batch))
        batch_results = self._runner.run_batch(batch)

        fitness_sum = 0
        for environment_result in batch_results.environment_results:
            fitness_sum += self.get_fitness(environment_result.environment_states)

        return fitness_sum / len(self._robot_bodies)'''
