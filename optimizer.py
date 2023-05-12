import logging
import pickle
import os

import numpy as np
import numpy.typing as npt
from revolve2.core.modular_robot import ActiveHinge, Body, Brick, ModularRobot
from revolve2.core.physics.running import ActorState, ActorControl, Runner, Batch, Environment, PosedActor
from revolve2.core.physics.environment_actor_controller import EnvironmentActorController
from revolve2.runners.mujoco import LocalRunner
from revolve2.standard_resources import terrains
from pyrr import Vector3, Quaternion
from brain.decentralized_brain import DecentralizedBrain
from state_measures import retrieve_decentralized_info_from_actor

import neat


class DecentralizedNEATOptimizer:
    """
    Evolutionary NEAT optimizer. Does not need to implement much, just a Revolve2 gimmick
    """
    population_bu: neat.Population
    population_td: neat.Population

    _robot_body: Body

    _runner: Runner
    _TERRAIN = terrains.flat()

    _simulation_time: int
    _sampling_frequency: float
    _control_frequency: float

    def __init__(self, robot_body: Body, configBU_path: str, configTD_path: str, population_size: int, simulation_time: int, sampling_frequency: float, control_frequency: float):

        sensory_length = 7 * 2  # joint info + body info

        single_message_length = 32
        # vector message length * maximum estimated number of messages (i.e. number of children of the largest body)
        full_message_length = 30 * single_message_length
        actuator_number = 1

        # finds input, output, population and updates their values
        set_config(sensory_length, single_message_length, population_size, configBU_path)  # TODO: (maybe) check whether bottom-up output size can be evolved as well?
        set_config(full_message_length, actuator_number, population_size, configTD_path)

        configbu = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                               neat.DefaultSpeciesSet, neat.DefaultStagnation,
                               configBU_path)

        configtd = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                               neat.DefaultSpeciesSet, neat.DefaultStagnation,
                               configTD_path)

        # create population and set its reporters
        self.population_bu = neat.Population(configbu)
        self.population_bu.add_reporter(neat.StdOutReporter(True))
        stats = neat.StatisticsReporter()
        self.population_bu.add_reporter(stats)
        self.population_bu.add_reporter(neat.Checkpointer(generation_interval=1, filename_prefix='bu_checkpoint'))  # comment if you want to run from other checkpoint (see self.run())

        # create population and set its reporters
        self.population_td = neat.Population(configtd)
        self.population_td.add_reporter(neat.StdOutReporter(True))
        stats = neat.StatisticsReporter()
        self.population_td.add_reporter(stats)
        self.population_td.add_reporter(neat.Checkpointer(generation_interval=1, filename_prefix='td_checkpoint'))  # comment if you want to run from other checkpoint (see self.run())

        self._robot_body = robot_body

        self._simulation_time = simulation_time
        self._sampling_frequency = sampling_frequency
        self._control_frequency = control_frequency

        self._runner = LocalRunner(headless=True)

    async def run(self, num_generations):

        # Uncomment if you want to run from checkpoint n
        # checkpoint_file = 'neat-checkpoint-n'
        # self.population = neat.Checkpointer.restore_checkpoint(checkpoint_file)

        best = await self.population_bu.run(self.evaluate_generation, num_generations)

        with open("best.pickle", "wb") as f:
            pickle.dump(best, f)

    async def evaluate_generation(self, genomes, config):

        batch = Batch(
            simulation_time=self._simulation_time,
            sampling_frequency=self._sampling_frequency,
            control_frequency=self._control_frequency,
        )

        for _, genotype in genomes:
            _, controller = develop(genotype, self._robot_body, config).make_actor_and_controller()
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

        batch_results = await self._runner.run_batch(batch)  # await

        for environment_result, genome in zip(batch_results.environment_results, genomes):
            idg, genotype = genome
            if genotype.fitness is None:
                genotype.fitness = self._calculate_fitness(
                    environment_result.environment_states[0].actor_states[0],
                    environment_result.environment_states[-1].actor_states[0],
                )

    @staticmethod
    def _calculate_fitness(begin_state: ActorState, end_state: ActorState) -> float:
        # distance traveled on the xy plane
        return float(
            (begin_state.position[0] - end_state.position[0]) ** 2
            + ((begin_state.position[1] - end_state.position[1]) ** 2)
        )


def set_config(input_param: int, output_param: int, population_param: int, path) -> None:
    """
    Sets NEAT's config.txt so that it matches with the robot shape.

    :param input_param: input neurons (position, orientation and rotation of all blocks)
    :param output_param: number of DOF output
    :param population_param: population size for each generation
    :param path: path of config file
    """
    import re

    with open(path, 'r+') as config_file:
        content = config_file.read()

        # Use the replace method to replace the word
        content = re.sub(r'num_inputs {14}= \d+', 'num_inputs              = ' + str(input_param), content)
        content = re.sub(r'num_outputs {13}= \d+', 'num_outputs             = ' + str(output_param), content)
        content = re.sub(r'pop_size {14}= \d+', 'pop_size              = ' + str(population_param), content)

        # Move the file pointer to the beginning of the file
        config_file.seek(0)

        # Write the updated contents back to the file
        config_file.write(content)

        # Truncate the remaining content if any
        config_file.truncate()


def develop(genotype, robot_body, config) -> ModularRobot:

    dof_ranges = np.full(len(robot_body.find_active_hinges()), 1.0)
    brain = DecentralizedBrain(genotype, dof_ranges, config)

    return ModularRobot(robot_body, brain)
