import math
import pickle

import numpy as np
from typing import List
from revolve2.core.modular_robot import Body, ModularRobot
from revolve2.core.physics.running import EnvironmentState, Runner, Batch, Environment, PosedActor
from revolve2.core.physics.environment_actor_controller import EnvironmentActorController
from revolve2.runners.mujoco import LocalRunner
from revolve2.standard_resources import terrains
from pyrr import Vector3, Quaternion
from brain.decentralized_brain import DecentralizedBrain
from double_population import DoublePopulation

import neat


class DecentralizedNEATOptimizer:
    """
    Evolutionary NEAT optimizer. Does not need to implement much, just a Revolve2 gimmick
    """
    population: DoublePopulation

    _robot_bodies: List[Body]

    _runner: Runner
    _TERRAIN = terrains.flat()

    _simulation_time: int
    _sampling_frequency: float
    _control_frequency: float

    _sensory_length: int
    _single_message_length: int
    _full_message_length: int

    def __init__(self, robot_bodies: List[Body],
                 config_bu_path: str, config_td_path: str,
                 population_size: int, simulation_time: int,
                 sampling_frequency: float, control_frequency: float,
                 sensory_length: int, single_message_length: int, biggest_body: int):

        # vector message length (with dof output) * maximum estimated number of messages (i.e. number of joints of the largest body)
        full_message_length = biggest_body * (single_message_length + 1) + 1

        # finds input, output, population and updates their values
        set_config(sensory_length, single_message_length, population_size, config_bu_path)
        set_config(full_message_length, 1, population_size, config_td_path)

        config_bu = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                                neat.DefaultSpeciesSet, neat.DefaultStagnation,
                                config_bu_path)

        config_td = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                                neat.DefaultSpeciesSet, neat.DefaultStagnation,
                                config_td_path)

        # create population and set its reporters
        self.population = DoublePopulation(config_bu, config_td)
        self.population.add_reporter(neat.StdOutReporter(True))
        stats1 = neat.StatisticsReporter()
        stats2 = neat.StatisticsReporter()
        self.population.add_reporter(stats1, 0)
        self.population.add_reporter(stats2, 1)
        self.population.add_reporter(neat.Checkpointer(generation_interval=1, filename_prefix='Checkpoints/bu_checkpoint-'), 0)  # comment if you want to run from other checkpoint (see self.run())
        self.population.add_reporter(neat.Checkpointer(generation_interval=1, filename_prefix='Checkpoints/td_checkpoint-'), 1)  # comment if you want to run from other checkpoint (see self.run())

        self._robot_bodies = robot_bodies

        self._simulation_time = simulation_time
        self._sampling_frequency = sampling_frequency
        self._control_frequency = control_frequency

        self._sensory_length = sensory_length
        self._single_message_length = single_message_length
        self._full_message_length = full_message_length

        self._runner = LocalRunner(headless=True)

    async def run(self, num_generations):

        # Uncomment if you want to run from checkpoint n
        # checkpoint_file = 'neat-checkpoint-n'
        # self.population = neat.Checkpointer.restore_checkpoint(checkpoint_file)

        best_bu, best_td = await self.population.run(self.evaluate_generation, num_generations)

        with open("Checkpoints/best_bu.pickle", "wb") as f:
            pickle.dump(best_bu, f)
        with open("Checkpoints/best_td.pickle", "wb") as f:
            pickle.dump(best_td, f)

    async def evaluate_generation(self, genomes_bu, genomes_td, config_bu, config_td):

        batch = Batch(
            simulation_time=self._simulation_time,
            sampling_frequency=self._sampling_frequency,
            control_frequency=self._control_frequency,
        )
        for genome_bu, genome_td in zip(genomes_bu, genomes_td):
            _, genotype_bu = genome_bu
            _, genotype_td = genome_td

            for body in self._robot_bodies:

                _, controller = self.develop(genotype_bu, genotype_td, body, self._sensory_length, self._single_message_length, self._full_message_length, config_bu, config_td).make_actor_and_controller()
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

        batch_results = await self._runner.run_batch(batch)

        counter = 0
        for genome_bu, genome_td in zip(genomes_bu, genomes_td):
            _, genotype_bu = genome_bu
            _, genotype_td = genome_td
            for environment_result in batch_results.environment_results[counter:counter+len(self._robot_bodies)]:

                if genotype_bu.fitness is None:
                    genotype_bu.fitness = await self._calculate_fitness(environment_result.environment_states)
                else:
                    genotype_bu.fitness += await self._calculate_fitness(environment_result.environment_states)

            genotype_bu.fitness /= len(self._robot_bodies)
            genotype_td.fitness = genotype_bu.fitness
            counter += len(self._robot_bodies)

    @staticmethod
    async def _calculate_fitness(states: List[EnvironmentState]) -> float:
        # distance traveled on the xy plane
        z = [state.actor_states[0].position[2] for state in states]
        stdev = np.std(z)
        return math.sqrt(
            (states[0].actor_states[0].position[0] - states[-1].actor_states[0].position[0]) ** 2
            + ((states[0].actor_states[0].position[1] - states[-1].actor_states[0].position[1]) ** 2))-(stdev/2)

    @staticmethod
    def develop(genotype_bu, genotype_td, robot_body,
                sensory_length, single_message_length, full_message_length,
                config_bu, config_td) -> ModularRobot:

        dof_ranges = np.full(len(robot_body.find_active_hinges()), 1.0)
        brain = DecentralizedBrain(genotype_bu, genotype_td, dof_ranges, sensory_length, single_message_length,
                                   full_message_length, config_bu, config_td)

        return ModularRobot(robot_body, brain)


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
