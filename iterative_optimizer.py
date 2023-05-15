import math
import pickle

import numpy as np
from typing import List
import neat
from revolve2.core.modular_robot import ActiveHinge, Body, Brick, ModularRobot
from revolve2.core.physics.running import EnvironmentState, ActorState, ActorControl, Runner, Batch, Environment, PosedActor
from revolve2.core.physics.environment_actor_controller import EnvironmentActorController
from revolve2.runners.mujoco import LocalRunner
from revolve2.standard_resources import terrains
from pyrr import Vector3, Quaternion
from brain.decentralized_brain import DecentralizedBrain


class DecentralizedNEATOptimizer:
    """
    Evolutionary NEAT optimizer. Does not need to implement much, just a Revolve2 gimmick
    """
    population: neat.Population

    _robot_bodies: List[Body]
    type_genome: str

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
                 sensory_length: int, single_message_length: int, biggest_body: int, type_genome: str):

        # vector message length (with dof output) * maximum estimated number of messages (i.e. number of joints of the largest body)
        full_message_length = biggest_body * (single_message_length + 1)

        # finds input, output, population and updates their values
        if type_genome == 'bu':
            set_config(sensory_length, single_message_length, population_size, config_bu_path)
            config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                                 neat.DefaultSpeciesSet, neat.DefaultStagnation,
                                 config_bu_path)
        else:
            set_config(full_message_length, 1, population_size, config_td_path)
            config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                                 neat.DefaultSpeciesSet, neat.DefaultStagnation,
                                 config_td_path)
        self.type_genome = type_genome

        # create population and set its reporters
        self.population = neat.Population(config)
        stats = neat.StatisticsReporter()
        self.population.add_reporter(stats)
        prefix = type_genome + 'Checkpoints/bu_checkpoint-'
        self.population.add_reporter(neat.Checkpointer(generation_interval=1, filename_prefix= prefix))  # comment if you want to run from other checkpoint (see self.run())
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

        best = await self.population.run(self.evaluate_generation, num_generations)
        filename = 'Checkpoints/best_' + self.type_genome + '.pickle'
        with open(filename, "wb") as f:
            pickle.dump(best, f)

    async def evaluate_generation(self, genomes, config):

        batch = Batch(
            simulation_time=self._simulation_time,
            sampling_frequency=self._sampling_frequency,
            control_frequency=self._control_frequency,
        )
        i = 0
        for _, genotype in genomes:

            body_index = i % len(self._robot_bodies)

            body = self._robot_bodies[body_index]

            _, controller = develop(genotype, body, self._sensory_length, self._single_message_length, self._full_message_length, config, self.type_genome).make_actor_and_controller()
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
            i += 1

        batch_results = await self._runner.run_batch(batch)

        for environment_result, genome in zip(batch_results.environment_results, genomes):

            if genome.fitness is None:
                genome.fitness = await self._calculate_fitness(environment_result.environment_states)
            else:
                new_fitness = await self._calculate_fitness(environment_result.environment_states)
                genome.fitness = (genome.fitness + new_fitness) / 2

    @staticmethod
    async def _calculate_fitness(states: List[EnvironmentState]) -> float:
        # distance traveled on the xy plane
        z = [state.actor_states[0].position[2] for state in states]
        stdev = np.std(z)
        return math.sqrt(
            (states[0].actor_states[0].position[0] - states[-1].actor_states[0].position[0]) ** 2
            + ((states[0].actor_states[0].position[1] - states[-1].actor_states[0].position[1]) ** 2))-(stdev/4)


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


def develop(genotype, robot_body,
            sensory_length, single_message_length, full_message_length,
            config, type_genome) -> ModularRobot:

    if type_genome == 'bu':
        genotype1 = genotype
        config1 = config

        config2 = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                                neat.DefaultSpeciesSet, neat.DefaultStagnation,
                                'configTD.txt')
        with open('Checkpoints/best_td.pickle', 'rb') as f:
            genotype2 = pickle.load(f)
    else:
        genotype2 = genotype
        config2 = config

        config1 = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                                neat.DefaultSpeciesSet, neat.DefaultStagnation,
                                'configBU.txt')
        with open('Checkpoints/best_bu.pickle', 'rb') as f:
            genotype1 = pickle.load(f)

    dof_ranges = np.full(len(robot_body.find_active_hinges()), 1.0)
    brain = DecentralizedBrain(genotype1, genotype2, dof_ranges, sensory_length, single_message_length, full_message_length, config1, config2)

    return ModularRobot(robot_body, brain)

