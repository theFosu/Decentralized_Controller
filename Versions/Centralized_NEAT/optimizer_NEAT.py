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
from brain.neat_brain import NEATBrain

import neat


class NEATOptimizer:
    """
    Evolutionary NEAT optimizer. Does not need to implement much, just a Revolve2 gimmick
    """
    population: neat.Population
    _robot_body: Body

    _runner: Runner
    _TERRAIN = terrains.flat()

    _simulation_time: int
    _sampling_frequency: float
    _control_frequency: float

    def __init__(self, robot_body: Body, config_path: str, population_size: int, simulation_time: int, sampling_frequency: float, control_frequency: float):
        # extract number of input and output nodes
        ninput, dof_ranges = make_io(robot_body)

        noutput = len(dof_ranges)

        # finds input, output, population and updates their values
        set_config(ninput, noutput, population_size, config_path)

        config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                             neat.DefaultSpeciesSet, neat.DefaultStagnation,
                             config_path)
        # create population and set its reporters
        self.population = neat.Population(config)
        self.population.add_reporter(neat.StdOutReporter(True))
        stats = neat.StatisticsReporter()
        self.population.add_reporter(stats)
        #self.population.add_reporter(neat.Checkpointer(1))

        self._robot_body = robot_body

        self._simulation_time = simulation_time
        self._sampling_frequency = sampling_frequency
        self._control_frequency = control_frequency

        self._runner = LocalRunner(headless=True)

    async def run(self, num_generations):

        if False:
            checkpoint_file = 'neat-checkpoint-6'
            self.population = neat.Checkpointer.restore_checkpoint(checkpoint_file)

        best = await self.population.run(self.evaluate_generation, 1)

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


def make_io(robot_body: Body):
    """Makes the length of the input and the DOF ranges corresponding to the body"""

    ninput = ((len(robot_body.find_active_hinges()) + len(robot_body.find_bricks()) + 1) * 7)

    dof_ranges = np.full(len(robot_body.find_active_hinges()), 1.0)

    return ninput, dof_ranges


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

    ninput, dof_ranges = make_io(robot_body)
    brain = NEATBrain(genotype, dof_ranges, config)

    return ModularRobot(robot_body, brain)
