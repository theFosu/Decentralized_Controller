"""Visualize and run a modular robot using Mujoco."""
import logging
import os

from revolve2.standard_resources.modular_robots import *
from initial_optimizer import DecentralizedNEATOptimizer as InitialOptimizer
from iterative_optimizer import DecentralizedNEATOptimizer as IterativeOptimizer


async def main() -> None:
    """Run the simulation."""

    # Evolutionary hyperparameters
    POPULATION = 40
    NUM_GENERATIONS = 3

    # Simulation (hyper)parameters
    SIMULATION_TIME = 15
    SAMPLING_FREQUENCY = 60
    CONTROL_FREQUENCY = 30

    SWITCH_TIME = 4
    NUM_ITERATIONS = 5

    # Neural network hyperparameters
    SENSORY_LENGTH = 7 + 10  # joint info + body info
    SINGLE_MESSAGE_LENGTH = 32
    BIGGEST_BODY = 11

    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] [%(levelname)s] [%(module)s] %(message)s",
    )
    logging.info(f"Starting optimization")

    bodies = [
        babya(),
        insect(),
        spider(),
        ant()
    ]
    # not included: queen, squarish, zappa, park. Original test bodies: babyb, gecko, penguin

    # get neat config for its hyperparameters
    local_dir = os.path.dirname(__file__)
    config_bu_path = os.path.join(local_dir, 'configBU.txt')
    config_td_path = os.path.join(local_dir, 'configTD.txt')

    optimizer = InitialOptimizer(bodies, config_bu_path, config_td_path, POPULATION, SIMULATION_TIME, SAMPLING_FREQUENCY, CONTROL_FREQUENCY,
                                 SENSORY_LENGTH, SINGLE_MESSAGE_LENGTH, BIGGEST_BODY)

    logging.info("Starting initial optimization process..")

    await optimizer.run(NUM_GENERATIONS)

    logging.info(f"Finished optimizing.")

    optimizer = IterativeOptimizer(bodies, SIMULATION_TIME, SAMPLING_FREQUENCY, CONTROL_FREQUENCY,
                                   SENSORY_LENGTH, SINGLE_MESSAGE_LENGTH, BIGGEST_BODY)

    logging.info("Starting iterative optimization process..")

    await optimizer.run(NUM_ITERATIONS, SWITCH_TIME)

    logging.info(f"Finished optimizing.")

if __name__ == "__main__":
    import asyncio

    asyncio.run(main())
