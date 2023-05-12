"""Visualize and run a modular robot using Mujoco."""
import logging
import os

from revolve2.core.modular_robot import ActiveHinge, Body, Brick, ModularRobot
from revolve2.standard_resources.modular_robots import *
from optimizer_NEAT import NEATOptimizer


# REMEMBER: little modification to allow population.run to run asynchronously

async def main() -> None:
    """Run the simulation."""

    # Evolutionary hyperparameters
    INITIAL_POPULATION = 5
    NUM_GENERATIONS = 3

    # Simulation (hyper)parameters
    SIMULATION_TIME = 10
    SAMPLING_FREQUENCY = 60
    CONTROL_FREQUENCY = 60

    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] [%(levelname)s] [%(module)s] %(message)s",
    )
    logging.info(f"Starting optimization")

    body = pentapod()

    # get neat config for its hyperparameters
    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, 'config.txt')

    optimizer = NEATOptimizer(body, config_path, INITIAL_POPULATION, SIMULATION_TIME, SAMPLING_FREQUENCY, CONTROL_FREQUENCY)

    logging.info("Starting optimization process..")

    await optimizer.run(NUM_GENERATIONS)

    logging.info(f"Finished optimizing.")


if __name__ == "__main__":
    import asyncio

    asyncio.run(main())
