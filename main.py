"""Visualize and run a modular robot using Mujoco."""
import logging
import os

from revolve2.standard_resources.modular_robots import spider
from optimizer import DecentralizedNEATOptimizer


# REMEMBER: little modification in neat library to allow population.run to run asynchronously

# TODO: more statistics, better fitness, several bodies

async def main() -> None:
    """Run the simulation."""

    # Evolutionary hyperparameters
    INITIAL_POPULATION = 7
    NUM_GENERATIONS = 2

    # Simulation (hyper)parameters
    SIMULATION_TIME = 10
    SAMPLING_FREQUENCY = 60
    CONTROL_FREQUENCY = 60

    # Neural network hyperparameters
    SENSORY_LENGTH = 7 * 2  # joint info + body info
    SINGLE_MESSAGE_LENGTH = 32
    BIGGEST_BODY = 15

    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] [%(levelname)s] [%(module)s] %(message)s",
    )
    logging.info(f"Starting optimization")

    body = spider()

    # get neat config for its hyperparameters
    local_dir = os.path.dirname(__file__)
    config_bu_path = os.path.join(local_dir, 'configBU.txt')
    config_td_path = os.path.join(local_dir, 'configTD.txt')

    optimizer = DecentralizedNEATOptimizer(body, config_bu_path, config_td_path, INITIAL_POPULATION,
                                           SIMULATION_TIME, SAMPLING_FREQUENCY, CONTROL_FREQUENCY,
                                           SENSORY_LENGTH, SINGLE_MESSAGE_LENGTH, BIGGEST_BODY)

    logging.info("Starting optimization process..")

    await optimizer.run(NUM_GENERATIONS)

    logging.info(f"Finished optimizing.")


if __name__ == "__main__":
    import asyncio

    asyncio.run(main())
