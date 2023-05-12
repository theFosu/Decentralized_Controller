"""Visualize and run a modular robot using Mujoco."""
import logging
import os

from random import Random

from revolve2.core.modular_robot import ActiveHinge, Body, Brick, ModularRobot
from revolve2.standard_resources.modular_robots import spider
from revolve2.core.modular_robot.brains import *
from revolve2.core.optimization import DbId
from revolve2.core.database import open_async_database_sqlite
from genotype import random as random_genotype
from optimizer import Optimizer


async def main() -> None:
    """Run the simulation."""

    # Evolutionary hyperparameters
    POPULATION_SIZE = 10
    OFFSPRING_SIZE = 5  # Tip: explore more exploit (a little) less
    NUM_GENERATIONS = 6

    # Simulation (hyper)parameters
    SIMULATION_TIME = 10
    SAMPLING_FREQUENCY = 60
    CONTROL_FREQUENCY = 60

    # Erase database if already present (could cause errors)
    if os.path.exists('./database/db.sqlite'):
        prompt = input('do you want to erase existent database? (press enter to confirm)')
        if prompt == '':
            os.remove('./database/db.sqlite')
    database = open_async_database_sqlite("./database", create=True)

    # unique database identifier for optimizer
    db_id = DbId.root("evolutionarylocomotion")

    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] [%(levelname)s] [%(module)s] %(message)s",
    )
    logging.info(f"Starting optimization")

    rng = Random()
    rng.seed(5)

    # TODO: (later) initialize several types of robots, OR make bodies evolve too
    body = spider()

    initial_population = [
        random_genotype(rng, body)

        for _ in range(POPULATION_SIZE)
    ]

    maybe_optimizer = await Optimizer.from_database(
        database=database,
        db_id=db_id,
        rng=rng,
        robot_body=body,
    )
    if maybe_optimizer is not None:
        optimizer = maybe_optimizer
    else:
        optimizer = await Optimizer.new(
            database=database,
            db_id=db_id,
            offspring_size=OFFSPRING_SIZE,
            initial_population=initial_population,
            rng=rng,
            robot_body=body,
            simulation_time=SIMULATION_TIME,
            sampling_frequency=SAMPLING_FREQUENCY,
            control_frequency=CONTROL_FREQUENCY,
            num_generations=NUM_GENERATIONS,

        )

    logging.info("Starting optimization process..")

    await optimizer.run()

    logging.info(f"Finished optimizing.")


if __name__ == "__main__":
    import asyncio

    asyncio.run(main())
