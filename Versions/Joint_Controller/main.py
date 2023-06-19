import logging

from standard_resources.modular_robots import *
from optimizer import DecentralizedOptimizer as Optimizer

'''Revolve has been changed:
    _runner interface should async run batch
    _local_runner should async run batch
    _local_runner's get actor state now includes DOFs
    _results now has dof_state on top of position and orientation'''


def main() -> None:
    """Run the simulation."""

    # Evolutionary hyperparameters
    POPULATION = 600
    NUM_GENERATIONS = 700

    # Simulation (hyper)parameters
    NUM_SIMULATORS = 60
    SIMULATION_TIME = 5
    SAMPLING_FREQUENCY = 60
    CONTROL_FREQUENCY = 60

    # Neural network hyperparameters
    BATCH_SIZE = 16
    SENSORY_LENGTH = 7 + 7  # joint info + body info
    SINGLE_MESSAGE_LENGTH = 16
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

    optimizer = Optimizer(bodies, POPULATION, SIMULATION_TIME, SAMPLING_FREQUENCY, CONTROL_FREQUENCY,
                          SENSORY_LENGTH, BATCH_SIZE, SINGLE_MESSAGE_LENGTH, BIGGEST_BODY, NUM_SIMULATORS)

    logging.info("Starting initial optimization process..")

    optimizer.run(NUM_GENERATIONS)

    logging.info(f"Finished optimizing.")


if __name__ == "__main__":
    #import asyncio
    #asyncio.run(main())

    main()
