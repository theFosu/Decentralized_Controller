import logging

from standard_resources.modular_robots import *
from optimizer import DecentralizedOptimizer as Optimizer

'''Revolve has been changed:
    _runner interface should async run batch
    _local_runner should async run batch
    _local_runner's get actor state now includes DOFs
    _results now has dof_state on top of position and orientation
    armature was 0.002'''


def main() -> None:
    """Run the simulation."""

    # Evolutionary hyperparameters
    POPULATION = 254
    NUM_GENERATIONS = 150

    # Simulation (hyper)parameters
    NUM_SIMULATORS = 64
    SIMULATION_TIME = 6
    SAMPLING_FREQUENCY = 60
    CONTROL_FREQUENCY = 60

    # Neural network hyperparameters
    SENSORY_LENGTH = 7 + 7  # joint info + body info
    NEIGHBORS = 4
    NUM_SINUSOIDS = 16

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
    # test bodies: babyb, gecko

    optimizer = Optimizer(bodies, POPULATION, SIMULATION_TIME, SAMPLING_FREQUENCY, CONTROL_FREQUENCY,
                          SENSORY_LENGTH, NEIGHBORS, NUM_SINUSOIDS, NUM_SIMULATORS)

    logging.info("Starting initial optimization process..")

    optimizer.run(NUM_GENERATIONS)

    logging.info(f"Finished optimizing.")


if __name__ == "__main__":
    main()
