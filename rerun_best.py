"""Visualize and simulate the best robot from the optimization process."""

from revolve2.runners.mujoco import ModularRobotRerunner
from revolve2.standard_resources import terrains
from revolve2.standard_resources.modular_robots import spider, gecko, snake
import neat
from optimizer import develop
import pickle
import os


async def main() -> None:
    """Run the script."""

    with open('best.pickle', 'rb') as f:
        c = pickle.load(f)

    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, 'configBU.txt')
    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation,
                         config_path)

    print(f"fitness: {c.fitness}")

    rerunner = ModularRobotRerunner()
    await rerunner.rerun(develop(c, spider(), config), 60, terrain=terrains.flat())


if __name__ == "__main__":
    import asyncio

    asyncio.run(main())


