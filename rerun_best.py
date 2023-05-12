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

    with open('Checkpoints/best_bu.pickle', 'rb') as f:
        cbu = pickle.load(f)
    with open('Checkpoints/best_td.pickle', 'rb') as f:
        ctd = pickle.load(f)

    local_dir = os.path.dirname(__file__)
    config_bu_path = os.path.join(local_dir, 'configBU.txt')
    config_bu = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                            neat.DefaultSpeciesSet, neat.DefaultStagnation,
                            config_bu_path)
    local_dir = os.path.dirname(__file__)
    config_td_path = os.path.join(local_dir, 'configTD.txt')
    config_td = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                            neat.DefaultSpeciesSet, neat.DefaultStagnation,
                            config_td_path)

    print(f"fitness: {cbu.fitness}")

    rerunner = ModularRobotRerunner()
    await rerunner.rerun(develop(cbu, ctd, spider(), 14, 32, 15*33, config_bu, config_td), 60, terrain=terrains.flat())


if __name__ == "__main__":
    import asyncio

    asyncio.run(main())


