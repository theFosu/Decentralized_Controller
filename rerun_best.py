"""Visualize and simulate the best robot from the optimization process."""

from revolve2.runners.mujoco import ModularRobotRerunner
from revolve2.standard_resources import terrains
from revolve2.standard_resources.modular_robots import *
import neat
from initial_optimizer import DecentralizedNEATOptimizer
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

    config_td_path = os.path.join(local_dir, 'configTD.txt')
    config_td = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                            neat.DefaultSpeciesSet, neat.DefaultStagnation,
                            config_td_path)

    print(f"fitness: {cbu.fitness}")

    rerunner = ModularRobotRerunner()
    await rerunner.rerun(DecentralizedNEATOptimizer.develop(cbu, ctd, babya(), 17, 32, 11*33, config_bu, config_td), 4, terrain=terrains.flat())


def finalize_checkpoint():
    import neat
    from neat.six_util import itervalues

    pop_bu = neat.Checkpointer.restore_checkpoint('Checkpoints/bu_checkpoint-1')
    pop_td = neat.Checkpointer.restore_checkpoint('Checkpoints/td_checkpoint-1')

    best_bu = None
    best_td = None
    for g in itervalues(pop_bu.population):
        if g is None:
            continue
        if g.fitness is None:
            g.fitness = -9999
        if best_bu is None or g.fitness > best_bu.fitness:
            best_bu = g
    for g in itervalues(pop_td.population):
        if g is None:
            continue
        if g.fitness is None:
            g.fitness = -9999
        if best_td is None or g.fitness > best_bu.fitness:
            best_td = g
    print(best_bu.fitness)

    with open("Checkpoints/best_bu.pickle", "wb") as f:
        pickle.dump(best_bu, f)
    with open("Checkpoints/best_td.pickle", "wb") as f:
        pickle.dump(best_td, f)


if __name__ == "__main__":
    import asyncio

    asyncio.run(main())


