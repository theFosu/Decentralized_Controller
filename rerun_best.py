"""Visualize and simulate the best robot from the optimization process."""

from revolve2.runners.mujoco import ModularRobotRerunner
from revolve2.core.physics.running import RecordSettings
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

    #recording = RecordSettings('Videos/spider')
    #, simulation_time=20, record_settings=recording

    rerunner = ModularRobotRerunner()
    await rerunner.rerun(DecentralizedNEATOptimizer.develop(cbu, ctd, spider(), 17, 32, 11*33+1, config_bu, config_td), 20, terrain=terrains.flat())


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
            g.fitness = -99
        if best_bu is None or g.fitness > best_bu.fitness:
            best_bu = g
    for g in itervalues(pop_td.population):
        if g is None:
            continue
        if g.fitness is None:
            g.fitness = -99
        if best_td is None or g.fitness > best_bu.fitness:
            best_td = g
    print(best_bu.fitness)

    with open("Checkpoints/best_bu.pickle", "wb") as f:
        pickle.dump(best_bu, f)
    with open("Checkpoints/best_td.pickle", "wb") as f:
        pickle.dump(best_td, f)


def final_stats(generations: int, threshold: int):
    import neat
    from neat.six_util import itervalues
    import visualize

    stats_bu = neat.StatisticsReporter()
    stats_td = neat.StatisticsReporter()

    best_bu = None
    best_td = None

    for i in range(generations):
        filename = 'Checkpoints/bu_checkpoint-' + str(i)
        pop_bu = neat.Checkpointer.restore_checkpoint(filename)
        filename = 'Checkpoints/td_checkpoint-' + str(i)
        pop_td = neat.Checkpointer.restore_checkpoint(filename)

        pop_bu.generation = i
        pop_td.generation = i

        for g in itervalues(pop_bu.population):
            if g is None:
                continue
            if g.fitness is None:
                g.fitness = -9
            if best_bu is None or g.fitness > best_bu.fitness:
                best_bu = g
        for g in itervalues(pop_td.population):
            if g is None:
                continue
            if g.fitness is None:
                g.fitness = -9
            if best_td is None or g.fitness > best_bu.fitness:
                best_td = g

        stats_bu.post_evaluate(pop_bu.config, pop_bu.population, pop_bu.species, best_bu)
        stats_td.post_evaluate(pop_td.config, pop_td.population, pop_td.species, best_td)

    filename = 'Graphs/avg_fitness-bu.svg'
    visualize.plot_stats(statistics=stats_bu, threshold=threshold, ylog=False, view=False, filename=filename)
    filename = 'Graphs/speciation-bu.svg'
    visualize.plot_species(statistics=stats_bu, threshold=threshold, view=False, filename=filename)

    filename = 'Graphs/avg_fitness-td.svg'
    visualize.plot_stats(statistics=stats_td, threshold=threshold, ylog=False, view=False, filename=filename)
    filename = 'Graphs/speciation-td.svg'
    visualize.plot_species(statistics=stats_td, threshold=threshold, view=False, filename=filename)


if __name__ == "__main__":
    import asyncio

    asyncio.run(main())


