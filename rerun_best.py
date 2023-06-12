"""Visualize and simulate the best robot from the optimization process."""

from revolve2.runners.mujoco import ModularRobotRerunner
from standard_resources import terrains
from standard_resources.modular_robots import *
from evotorch.neuroevolution import NEProblem
from brain.ModularPolicy import JointPolicy
import torch
from optimizer import DecentralizedOptimizer
from CustomNE import CustomNE
import pickle
import os


async def main() -> None:
    """Run the script."""

    policy_args = {'state_dim': 14, 'action_dim': 1,
                   'msg_dim': 16, 'batch_size': 16,
                   'max_action': 1, 'max_children': 11}

    torch.set_default_dtype(torch.double)
    problem = CustomNE(objective_sense='max', network=JointPolicy, network_args=policy_args, single_message_length=16, full_message_length=(11*16), num_simulators=1, bodies=[spider()])

    with open('Checkpoints/_generation0700.pickle', 'rb') as f:
        file = pickle.load(f)
    print(file)
    best = file['best']
    network = problem.parameterize_net(best)

    print(f"fitness: {file['best_eval']}")

    # recording = RecordSettings('Videos/spider')
    # , simulation_time=20, record_settings=recording

    rerunner = ModularRobotRerunner()
    await rerunner.rerun(CustomNE.develop(network, spider(), 11*16, 16), 60, start_paused=False, terrain=terrains.flat())


def final_stats(generations: int, threshold: int):
    import neat
    from neat.six_util import itervalues
    import visualize

    stats_bu = neat.StatisticsReporter()

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


if __name__ == "__main__":
    import asyncio

    asyncio.run(main())


