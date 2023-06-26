import pandas as pd

from standard_resources import terrains
from revolve2.core.physics.running import Batch, Environment, PosedActor
from revolve2.core.physics.environment_actor_controller import EnvironmentActorController
from revolve2.runners.mujoco import LocalRunner
from standard_resources.modular_robots import *
from evotorch.neuroevolution.net.layers import LocomotorNet
import torch
from customNE import CustomNE
import pickle
from pyrr import Vector3, Quaternion
import graphviz
from matplotlib.patches import Patch
import matplotlib as mpl
import matplotlib.pyplot as plt


def main() -> None:
    policy_args = {'in_features': (14 * 5), 'out_features': 1, 'num_sinusoids': 16}

    torch.set_default_dtype(torch.double)
    problem = CustomNE(objective_sense='max', network=LocomotorNet, network_args=policy_args, num_neighbors=4,
                       state_length=14, num_simulators=1, bodies=[spider()])

    with open('new-Checkpoints/_generation0150.pickle', 'rb') as f:
        file = pickle.load(f)

    best = file['best']
    network = problem.make_net(best)

    fitnesses = get_fitnesses(network)

    plot_fitnesses(fitnesses)

    plot_effects(fitnesses)


def get_fitnesses(network):

    sample_size = 100

    runner = LocalRunner(headless=True, num_simulators=int(sample_size/2))

    bodies = [
        babya(),
        insect(),
        spider(),
        ant(),
        babyb(),
        gecko()
    ]

    names = ['Babya', 'Insect', 'Spider', 'Ant', 'Babyb', 'Gecko']

    fitnesses = {}

    for body, name in zip(bodies, names):
        batch = Batch(
            simulation_time=6,
            sampling_frequency=60,
            control_frequency=60,
            simulation_timestep=0.001
        )

        for i in range(sample_size):
            _, controller = CustomNE.develop(network, body, 4, 14).make_actor_and_controller()
            actor = controller.actor
            bounding_box = actor.calc_aabb()
            env = Environment(EnvironmentActorController(controller))
            env.static_geometries.extend(terrains.flat().static_geometry)
            env.actors.append(
                PosedActor(
                    actor,
                    Vector3(
                        [
                            0.0,
                            0.0,
                            bounding_box.size.z / 2.0 - bounding_box.offset.z,
                        ]
                    ),
                    Quaternion(),
                    [0.0 for _ in controller.get_dof_targets()],
                )
            )
            batch.environments.append(env)

        batch_results = runner.run_batch(batch)

        fitness = []
        for environment_result in batch_results.environment_results:
            fitness.append(CustomNE.get_fitness(environment_result.environment_states))

        fitnesses[name] = fitness

    return fitnesses


def plot_effects(fitnesses):
    dic = {'Train Bodies': [], 'Test Bodies': []}
    for body, fitness in fitnesses.items():
        if body == 'Babyb' or body == 'Gecko':
            dic['Test Bodies'].extend(fitness)
        else:
            dic['Train Bodies'].extend(fitness)

    # Bootstrap to match sizes. We're not making statistical tests so it should not matter
    dic['Test Bodies'].extend(dic['Test Bodies'])

    mpl.use('TkAgg')
    df = pd.DataFrame.from_dict(dic)

    df.boxplot()
    plt.xticks(range(1, len(df.columns) + 1), df.columns)

    plt.xlabel('Body Sets')
    plt.ylabel('Fitnesses')
    plt.title('Average Fitness Per Body Set')

    plt.savefig('Graphs/fitness_across_main_testtrain')
    plt.show()


def plot_fitnesses(fitnesses):
    mpl.use('TkAgg')

    colors = ['blue', 'blue', 'blue', 'blue', 'red', 'red']

    df = pd.DataFrame.from_dict(fitnesses)

    df.boxplot()
    plt.xticks(range(1, len(df.columns)+1), df.columns)

    '''legend_colors = ['blue', 'red']
    legend_labels = ['Training Bodies', 'Testing Bodies']
    legend_patches = [Patch(color=color, label=label) for color, label in zip(legend_colors, legend_labels)]
    plt.legend(handles=legend_patches, loc='lower left')'''

    plt.xlabel('Bodies')
    plt.ylabel('Fitnesses')
    plt.title('Average Fitness Per Body')

    plt.savefig('Graphs/fitness_across_main')
    plt.show()


if __name__ == "__main__":
    main()
