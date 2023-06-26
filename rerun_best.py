"""Visualize and simulate the best robot from the optimization process."""

from revolve2.runners.mujoco import ModularRobotRerunner
from revolve2.core.physics.running import RecordSettings
from standard_resources import terrains
from standard_resources.modular_robots import *
from evotorch.neuroevolution.net.layers import LocomotorNet
import torch
from customNE import CustomNE
import pickle


async def main() -> None:
    """Run the script."""

    policy_args = {'in_features': (14*5), 'out_features': 1, 'num_sinusoids': 16}

    torch.set_default_dtype(torch.double)
    problem = CustomNE(objective_sense='max', network=LocomotorNet, network_args=policy_args, num_neighbors=4, state_length=14, num_simulators=1, bodies=[spider()])

    with open('DecLoco-Checkpoints/_generation0150.pickle', 'rb') as f:
        file = pickle.load(f)
    # print(file)
    best = file['best']
    network = problem.make_net(best)

    print(f"fitness: {file['best_eval']}")

    recording = RecordSettings('Media/babya')
    # , simulation_time=20, record_settings=recording

    rerunner = ModularRobotRerunner()
    rerunner.rerun(CustomNE.develop(network, gecko(), 4, 14), 60, start_paused=False, terrain=terrains.flat(),
                   simulation_time=4, record_settings=recording)


if __name__ == "__main__":
    import asyncio

    asyncio.run(main())


