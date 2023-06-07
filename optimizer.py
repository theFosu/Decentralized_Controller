import pickle
from typing import List
from revolve2.core.modular_robot import Body
from revolve2.core.physics.running import Runner
from revolve2.runners.mujoco import LocalRunner
from brain.ModularPolicy import JointPolicy
from CustomNE import CustomNE

from evotorch.algorithms import PGPE
from evotorch.logging import StdOutLogger, PicklingLogger, PandasLogger
import torch


class DecentralizedOptimizer:
    """
    Evolutionary NEAT optimizer. Does not need to implement much, just a Revolve2 gimmick
    """
    solver: PGPE
    Logger: StdOutLogger
    Pickler: PicklingLogger
    Pandaer: PandasLogger

    _robot_bodies: List[Body]

    _runner: Runner

    _simulation_time: int
    _sampling_frequency: float
    _control_frequency: float

    _full_message_length: int
    _single_message_length: int

    def __init__(self, robot_bodies: List[Body],
                 population_size: int, simulation_time: int,
                 sampling_frequency: float, control_frequency: float,
                 sensory_length: int, batch_size: int, single_message_length: int, biggest_body: int):

        self._full_message_length = single_message_length * biggest_body

        self._runner = LocalRunner(headless=True, num_simulators=8)

        policy_args = {'state_dim': sensory_length, 'action_dim': 1,
                       'msg_dim': single_message_length, 'batch_size': batch_size,
                       'max_action': 1, 'max_children': biggest_body}

        torch.set_default_dtype(torch.double)

        problem = CustomNE(bodies=robot_bodies,
                           single_message_length=single_message_length, full_message_length=self._full_message_length,
                           runner=self._runner,
                           objective_sense="max", network=JointPolicy, network_args=policy_args,
                           simulation_time=simulation_time,
                           sampling_frequency=sampling_frequency,
                           control_frequency=control_frequency,
                           initial_bounds=(-1, 1))

        self.solver = PGPE(problem, popsize=population_size, stdev_init=5, distributed=False,
                           center_learning_rate=0.01125, stdev_learning_rate=0.1)

        self.Logger = StdOutLogger(self.solver)
        self.Pickler = PicklingLogger(self.solver, interval=1, directory='Checkpoints/', prefix='',
                                      items_to_save=('best', 'pop_best', 'center', 'best_eval', 'worst_eval', 'median_eval', 'mean_eval', 'pop_best_eval'))
        self.Pandaer = PandasLogger(self.solver)

        self._robot_bodies = robot_bodies

        self._simulation_time = simulation_time
        self._sampling_frequency = sampling_frequency
        self._control_frequency = control_frequency

        self._single_message_length = single_message_length

    def run(self, num_generations):

        self.solver.run(num_generations)
        with open('graphs/dataframe.pickle', 'wb') as f:
            pickle.dump(self.Pandaer.to_dataframe(), f)
        print(self.solver.status)
