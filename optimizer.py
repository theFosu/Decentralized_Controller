import requests
from typing import List
from revolve2.core.modular_robot import Body
from customNE import CustomNE

from evotorch.algorithms import PGPE
from evotorch.neuroevolution.net.layers import LocomotorNet
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

    _simulation_time: int
    _sampling_frequency: float
    _control_frequency: float

    _num_neighbors: int

    def __init__(self, robot_bodies: List[Body],
                 population_size: int, simulation_time: int,
                 sampling_frequency: float, control_frequency: float,
                 sensory_length: int, num_neighbors: int, num_sinusoids: int, num_simulators: int):

        input_length = sensory_length + (sensory_length*num_neighbors)

        policy_args = {'in_features': input_length, 'out_features': 1, 'num_sinusoids': num_sinusoids}

        torch.set_default_dtype(torch.double)

        problem = CustomNE(bodies=robot_bodies,
                           num_neighbors=num_neighbors, state_length=sensory_length,
                           num_simulators=num_simulators,
                           objective_sense="max", network=LocomotorNet, network_args=policy_args,
                           simulation_time=simulation_time,
                           sampling_frequency=sampling_frequency,
                           control_frequency=control_frequency,
                           initial_bounds=(-1, 1))

        max_speed = 4.5/15.
        self.solver = PGPE(
            problem,
            popsize=population_size,
            radius_init=4.5,
            # The searcher can be initialised directly with an initial radius, rather than stdev
            center_learning_rate=max_speed/2.,
            stdev_learning_rate=0.1,  # stdev learning rate of 0.1 was used across all experiments
            optimizer="clipup",  # Using the ClipUp optimiser
            optimizer_config={
                'max_speed': max_speed,  # with the defined max speed
                'momentum': 0.9,  # and momentum fixed to 0.9
            })

        self.Logger = StdOutLogger(self.solver)
        self.Pickler = PicklingLogger(self.solver, interval=5, directory='new-Checkpoints/', prefix='', zfill=4, after_first_step=True,
                                      items_to_save=('best', 'pop_best', 'center', 'best_eval', 'worst_eval', 'median_eval', 'mean_eval', 'pop_best_eval'))
        self.Pandaer = PandasLogger(self.solver)

        self._robot_bodies = robot_bodies

        self._simulation_time = simulation_time
        self._sampling_frequency = sampling_frequency
        self._control_frequency = control_frequency



    def run(self, num_generations):

        for i in range(num_generations):
            self.solver.step()

            try:
                if (i+1) % 5 == 0 or i == 0:
                    url = f"https://api.telegram.org/bot{self.TOKEN}/sendMessage?chat_id={self.chat_id}&text=PGPE G number:{str(i+1)} Best:{str(self.solver.status['best_eval'])}"
                    print(requests.get(url).json())
            except:
                pass

        self.Pandaer.to_dataframe().to_pickle('new-Checkpoints/pgpeDataframe.pickle')
        print(self.solver.status)
