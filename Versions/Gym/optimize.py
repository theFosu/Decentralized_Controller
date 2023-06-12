from evotorch.algorithms import PGPE
from evotorch.logging import StdOutLogger
from evotorch.neuroevolution import VecGymNE

# Specialized Problem class for RL
problem = VecGymNE(
    env="Humanoid-v4",
    # Linear policy
    network="Linear(obs_length, act_length)",
    observation_normalization=True,
    decrease_rewards_by=5.0,
    # Use all available CPU cores
    num_actors="max",
)

searcher = PGPE(
    problem,
    popsize=200,
    center_learning_rate=0.01125,
    stdev_learning_rate=0.1,
    optimizer_config={"max_speed": 0.015},
    radius_init=0.27,
    num_interactions=150000,
    popsize_max=3200,
)
logger = StdOutLogger(searcher)
searcher.run(500)

population_center = searcher.status["center"]
policy = problem.to_policy(population_center)
problem.visualize(policy)
