from revolve2.core.optimization.ea.generic_ea import EAOptimizer
import revolve2.core.optimization.ea.generic_ea.population_management as population_management
import revolve2.core.optimization.ea.generic_ea.selection as selection
from revolve2.core.database.serializers import FloatSerializer
from revolve2.actor_controller import ActorController
from genotype import *

from revolve2.core.modular_robot import Body
from revolve2.core.physics.environment_actor_controller import EnvironmentActorController
from revolve2.core.physics.running import ActorState, ActorControl, Runner, Batch, Environment, PosedActor
from revolve2.runners.mujoco import LocalRunner
from standard_resources import terrains
from pyrr import Vector3, Quaternion

from sqlalchemy.ext.asyncio.session import AsyncSession
from sqlalchemy.ext.asyncio import AsyncEngine
import sqlalchemy
from sqlalchemy.ext.declarative import declarative_base
from revolve2.core.optimization import DbId

from random import Random
import pickle
from typing import List, Tuple


class Optimizer(EAOptimizer[Genotype, float]):

    _TERRAIN = terrains.flat()
    _db_id: DbId
    _rng: Random
    _runner: Runner
    _robot_body: Body
    _num_generations: int
    _simulation_time: int
    _sampling_frequency: float
    _control_frequency: float

    async def _evaluate_generation(
        self,
        genotypes: List[Genotype],
        database: AsyncEngine,
        db_id: DbId,
    ) -> List[float]:
        """Here both the evaluation and the entirety of the simulation (simple optimization main.py) is performed"""

        batch = Batch(
            simulation_time=self._simulation_time,
            sampling_frequency=self._sampling_frequency,
            control_frequency=self._control_frequency,
            simulation_timestep=0.001
        )

        for genotype in genotypes:
            actor, controller = develop(genotype, self._robot_body).make_actor_and_controller()
            bounding_box = actor.calc_aabb()
            env = Environment(EnvironmentActorController(controller))
            env.static_geometries.extend(self._TERRAIN.static_geometry)
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

        batch_results = await self._runner.run_batch(batch)

        return [
            self._calculate_fitness(
                environment_result.environment_states[0].actor_states[0],
                environment_result.environment_states[-1].actor_states[0],
            )
            for environment_result in batch_results.environment_results
        ]

    '''def _control(self, environment_index: int, dt: float, control: ActorControl) -> None:
        """Get the right controller and step (helper function)"""
        controller = self._controllers[environment_index]
        controller.step(dt)
        control.set_dof_targets(0, controller.get_dof_targets())'''

    def _select_parents(
        self,
        population: List[Genotype],
        fitnesses: List[float],
        num_parent_groups: int,
    ) -> List[List[int]]:
        return [
            selection.multiple_unique(
                2,
                population,
                fitnesses,
                lambda _, fitnesses: selection.tournament(self._rng, fitnesses, k=2),
            )
            for _ in range(num_parent_groups)
        ]

    def _select_survivors(
        self,
        old_individuals: List[Genotype],
        old_fitnesses: List[float],
        new_individuals: List[Genotype],
        new_fitnesses: List[float],
        num_survivors: int,
    ) -> Tuple[List[int], List[int]]:
        return population_management.steady_state(
            old_individuals,
            old_fitnesses,
            new_individuals,
            new_fitnesses,
            lambda n, genotypes, fitnesses: selection.multiple_unique(
                n,
                genotypes,
                fitnesses,
                lambda genotypes, fitnesses: selection.tournament(
                    self._rng, fitnesses, k=2
                ),
            ),
        )

    def _crossover(self, parents: List[Genotype]) -> Genotype:
        assert len(parents) == 2
        return crossover(parents[0], parents[1], self._rng)

    def _mutate(self, genotype: Genotype) -> Genotype:
        return mutate(genotype, self._rng)

    def _must_do_next_gen(self) -> bool:
        """Automatically stop whenever pre-established maximum number of generations is reached"""
        return self.generation_index != self._num_generations

    @staticmethod
    def _calculate_fitness(begin_state: ActorState, end_state: ActorState) -> float:
        # distance traveled on the xy plane
        return float(
            math.sqrt(
                (begin_state.position[0] - end_state.position[0]) ** 2
                + ((begin_state.position[1] - end_state.position[1]) ** 2)
            )
        )

    async def ainit_new(
            self,
            database: AsyncEngine,
            session: AsyncSession,
            db_id: DbId,
            offspring_size: int,
            initial_population: List[Genotype],
            rng: Random,
            robot_body: Body,
            num_generations: int,
            simulation_time: int,
            sampling_frequency: float,
            control_frequency: float,
    ) -> None:
        await super().ainit_new(
            database=database,
            session=session,
            db_id=db_id,
            genotype_type=Genotype,
            genotype_serializer=GenotypeSerializer,
            fitness_type=float,
            fitness_serializer=FloatSerializer,
            offspring_size=offspring_size,
            initial_population=initial_population,
        )

        self._db_id = db_id
        self._init_runner()
        self._rng = rng
        self._robot_body = robot_body
        self._num_generations = num_generations
        self._simulation_time = simulation_time
        self._sampling_frequency = sampling_frequency
        self._control_frequency = control_frequency

        # create database structure if not exists
        await (await session.connection()).run_sync(DbBase.metadata.create_all)

        # save to database
        self._on_generation_checkpoint(session)

    async def ainit_from_database(  # type: ignore # see comment at ainit_new
        self,
        database: AsyncEngine,
        session: AsyncSession,
        db_id: DbId,
        rng: Random,
        robot_body: Body,
    ) -> bool:
        if not await super().ainit_from_database(
                database=database,
                session=session,
                db_id=db_id,
                genotype_type=Genotype,
                genotype_serializer=GenotypeSerializer,
                fitness_type=float,
                fitness_serializer=FloatSerializer,
        ):
            return False

        self._db_id = db_id
        self._init_runner()
        self._robot_body = robot_body

        opt_row = (
            (
                await session.execute(
                    select(DbOptimizerState)
                    .filter(DbOptimizerState.db_id == db_id.fullname)
                    .order_by(DbOptimizerState.generation_index.desc())
                )
            )
            .scalars()
            .first()
        )

        # if this happens something is wrong with the database
        if opt_row is None:
            raise IncompatibleError

        # More assignments here!

        self._rng = rng
        self._rng.setstate(pickle.loads(opt_row.rng))
        self._simulation_time = opt_row.simulation_time
        self._sampling_frequency = opt_row.sampling_frequency
        self._control_frequency = opt_row.control_frequency
        self._num_generations = opt_row.num_generations

        return True

    def _init_runner(self) -> None:
        self._runner = LocalRunner(headless=True)

    def _on_generation_checkpoint(self, session: AsyncSession) -> None:
        session.add(
            DbOptimizerState(
                db_id=self._db_id.fullname,
                generation_index=self.generation_index,
                rng=pickle.dumps(self._rng.getstate()),
                simulation_time=self._simulation_time,
                sampling_frequency=self._sampling_frequency,
                control_frequency=self._control_frequency,
                num_generations=self._num_generations,
            )
        )


DbBase = declarative_base()


class DbOptimizerState(DbBase):
    __tablename__ = "optimizer"

    db_id = sqlalchemy.Column(
        sqlalchemy.String,
        nullable=False,
        primary_key=True,
    )
    generation_index = sqlalchemy.Column(
        sqlalchemy.Integer, nullable=False, primary_key=True
    )
    rng = sqlalchemy.Column(sqlalchemy.PickleType, nullable=False)
    simulation_time = sqlalchemy.Column(sqlalchemy.Integer, nullable=False)
    sampling_frequency = sqlalchemy.Column(sqlalchemy.Float, nullable=False)
    control_frequency = sqlalchemy.Column(sqlalchemy.Float, nullable=False)
    num_generations = sqlalchemy.Column(sqlalchemy.Integer, nullable=False)
