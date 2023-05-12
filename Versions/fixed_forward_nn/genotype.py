from dataclasses import dataclass
from random import Random
from typing import List
import math

import numpy as np
import numpy.typing as npt

from revolve2.core.modular_robot import Body, ModularRobot
from brain.reinforcement_network import ReinforcementLearner
from brain.reinforcement_brain import ReinforcementBrain

# Database imports
from revolve2.core.database import Serializer, IncompatibleError
from revolve2.core.database.serializers import Ndarray1xnSerializer
from sqlalchemy.ext.asyncio.session import AsyncSession
from sqlalchemy.future import select
from sqlalchemy.ext.declarative import declarative_base
import sqlalchemy


@dataclass
class Genotype:
    weight_array: npt.NDArray[np.float_]


def random(rng: Random, robot_body: Body) -> Genotype:
    """
    Create a random starting point.

    :returns: Genotype containing array of weight for nn controlling robot brain"""

    # extract number of input and output nodes
    ninput, dof_ranges = make_io(robot_body)

    noutput = len(dof_ranges)

    # The model is not going to be used right away, but needs to be instantiated to assign its weights
    model = ReinforcementLearner(ninput, noutput)

    nprng = np.random.Generator(
        np.random.PCG64(rng.randint(0, 2 ** 63)))

    larray = 0

    for layer in model.net:
        if hasattr(layer, 'weight'):
            larray += len(layer.weight) + len(layer.bias)

    weights = nprng.standard_normal(larray)  # Mean: 0 SD: 1

    return Genotype(weights)


def develop(genotype: Genotype, robot_body: Body) -> ModularRobot:
    """From genotype (weights of the neural network) to phenotype(modular robot with body and brain)"""

    ninput, dof_ranges = make_io(robot_body)

    brain = ReinforcementBrain(genotype.weight_array, dof_ranges, ninput)

    return ModularRobot(robot_body, brain)


def make_io(robot_body: Body):
    """Makes the length of the input and the DOF ranges corresponding to the body"""

    ninput = ((len(robot_body.find_active_hinges()) + len(robot_body.find_bricks()) + 1) * 7) + 1

    dof_ranges = np.full(len(robot_body.find_active_hinges()), 1.0)

    return ninput, dof_ranges


def mutate(
    genotype: Genotype,
    rng: Random,
) -> Genotype:
    """Apply a small normally distributed noise to every weight"""

    nprng = np.random.Generator(
        np.random.PCG64(rng.randint(0, 2 ** 63)))

    mutations = nprng.standard_normal(size=genotype.weight_array.shape) * 0.1  # SD: 0.1 to avoid too little exploitation

    genotype.weight_array += mutations

    return genotype


def crossover(
    parent1: Genotype,
    parent2: Genotype,
    rng: Random,
) -> Genotype:

    nprng = np.random.Generator(
        np.random.PCG64(rng.randint(0, 2 ** 63)))

    parent1_weights = parent1.weight_array
    parent2_weights = parent2.weight_array

    choices = nprng.choice([True, False], size=parent1_weights.shape)

    child_weights = np.where(choices, parent1_weights, parent2_weights)

    return Genotype(child_weights)


class GenotypeSerializer(Serializer[Genotype]):
    @classmethod
    async def create_tables(cls, session: AsyncSession) -> None:
        await (await session.connection()).run_sync(DbGenotype.metadata.create_all)
        await Ndarray1xnSerializer.create_tables(session)

    @classmethod
    def identifying_table(cls) -> str:
        return DbGenotype.__tablename__

    @classmethod
    async def to_database(
        cls, session: AsyncSession, objects: List[Genotype]
    ) -> List[int]:

        array_ids = await Ndarray1xnSerializer.to_database(
            session, [o.weight_array for o in objects]
        )
        dbgenotypes = [
            DbGenotype(array_id=array_id)
            for array_id in array_ids
        ]
        session.add_all(dbgenotypes)
        await session.flush()
        ids = [
            dbfitness.id for dbfitness in dbgenotypes if dbfitness.id is not None
        ]  # cannot be none because not nullable. check if only there to silence mypy.
        assert len(ids) == len(objects)  # but check just to be sure
        return ids

    @classmethod
    async def from_database(
        cls, session: AsyncSession, ids: List[int]
    ) -> List[Genotype]:
        rows = (
            (await session.execute(select(DbGenotype).filter(DbGenotype.id.in_(ids))))
            .scalars()
            .all()
        )

        if len(rows) != len(ids):
            raise IncompatibleError()

        id_map = {t.id: t for t in rows}
        array_ids = [id_map[id].array_id for id in ids]  # and here

        arrays = await Ndarray1xnSerializer.from_database(
            session, array_ids
        )

        return [Genotype(array) for array in arrays]


DbBase = declarative_base()


class DbGenotype(DbBase):
    __tablename__ = "genotype"

    id = sqlalchemy.Column(
        sqlalchemy.Integer,
        nullable=False,
        unique=True,
        autoincrement=True,
        primary_key=True,
    )
    # Only element to store is the weight array
    array_id = sqlalchemy.Column(sqlalchemy.Integer, nullable=False)
