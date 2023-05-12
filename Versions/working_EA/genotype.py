from dataclasses import dataclass
from random import Random
from typing import List
import math

import numpy as np
import numpy.typing as npt

from revolve2.core.modular_robot import Body, ModularRobot
from revolve2.core.modular_robot.brains import BrainCpgNetworkStatic, make_cpg_network_structure_neighbour
from revolve2.actor_controllers.cpg import CpgNetworkStructure

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

    :returns: Genotype containing fully formed robot brain"""

    # extract cpg structure
    cpg_network_structure = make_cgp_network(robot_body)

    # Actual random weights for active hinges
    nprng = np.random.Generator(
        np.random.PCG64(rng.randint(0, 2 ** 63)))

    weights = nprng.standard_normal(cpg_network_structure.num_connections)  # Mean: 0 SD: 1

    return Genotype(weights)


def develop(genotype: Genotype, robot_body: Body) -> ModularRobot:
    """From genotype to phenotype"""

    cpg_network_structure = make_cgp_network(robot_body)
    weight_matrix = cpg_network_structure.make_connection_weights_matrix_from_params(genotype.weight_array)

    brain = make_brain(cpg_network_structure, weight_matrix)

    return ModularRobot(robot_body, brain)


def make_cgp_network(robot_body: Body) -> CpgNetworkStructure:
    """Finds active hinges and makes cpg network structure out of it"""

    _, dof_ids = robot_body.to_actor()
    active_hinges_unsorted = robot_body.find_active_hinges()
    active_hinge_map = {
        active_hinge.id: active_hinge for active_hinge in active_hinges_unsorted
    }
    active_hinges = [active_hinge_map[dof_id] for dof_id in dof_ids]

    return make_cpg_network_structure_neighbour(active_hinges)


def make_brain(cpg_network_structure: CpgNetworkStructure, weight_matrix: npt.NDArray[np.float_],
               ) -> BrainCpgNetworkStatic:

    initial_state = cpg_network_structure.make_uniform_state(0.5 * math.pi / 2.0)

    dof_ranges = cpg_network_structure.make_uniform_dof_ranges(1.0)

    brain = BrainCpgNetworkStatic(
        initial_state,
        cpg_network_structure.num_cpgs,
        weight_matrix,
        dof_ranges,
    )

    return brain


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
