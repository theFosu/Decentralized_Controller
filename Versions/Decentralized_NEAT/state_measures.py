from revolve2.core.physics.actor import Actor, Joint, RigidBody
from revolve2.core.modular_robot import ActiveHinge, Body
from pyrr import Quaternion, Vector3
from typing import List


def is_root(body: RigidBody) -> bool:
    """Checks whether the specified body is a root node of the actor graph for actuation"""
    if body.name == 'origin':
        return True
    return False


def is_leaf(actor: Actor, body: RigidBody) -> bool:
    """Checks whether the specified body is a leaf node of the actor graph for message-passing"""

    for joint in actor.joints:
        if joint.body1.name == body.name:
            return False
    return True


def retrieve_joint_info_from_actor(actor: Actor) -> List:

    positions = []

    for joint in actor.joints:

        positions.extend(retrieve_joint_info(joint))

    return positions


def retrieve_body_info_from_actor(actor: Actor) -> List:

    positions = []

    for body in actor.bodies:

        positions.extend(retrieve_body_info(body))

    return positions


def retrieve_extended_joint_info(joint: Joint) -> List:
    """Retrieve not only the joint's info but also its left body's."""

    info = retrieve_body_info(joint.body2)
    info.extend(retrieve_joint_info(joint))

    return info


def retrieve_joint_info(joint: Joint) -> List:

    return [joint.position.x, joint.position.y, joint.position.z,
            joint.orientation.x, joint.orientation.y, joint.orientation.z, joint.orientation.w]


def retrieve_body_info(body: RigidBody) -> List:
    com = body.center_of_mass()
    return [body.position.x, body.position.y, body.position.z,
            body.orientation.x, body.orientation.y, body.orientation.z, body.orientation.w,
            com.x, com.y, com.z]
