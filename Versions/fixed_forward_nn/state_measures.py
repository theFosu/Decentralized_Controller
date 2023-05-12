from revolve2.core.physics.actor import Actor, Joint, RigidBody
from pyrr import Quaternion, Vector3
from typing import List


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


def retrieve_joint_info(joint: Joint) -> List:

    return [joint.position.x, joint.position.y, joint.position.z,
            joint.orientation.x, joint.orientation.y, joint.orientation.z, joint.orientation.w]


def retrieve_body_info(body: RigidBody) -> List:

    return [body.position.x, body.position.y, body.position.z,
            body.orientation.x, body.orientation.y, body.orientation.z, body.orientation.w]

# TODO: add neighboring messages when decentralizing the controller
