# Copyright 2026 German Aerospace Center (DLR)
# Institute Systems Engineering for Future Mobility (SE)
#
# Contributors:
#   - Thies de Graaff <thies.degraaff@dlr.de>
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


from typing import Dict, Optional, Tuple

import carla
import numpy as np
from pyquaternion import Quaternion

from cdr.utils.camera_utils import apply_transformation, pose_to_transformation_matrix


# Cache of the actor's BBs to avoid RPC calls. The cache is dependent on the world (first key) to avoid cache
# hits when actors are assigned the same ID as an actor had in a previous episode (unlikely, but safety first).
ACTOR_BB_CACHE: Dict[int, Dict[int, carla.BoundingBox]] = {}


def get_bounding_box(actor: carla.Actor) -> carla.BoundingBox:
    """
    Returns the bounding box of the actor (the same as using `actor.bounding_box`) but fixes two issues
    with the BBs of 2-wheeled vehicles:
    1. There seems to be a bug in CARLA (see Github Issues 5376, 5321, 4189, 3670) leading to sometimes corrupted
       bounding boxes for 2-wheeled vehicles (width is zero, location is wrong).
    2. Typical datasets include the rider in the bounding box, while the CARLA BB only includes the vehicle.

    Args:
        actor (carla.Actor): the actor whose bounding box shall be returned

    Returns:
        Optional[carla.BoundingBox]: the bounding box of the actor
    """
    # Use hardcoded bounding boxes for 2-wheeled vehicles, which then also include the rider.
    # no cover: start
    if actor.type_id == 'vehicle.bh.crossbike':
        actor_bb = carla.BoundingBox(carla.Location(x=0., y=0., z=0.8),
                                     carla.Vector3D(x=0.75, y=0.43, z=0.8))
    elif actor.type_id == 'vehicle.diamondback.century':
        actor_bb = carla.BoundingBox(carla.Location(x=0., y=0., z=0.91),
                                     carla.Vector3D(x=0.83, y=0.25, z=0.91))
    elif actor.type_id == 'vehicle.gazelle.omafiets':
        actor_bb = carla.BoundingBox(carla.Location(x=0., y=-0.01, z=0.9),
                                     carla.Vector3D(x=0.93, y=0.29, z=0.9))
    elif actor.type_id == 'vehicle.harley-davidson.low_rider':
        actor_bb = carla.BoundingBox(carla.Location(x=0., y=0., z=0.85),
                                     carla.Vector3D(x=1.18, y=0.38, z=0.825))
    elif actor.type_id == 'vehicle.kawasaki.ninja':
        actor_bb = carla.BoundingBox(carla.Location(x=-0.02, y=0., z=0.77),
                                     carla.Vector3D(x=1.02, y=0.4, z=0.75))
    elif actor.type_id == 'vehicle.vespa.zx125':
        actor_bb = carla.BoundingBox(carla.Location(x=0.01, y=0., z=0.82),
                                     carla.Vector3D(x=0.9, y=0.37, z=0.81))
    elif actor.type_id == 'vehicle.yamaha.yzf':
        actor_bb = carla.BoundingBox(carla.Location(x=0., y=0., z=0.8),
                                     carla.Vector3D(x=1.1, y=0.43, z=0.77))
    # no cover: stop
    else:
        # For all other actors, simply return their bounding box
        # We currently cache the bounding boxes of the actors, since accessing the bounding_box attribute
        # actually creates an RPC call to the CARLA server.
        # TODO Evaluate, if this is potentially dangerous due to opening/closing of doors.
        world_id = actor.get_world().id
        if world_id not in ACTOR_BB_CACHE:
            ACTOR_BB_CACHE[world_id] = {}

        if actor.id in ACTOR_BB_CACHE[world_id]:
            actor_bb = ACTOR_BB_CACHE[world_id][actor.id]
        else:
            actor_bb = actor.bounding_box  # this creates an RPC call
            ACTOR_BB_CACHE[world_id][actor.id] = actor_bb

    # We need to make a deep copy, since CARLA's `transform` functions actually do inplace modifications
    actor_bb = carla.BoundingBox(
        carla.Location(actor_bb.location.x, actor_bb.location.y, actor_bb.location.z),
        carla.Vector3D(actor_bb.extent.x, actor_bb.extent.y, actor_bb.extent.z)
    )
    return actor_bb


def get_global_3d_bounding_box(actor: carla.Actor, actor_snapshot: carla.ActorSnapshot)\
        -> Tuple[carla.Vector3D, carla.Rotation, np.ndarray]:
    """
    Returns the bounding box of the demanded actor in the global coordinate system.

    Args:
        actor (carla.Actor): the CARLA actor
        actor_snapshot (carla.ActorSnapshot): the actor snapshot from the interested time

    Returns:
        Tuple[np.ndarray, Quaternion, np.ndarray]: a tuple of global location, rotation and
            size (length, width, height) of the BB.
    """
    bb = get_bounding_box(actor)

    actor_transform = actor_snapshot.get_transform()
    bb_location_g = actor_transform.transform(bb.location)
    bb_rotation_g = actor_transform.rotation  # rotation of bounding boxes are always (0., 0., 0.)
    bb_size = np.array((bb.extent.x * 2, bb.extent.y * 2, bb.extent.z * 2))  # length, width, height

    return bb_location_g, bb_rotation_g, bb_size


def bounding_box_contains_points(bb_center: np.ndarray, bb_size: np.ndarray, bb_rotation: Quaternion,
                                 points: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """
    Checks for all points if they are contained by the given bounding box, returning a boolean mask with `len(points)`
    entries.

    It is assumed that the given bounding box is specified in the same coordinate system as the points are.
    I.e. if points are in global coordinates, the bounding box must be too. And vice versa.

    Args:
        bb_center (np.ndarray): the center point of the bounding box
        bb_size (np.ndarray): the total size of the bounding box (length, width, height)
        bb_rotation (Quaternion): the rotation of the bounding box
        points (np.ndarray): the points to test against the bounding box
        eps (float): epsilon to allow for boundary values to be accepted (robustness to floating point arithmetic)

    Returns:
        mask (np.ndarray): a boolean array of same length as `points`, being `True` if the respective point is contained
                           by the bounding box, and `False` otherwise
    """
    # The general idea is to transform all points such that they are relative to the bounding box.
    # Then, the contains-check reduces to simple axis-aligned boundary checks.

    # Transform the points relative to the bounding box
    global_to_bb_transform = np.linalg.inv(pose_to_transformation_matrix(bb_center, bb_rotation))
    points_local = apply_transformation(points, global_to_bb_transform, sample_axis=0)

    # Perform boundary checks
    return np.all(np.abs(points_local) <= bb_size[None, :] / 2 + eps, axis=-1)
