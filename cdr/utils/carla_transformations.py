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


from typing import Any, Callable, Dict, Tuple, Union

import carla

import numpy as np
from scipy.spatial.transform import Rotation
from pyquaternion import Quaternion

from .bounding_boxes import get_global_3d_bounding_box
from .carla_classes import blueprint_name_to_type


def carla_vector3d_to_array(vector: carla.Vector3D) -> np.ndarray:
    """
    Returns the given CARLA Vector3D as array.

    Args:
        vector (carla.Vector3D): a CARLA Vector3D

    Returns:
        vector (np.ndarray): the values represented by the vector as array
    """
    return np.array([vector.x, vector.y, vector.z])


def array_to_carla_vector3d(xyz: np.ndarray) -> carla.Vector3D:
    """
    Returns the given array as a CARLA Vector3D.

    Args:
        xyz (np.ndarray): a numpy array

    Returns:
        vector (carla.Vector3D): corresponding CARLA Vector3D
    """
    return carla.Vector3D(xyz[0], xyz[1], xyz[2])


def carla_location_to_array(location: carla.Location) -> np.ndarray:
    """
    Transforms the given CARLA Location from CARLA's left-handed coordinate system to a right-handed coordinate system
    and returns the (x, y, z) coordinate as array.

    Args:
        location (carla.Location): a CARLA Location in a left-handed coordinate system

    Returns:
       location (np.ndarray): corresponding (x, y, z) coordinate in a right-handed coordinate system
    """
    # invert y to align to CARLA coordinate system, which is left-handed
    return np.array([location.x, -location.y, location.z])


def array_to_carla_location(xyz: np.ndarray) -> carla.Location:
    """
    Transforms the given (x, y, z) coordinate from a right-handed coordinate system to CARLA's left-handed coordinate system
    and returns it as a CARLA Location.

    Args:
        xyz (np.ndarray): a (x, y, z) coordinate in a right-handed coordinate system

    Returns:
        location (carla.Location): corresponding CARLA Location in a left-handed coordinate system
    """
    # invert y to align to CARLA coordinate system, which is left-handed
    return carla.Location(xyz[0], -xyz[1], xyz[2])


def carla_rotation_to_quaternion(rotation: carla.Rotation, is_camera: bool) -> Quaternion:
    """
    Translates the given CARLA Rotation to a quaternion, while paying respect to different used coordinate systems.
    In general, the CARLA Rotation in a left-handed coordinate system is translated to a rotation in a right-handed
    coordinate system.

    Additionally, CARLA treats rotations for actors and sensors the same, i.e. the coordinate system is left-handed, where:
        x points to the front, y points to the right and z points upwards

    The literature (nearly) always use a right-handed coordinate system for cameras, where:
        x points to the right, y points downwards and z points to the front.

    If the Rotation belongs to a camera, specify this via `is_camera` so that the rotation is adjusted accordingly.

    Args:
        rotation (carla.Rotation): the CARLA Rotation in a left-handed coordinate system
        is_camera (bool): should be `True`, when the Rotation belongs to a camera

    Returns:
        rotation (Quaternion): a quaternion describing the rotation in a right-handed coordinate system
    """
    # Negate pitch and yaw to convert from left- to right-handed
    roll_pitch_yaw = np.array([rotation.roll, -rotation.pitch, -rotation.yaw])
    xyzw = Rotation.from_euler('xyz', roll_pitch_yaw, degrees=True).as_quat()
    quaternion = Quaternion(xyzw[[3, 0, 1, 2]])
    if is_camera:
        # While the literature (nearly) always use local camera coordinate systems, where
        #   x points to the right, y points downwards and z points to the front,
        # CARLA uses a camera coordinate system, where
        #   x points to the front, y points to the right and z points upwards
        # Adjust this by performing two rotations
        quaternion = quaternion * Quaternion(axis=[1., 0., 0.], degrees=-90) * Quaternion(axis=[0., 1., 0.], degrees=90)
    return quaternion


def quaternion_to_carla_rotation(quaternion: Quaternion, is_camera: bool) -> carla.Rotation:
    """
    Translates the given quaternion to a CARLA Rotation, while paying respect to different used coordinate systems.
    In general, the quaternion in a right-handed coordinate system is translated to a CARLA Rotation in a left-handed
    coordinate system.

    Additionally, CARLA treats rotations for actors and sensors the same, i.e. the coordinate system is left-handed, where:
        x points to the front, y points to the right and z points upwards

    The literature (nearly) always use a right-handed coordinate system for cameras, where:
        x points to the right, y points downwards and z points to the front.

    If the quaternion belongs to a camera, specify this via `is_camera` so that the rotation is adjusted accordingly.

    Args:
        quaternion (Quaternion): the quaternion in a right-handed coordinate system
        is_camera (bool): should be `True`, when the quaternion belongs to a camera

    Returns:
        rotation (carla.Rotation): a CARLA Rotation describing the rotation in CARLA's left-handed coordinate system
    """
    if is_camera:
        # While the literature (nearly) always use local camera coordinate systems, where
        #   x points to the right, y points downwards and z points to the front,
        # CARLA uses a camera coordinate system, where
        #   x points to the front, y points to the right and z points upwards
        # Adjust this by performing two rotations
        quaternion = quaternion * Quaternion(axis=[0., 1., 0.], degrees=-90) * Quaternion(axis=[1., 0., 0.], degrees=90)
    roll_pitch_yaw = Rotation.from_quat(quaternion.elements[[1, 2, 3, 0]]).as_euler('xyz', degrees=True)
    # Negate pitch and yaw to convert from right- to left-handed
    return carla.Rotation(roll=roll_pitch_yaw[0], pitch=-roll_pitch_yaw[1], yaw=-roll_pitch_yaw[2])


def decode_instseg_ids(channel_green: np.ndarray, channel_blue: np.ndarray) -> np.ndarray:
    """
    Decodes the instance segmentation IDs.

    Args:
        channel_green (np.ndarray): the green channel of CARLA's instance segmentation
        channel_blue (np.ndarray): the blue channel of CARLA's instance segmentation

    Returns:
        instseg_ids (np.ndarray): the decoded instance segmentation IDs
    """
    return channel_green + 256 * channel_blue.astype(np.uint16)


def encode_instseg_ids(instseg_ids: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Encodes the given instance segmentation IDs in CARLA's style (green and blue channel).

    Args:
        instseg_ids (np.ndarray): the instance segmentation IDs

    Returns:
        instseg_ids_encoded (Tuple[np.ndarray, np.ndarray]): the green and blue channel that encode the
        instance segmentation IDs
    """
    return (instseg_ids % 256).astype(np.uint8), (instseg_ids // 256).astype(np.uint8)


def image_to_array(carla_image: carla.Image) -> np.ndarray:
    """
    Converts a carla.Image object to a numpy array.

    Args:
        carla_image (carla.Image): carla.Image object returned by camera-based sensors

    Returns:
        image_array (np.ndarray): the corresponding numpy array
    """
    image_array = np.frombuffer(carla_image.raw_data, dtype=np.uint8)
    image_array = np.reshape(image_array, (carla_image.height, carla_image.width, 4))
    image_array = image_array.copy()
    image_array = image_array[:, :, :3]
    image_array = image_array[:, :, ::-1]
    return image_array


def image_to_dict(image: carla.Image) -> Dict[str, Any]:
    """
    Converts the information in the given carla.Image object to a dictionary.

    Args:
        image (carla.Image): carla.Image to convert

    Returns:
        image (Dict[str, Any]): the resulting dictionary containing the information of the carla.Image
    """
    return {
        'frame_id': image.frame,
        'timestamp': image.timestamp,
        'image_array': image_to_array(image),
        'location': carla_location_to_array(image.transform.location),
        'rotation': carla_rotation_to_quaternion(image.transform.rotation, is_camera=True)
    }


def point_cloud_to_dict(point_cloud: Union[carla.LidarMeasurement, carla.SemanticLidarMeasurement]) -> Dict[str, Any]:
    """
    Converts the information in the given carla.LidarMeasurement or carla.SemanticLidarMeasurement
    object to a dictionary.

    Args:
        point_cloud (Union[carla.LidarMeasurement, carla.SemanticLidarMeasurement]):
            carla.LidarMeasurement or carla.SemanticLidarMeasurement to convert

    Returns:
        point_cloud (Dict[str, Any]): the resulting dictionary containing the information of the
        carla.LidarMeasurement or carla.SemanticLidarMeasurement
    """
    point_cloud_dict: Dict[str, Any] = {
        'frame_id': point_cloud.frame,
        'timestamp': point_cloud.timestamp,
        'location': carla_location_to_array(point_cloud.transform.location),
        'rotation': carla_rotation_to_quaternion(point_cloud.transform.rotation, is_camera=False)
    }
    if isinstance(point_cloud, carla.LidarMeasurement):
        pc_raw = np.frombuffer(point_cloud.raw_data, np.float32).reshape((-1, 4)).copy()
        point_cloud_dict['vertex_attributes'] = {'intensity': pc_raw[:, -1]}
    elif isinstance(point_cloud, carla.SemanticLidarMeasurement):
        pc_raw = np.frombuffer(point_cloud.raw_data, np.float32).reshape((-1, 6)).copy()
        point_cloud_dict['vertex_attributes'] = {'cosine_incident_angle': pc_raw[:, 3],
                                                 'actor_id': pc_raw[:, 4].view(np.uint32),
                                                 'semseg_id': pc_raw[:, 5].view(np.uint32)}
    else:
        raise NotImplementedError(f'Point clouds of type {type(point_cloud)} are currently not supported.')

    points = pc_raw[:, :3]
    points[:, 1] *= -1  # invert y-axis to go from left-handed coordinates to right-handed
    point_cloud_dict['vertices'] = points

    return point_cloud_dict


def gnss_to_dict(gnss_data: carla.GnssMeasurement) -> Dict[str, Any]:
    """
    Converts the information in the given carla.GnssMeasurement object to a dictionary.

    Args:
        gnss_data (carla.GnssMeasurement): carla.GnssMeasurement to convert

    Returns:
        Dict[str, Any]: the resulting dictionary containing the information of the carla.GnssMeasurement
    """
    return {
        'frame_id': gnss_data.frame,
        'timestamp': gnss_data.timestamp,
        'location': carla_location_to_array(gnss_data.transform.location),
        'rotation': carla_rotation_to_quaternion(gnss_data.transform.rotation, is_camera=False),
        'altitude': gnss_data.altitude,
        'latitude': gnss_data.latitude,
        'longitude': gnss_data.longitude
    }


def imu_to_dict(imu_data: carla.IMUMeasurement) -> Dict[str, Any]:
    """
    Converts the information in the given carla.IMUMeasurement object to a dictionary.

    Args:
        gnss_data (carla.IMUMeasurement): carla.IMUMeasurement to convert

    Returns:
        Dict[str, Any]: the resulting dictionary containing the information of the carla.IMUMeasurement
    """
    acceleration = carla_vector3d_to_array(imu_data.accelerometer)
    acceleration[1] *= -1  # left-hand to right-hand coordinate system
    angular_velocity = carla_vector3d_to_array(imu_data.gyroscope)
    angular_velocity[[0, 2]] *= -1  # left-hand to right-hand coordinate system
    return {
        'frame_id': imu_data.frame,
        'timestamp': imu_data.timestamp,
        'location': carla_location_to_array(imu_data.transform.location),
        'rotation': carla_rotation_to_quaternion(imu_data.transform.rotation, is_camera=False),
        'acceleration': acceleration,
        'angular_velocity': angular_velocity,
        'compass': imu_data.compass
    }


def world_snapshot_to_dict(world_snapshot: carla.WorldSnapshot, world: carla.World) -> Dict[str, Any]:
    """
    Converts the information in the given carla.WorldSnapshot object to a dictionary.

    Args:
        world_snapshot (carla.WorldSnapshot): carla.WorldSnapshot to convert
        world (carla.World): carla.World instance

    Returns:
        world_snapshot (Dict[str, Any]): the resulting dictionary containing the information of the carla.WorldSnapshot
    """
    all_actors = world.get_actors()
    actors_items = {}
    for actor_snapshot in world_snapshot:
        actor = all_actors.find(actor_snapshot.id)
        if actor is None:
            raise ValueError(f'Failed to find actor with ID "{actor_snapshot.id}" in the world.')

        if actor.type_id == 'static.prop.mesh':
            # When static.prop.mesh is spawned as substitute for an environment vehicle, the actual type_id is stored
            # in the role_name.
            actor_type_id = actor.attributes['role_name'] if 'role_name' in actor.attributes else 'static.prop.mesh'
            actor_role_name = 'parking'
        else:
            actor_type_id = actor.type_id
            actor_role_name = actor.attributes['role_name'] if 'role_name' in actor.attributes else ''

        # Only collect the dynamic actors
        if not (actor_type_id.startswith('vehicle.') or actor_type_id.startswith('walker.')):
            continue

        # Get global 3D bounding box for the actor
        bb = get_global_3d_bounding_box(actor, actor_snapshot)
        assert bb is not None  # Holds, since we already filtered for vehicles and walkers (just for pylint)
        bb_loc_g, bb_rot_g, bb_size_lwh = bb

        # Function to transform other local vectors to global ones
        actor_transform = actor_snapshot.get_transform()
        def transform_to_global(vector: carla.Vector3D) -> np.ndarray:
            return carla_vector3d_to_array(actor_transform.transform_vector(vector))

        # The instance segmentation only supports 16-bit IDs, so we also wrap the actor ID by 16-bits.
        # This is not a real problem, since at least for our experiments, CARLA is not able to simulate more than
        # 2**16 actors simultaneously without crashing.
        actor_id = actor.id % 2**16
        actors_items[actor_id] = {
            'class_id': blueprint_name_to_type(actor_type_id).name.lower(),
            'center': carla_location_to_array(bb_loc_g),
            'size': bb_size_lwh,
            'rot': carla_rotation_to_quaternion(bb_rot_g, is_camera=False),
            'velocity': transform_to_global(actor_snapshot.get_velocity()),
            'acceleration': transform_to_global(actor_snapshot.get_acceleration()),
            'angular_velocity': transform_to_global(actor_snapshot.get_angular_velocity()),
            'type_id': actor_type_id,
            'role_name': actor_role_name
        }
    actors_items = dict(sorted(actors_items.items()))

    return {
        'frame_id': world_snapshot.frame,
        'timestamp': world_snapshot.timestamp.elapsed_seconds,
        'actors': actors_items
    }


def get_sensor_to_dict_function(sensor_blueprint: str) -> Callable[[carla.SensorData], Dict[str, Any]]:
    """
    For the given sensor blueprint, this function returns a function that transforms the sensor's data type into a
    dictionary of pickable data types, which can be passed into queues.

    Args:
        sensor_blueprint (str): the sensor blueprint name (e.g. sensor.camera.rgb)

    Raises:
        NotImplementedError: if the given sensor blueprint is not supported yet

    Returns:
        transform_func (Callable[[carla.SensorData], Dict[str, Any]]):
        a function that transforms `carla.SensorData` into a dictionary of pickable data types
        for the given sensor blueprint
    """
    if sensor_blueprint.startswith('sensor.camera'):
        return image_to_dict
    elif sensor_blueprint == 'sensor.lidar.ray_cast':
        return point_cloud_to_dict
    elif sensor_blueprint == 'sensor.lidar.ray_cast_semantic':
        return point_cloud_to_dict
    elif sensor_blueprint == 'sensor.other.gnss':
        return gnss_to_dict
    elif sensor_blueprint == 'sensor.other.imu':
        return imu_to_dict
    else:
        raise NotImplementedError(f'Sensor type "{sensor_blueprint}" not implemented yet.')
