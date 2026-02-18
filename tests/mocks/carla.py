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


import re
import uuid
from typing import Callable, Dict, List, Optional

import carla
import numpy as np


SENSOR_BP_NAMES_W_ATTR = {
    'sensor.camera.rgb':
     {'sensor_tick': None, 'image_size_x': '800', 'image_size_y': '600', 'fov': '90.'},
    'sensor.camera.semantic_segmentation':
     {'sensor_tick': None, 'image_size_x': '800', 'image_size_y': '600', 'fov': '90.'},
    'sensor.camera.instance_segmentation':
     {'sensor_tick': None, 'image_size_x': '800', 'image_size_y': '600', 'fov': '90.'},
    'sensor.camera.depth':
     {'sensor_tick': None, 'image_size_x': '800', 'image_size_y': '600', 'fov': '90.'},
    'sensor.lidar.ray_cast':
     {'sensor_tick': None, 'channels': '32', 'range': '10.', 'points_per_second': '56000',
      'rotation_frequency': '10.', 'upper_fov': '10.', 'lower_fov': '-30.', 'horizontal_fov': '360.'},
    'sensor.lidar.ray_cast_semantic':
     {'sensor_tick': None,  'channels': '32', 'range': '10.', 'points_per_second': '56000',
      'rotation_frequency': '10.', 'upper_fov': '10.', 'lower_fov': '-30.', 'horizontal_fov': '360.'},
    'sensor.other.gnss':
    {'sensor_tick': None, 'noise_alt_stddev': '0.', 'noise_lat_stddev': '0.', 'noise_lon_stddev': '0.'},
    'sensor.other.imu':
    {'sensor_tick': None, 'noise_accel_stddev_x': '0.', 'noise_accel_stddev_y': '0.', 'noise_accel_stddev_z': '0.',
     'noise_gyro_stddev_x': '0.', 'noise_gyro_stddev_y': '0.', 'noise_gyro_stddev_z': '0.'}
}


##### GENERAL MOCKS #####


class ActorMock:

    def __init__(self, world: 'WorldMock', actor_id: int, type_id: str, attributes: Optional[Dict[str, str]] = None,
                 transform: carla.Transform = carla.Transform(), bounding_box: carla.BoundingBox = carla.BoundingBox()):
        self.world = world
        self.id = actor_id
        self.type_id = type_id
        self.attributes = {} if attributes is None else attributes
        self.transform = transform
        self.bounding_box = bounding_box

    def get_world(self):
        return self.world

    def destroy(self):
        del self.world.actors[self.id]


class SensorMock(ActorMock):

    def __init__(self, world: 'WorldMock', actor_id: int, type_id: str, attributes: Optional[Dict[str, str]] = None,
                 transform: carla.Transform = carla.Transform(), bounding_box: carla.BoundingBox = carla.BoundingBox()):
        super().__init__(world, actor_id, type_id, attributes, transform, bounding_box)

        self.listen_callback = None

    def listen(self, callback: Callable[['SensorDataMock'], None]):
        self.listen_callback = callback

    def destroy(self):
        self.listen_callback = None
        super().destroy()

    def send_data(self, frame: int, timestamp: float, transform: carla.Transform):
        if self.listen_callback is None:  # pragma: no cover
            return

        if self.type_id.startswith('sensor.camera'):
            width, height = int(self.attributes['image_size_x']), int(self.attributes['image_size_y'])
            data = ImageMock(np.zeros((width, height, 4), np.uint8),
                             frame, timestamp, transform)
        elif self.type_id == 'sensor.lidar.ray_cast':
            num_points = int(int(self.attributes['points_per_second']) * float(self.attributes['sensor_tick']))
            data = LidarMeasurementMock(np.zeros((num_points, 3), np.float32), np.zeros(num_points, np.float32),
                                        frame, timestamp, transform)
        elif self.type_id == 'sensor.lidar.ray_cast_semantic':
            num_points = int(int(self.attributes['points_per_second']) * float(self.attributes['sensor_tick']))
            data = SemanticLidarMeasurementMock(np.zeros((num_points, 3), np.float32), np.zeros(num_points, np.float32),
                                                np.zeros(num_points, np.uint32), np.zeros(num_points, np.uint32),
                                                frame, timestamp, transform)
        elif self.type_id == 'sensor.other.gnss':
            data = GNSSMeasurementMock(0.000001, 0.000001, 0.2,
                                       frame, timestamp, transform)
        elif self.type_id == 'sensor.other.imu':
            data = IMUMeasurementMock(carla.Vector3D(0., 0., 0.), carla.Vector3D(0., 0., 0.), 0.,
                                      frame, timestamp, transform)
        else:
            raise NotImplementedError()
        self.listen_callback(data)


class ActorListMock(List[ActorMock]):

    def filter(self, wildcard_pattern: str) -> List[ActorMock]:
        pattern = re.compile(wildcard_pattern.replace('.', r'\.').replace('*', '.*'))
        actors = []
        for actor in self:
            if pattern.match(actor.type_id):
                actors.append(actor)
        return actors

    def find(self, actor_id: int):
        for actor in self:
            if actor.id == actor_id:
                return actor
        return None


class ActorBlueprintMock:

    def __init__(self, id: str, attributes: Optional[Dict[str, str]] = None):
        self.id = id
        self.attributes = attributes or {}

    def has_attribute(self, id: str) -> bool:
        return id in self.attributes.keys()

    def set_attribute(self, id: str, value: str):
        self.attributes[id] = value


class BlueprintLibraryMock:

    def __init__(self):
        self.blueprints = {bp_name: ActorBlueprintMock(bp_name, bp_attr) for bp_name, bp_attr in SENSOR_BP_NAMES_W_ATTR.items()}

    def find(self, id: str) -> ActorBlueprintMock:
        return self.blueprints[id]


class MapMock:

    def __init__(self, name: str):
        self.name = name


class ActorSnapshotMock:

    def __init__(self, actor_id: int,
                 transform: carla.Transform = carla.Transform(),
                 velocity: carla.Vector3D = carla.Vector3D(),
                 acceleration: carla.Vector3D = carla.Vector3D(),
                 angular_velocity: carla.Vector3D = carla.Vector3D()):
        self.id = actor_id
        self.transform = transform
        self.velocity = velocity
        self.acceleration = acceleration
        self.angular_velocity = angular_velocity

    def get_transform(self) -> carla.Transform:
        return self.transform

    def get_velocity(self) -> carla.Vector3D:
        return self.velocity

    def get_acceleration(self) -> carla.Vector3D:
        return self.acceleration

    def get_angular_velocity(self) -> carla.Vector3D:
        return self.angular_velocity


class TimestampMock:

    def __init__(self, frame: int, elapsed_seconds: float):
        self.frame = frame
        self.elapsed_seconds = elapsed_seconds


class WorldSnapshotMock:

    def __init__(self, frame: int, timestamp: TimestampMock, actor_snapshots: List[ActorSnapshotMock]):
        self.frame = frame
        self.timestamp = timestamp
        self.actor_snapshots = actor_snapshots

    def __iter__(self):
        for actor_snapshot in self.actor_snapshots:
            yield actor_snapshot


class EnvironmentObjectMock:

    def __init__(self, id: int, name: str, type: carla.CityObjectLabel, bounding_box = carla.BoundingBox()):
        self.id = id
        self.name = name
        self.type = type
        self.bounding_box = bounding_box


class WorldMock:

    def __init__(self, init_frame: int = 0, init_timestamp: float = 0.):
        self.id = uuid.uuid4().int
        self.actors = {}
        self.frame = init_frame
        self.elapsed_seconds = init_timestamp
        self.fixed_delta_seconds = 0.05

        self.on_tick_callbacks = {}
        self.snapshot = None

    def get_actor(self, actor_id: int) -> Optional[ActorMock]:
        return self.actors.get(actor_id, None)

    def get_actors(self) -> List[ActorMock]:
        return ActorListMock(self.actors.values())

    def spawn_actor(self, blueprint: ActorBlueprintMock, transform: carla.Transform,
                    attach_to: Optional[ActorMock] = None, attachment: carla.AttachmentType = carla.AttachmentType.Rigid):
        actor_id = int(np.max(list(self.actors.keys()) or 0) + 1)
        if blueprint.id in SENSOR_BP_NAMES_W_ATTR.keys():
            actor = SensorMock(self, actor_id, blueprint.id, blueprint.attributes, transform)
        else:
            actor = ActorMock(self, actor_id, blueprint.id, blueprint.attributes, transform)
        self.actors[actor_id] = actor
        return actor

    def get_environment_objects(self, object_type: carla.CityObjectLabel):
        return []

    def get_settings(self):
        return carla.WorldSettings()

    def apply_settings(self, settings: carla.WorldSettings):
        self.fixed_delta_seconds = settings.fixed_delta_seconds

    def get_weather(self) -> carla.WeatherParameters:
        return carla.WeatherParameters.Default

    def get_map(self) -> MapMock:
        return MapMock('test')

    def get_blueprint_library(self):
        return BlueprintLibraryMock()

    def tick(self, seconds: float = 10.0, invalid: bool = False) -> int:
        # Advance world by one frame
        self.frame += 1
        self.elapsed_seconds += self.fixed_delta_seconds

        # Create snapshots of new frame
        actor_snapshots = [ActorSnapshotMock(actor.id, actor.transform) for actor in self.actors.values()]
        offset = 1 if invalid else 0
        world_snapshot = WorldSnapshotMock(self.frame + offset, TimestampMock(self.frame, self.elapsed_seconds),
                                           actor_snapshots)
        self.snapshot = world_snapshot

        # Send data for all sensors
        for actor in self.actors.values():
            if isinstance(actor, SensorMock):
                actor.send_data(self.frame, self.elapsed_seconds, actor.transform)

        # Call on_tick callbacks with new snapshot
        for callback in list(self.on_tick_callbacks.values()):
            callback(world_snapshot)

        return self.frame

    def invalid_tick(self):
        """
        This custom function can be used to send inconsistent data from the world to the listeners, useful during
        testing.
        The data is inconsistent by sending a larger tick value in the WorldSnapshot compared to the ticks part of the
        sensor data.
        """
        self.tick(invalid=True)

    def on_tick(self, callback):
        callback_id = int(np.max(list(self.on_tick_callbacks.keys()) or 0) + 1)
        self.on_tick_callbacks[callback_id] = callback
        return callback_id

    def remove_on_tick(self, callback_id: int):
        del self.on_tick_callbacks[callback_id]

    def get_snapshot(self):
        assert self.snapshot is not None
        return self.snapshot

    def set_annotations_traverse_translucency(self, enable: bool): pass


class TrafficManagerMock:

    def __init__(self): pass

    def set_synchronous_mode(self, enable: bool): pass


class ClientMock:

    def __init__(self, host: str, port: int):
        self.world = WorldMock()
        self.trafficmanager = TrafficManagerMock()
        self.client_version = '0.9.16'

    def set_timeout(self, timeout: float): pass

    def get_world(self):
        return self.world

    def get_trafficmanager(self, port: int):
        return self.trafficmanager

    def get_client_version(self):
        return self.client_version


##### SENSOR DATA MOCKS #####


class SensorDataMock:

    def __init__(self, frame: int, timestamp: float, transform: carla.Transform):
        self.frame = frame
        self.timestamp = timestamp
        self.transform = transform


class ImageMock(SensorDataMock):

    def __init__(self, data: np.ndarray,
                 frame: int, timestamp: float, transform: carla.Transform):
        super().__init__(frame, timestamp, transform)
        self.raw_data = data.tobytes('C')
        self.height = data.shape[0]
        self.width = data.shape[1]


class LidarMeasurementMock(SensorDataMock):

    def __init__(self, xyz: np.ndarray, intensity: np.ndarray,
                 frame: int, timestamp: float, transform: carla.Transform):
        super().__init__(frame, timestamp, transform)
        data = np.concatenate((xyz, intensity[:, None]), axis=-1, dtype=np.float32)
        self.raw_data = data.tobytes('C')


class SemanticLidarMeasurementMock(SensorDataMock):

    def __init__(self, xyz: np.ndarray, cosine_incident_angle: np.ndarray, actor_id: np.ndarray, semseg_id: np.ndarray,
                 frame: int, timestamp: float, transform: carla.Transform):
        super().__init__(frame, timestamp, transform)
        data = np.concatenate((xyz, cosine_incident_angle[:, None],
                               actor_id[:, None].view(np.float32),
                               semseg_id[:, None].view(np.float32)), axis=-1, dtype=np.float32)
        self.raw_data = data.tobytes('C')


class GNSSMeasurementMock(SensorDataMock):

    def __init__(self, latitude: float, longitude: float, altitude: float,
                 frame: int, timestamp: float, transform: carla.Transform):
        super().__init__(frame, timestamp, transform)
        self.latitude = latitude
        self.longitude = longitude
        self.altitude = altitude


class IMUMeasurementMock(SensorDataMock):

    def __init__(self, acceleration: carla.Vector3D, angular_velocity: carla.Vector3D, compass: float,
                 frame: int, timestamp: float, transform: carla.Transform):
        super().__init__(frame, timestamp, transform)
        self.accelerometer = acceleration
        self.gyroscope = angular_velocity
        self.compass = compass
