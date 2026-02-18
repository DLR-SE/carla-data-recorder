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


from typing import Dict, Optional

import carla
import numpy as np
from pyquaternion import Quaternion

from cdr.utils.carla_utils import CARLAClient, find_ego_vehicle

from .abstract import DataListener
from cdr.utils.carla_transformations import (array_to_carla_location, get_sensor_to_dict_function,
                                             quaternion_to_carla_rotation)


class SensorDataListener(DataListener):

    def __init__(self,
                 sensor_type: str, sensor_attributes: Dict[str, str],
                 sensor_location: np.ndarray, sensor_rotation: Quaternion, attach_to: Optional[str],
                 carla_client: CARLAClient, world_fps: int):
        """
        Spawns and listens to a new sensor. The data will be published to all subscribers of this listener.

        Args:
            sensor_type (str): the name of the CARLA blueprint for the sensor
            sensor_attributes (Dict[str, str]): a dictionary mapping blueprint attributes to values
            sensor_location (np.ndarray): the location of the sensor in a right-hand coordinate system.
                If `attach_to` is not None, this location is relative to the parent.
            sensor_rotation (Quaternion): the rotation of the sensor as quaternion in a right-hand coordinate system
            attach_to (Optional[str]): the name of the parent in the simulation. If `None`, no parent will be set.
            carla_client (CARLAClient): The currently used CARLA client of the CDR
            world_fps (int): The frame rate of the world
        """
        sensor_fps = 1. / float(sensor_attributes['sensor_tick'])
        ticks_per_frame = world_fps / sensor_fps

        super().__init__(carla_client=carla_client, requires_tick=True, ticks_per_frame=ticks_per_frame)

        self._sensor_type = sensor_type
        self._sensor_attributes = sensor_attributes
        self._sensor_location = sensor_location
        self._sensor_rotation = sensor_rotation
        self._attach_to = attach_to

    def _setup(self):
        if self._attach_to is not None:
            self._attach_to = find_ego_vehicle(self._world, self._attach_to)

        # Set blueprint attributes
        sensor_bp = self._world.get_blueprint_library().find(self._sensor_type)
        for bp_attribute_key, bp_attribute_value in self._sensor_attributes.items():
            if sensor_bp.has_attribute(bp_attribute_key):
                sensor_bp.set_attribute(bp_attribute_key, bp_attribute_value)

        # Create sensor transform
        is_camera = self._sensor_type.startswith('sensor.camera')
        location = array_to_carla_location(self._sensor_location)
        if self._attach_to is not None:
            # Some vehicle's BB has an offset, so we apply this offset to the sensor's position accordingly
            location += self._attach_to.bounding_box.location
        rotation = quaternion_to_carla_rotation(self._sensor_rotation, is_camera=is_camera)
        transform = carla.Transform(location, rotation)

        # Spawn and create callback
        self._sensor = self._world.spawn_actor(sensor_bp, transform, attach_to=self._attach_to)

        convert_func = get_sensor_to_dict_function(self._sensor_type)
        def callback(sensor_data):
            self._publish_data(sensor_data, convert_func)
        self._sensor.listen(callback)

    def post_work(self):
        self._sensor.destroy()

    def on_graceful_exit(self):
        self.post_work()
        super().on_graceful_exit()
