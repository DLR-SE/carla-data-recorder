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


import carla
import numpy as np
from pyquaternion import Quaternion


def unpack_to_array(obj) -> np.ndarray:
    if isinstance(obj, (carla.Location, carla.Vector3D)):
        return np.array((obj.x, obj.y, obj.z))
    elif isinstance(obj, carla.Rotation):
        return np.array((obj.pitch, obj.yaw, obj.roll))
    elif isinstance(obj, Quaternion):
        return obj.elements
    else:
        return obj


def is_close(obj1, obj2):
    return np.all(np.isclose(unpack_to_array(obj1), unpack_to_array(obj2)))
