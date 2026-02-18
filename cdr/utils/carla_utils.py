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


from pathlib import Path
from typing import List, Optional

import carla


class CARLAClient:
    """
    This class acts as a wrapper around a `carla.Client` and forwards all usual `carla.Client` function calls to the
    wrapped client.

    Additionally, objects of this class can be pickled. It stores the configuration parameters of the
    client, so that a new `carla.Client` connection is established when being unpickled. This is useful when passing
    a `CARLAClient` to a `BaseWorker`:
    * In threaded mode, the client is simply passed to the thread and can directly be used.
    * In multiprocessing mode, the `carla.Client` is automatically reestablished during unpickling.
    """

    def __init__(self, host: str = '127.0.0.1', port: int = 2000, worker_threads: int = 0, timeout: float = 5.,
                 client: Optional[carla.Client] = None):
        """
        Initializes a new `CARLAClient`, storing the given client configuration and establishing an actual
        `carla.Client` connection to a CARLA server.

        Args:
            host (str, optional): IP address where a CARLA Simulator instance is running. Defaults to '127.0.0.1'.
            port (int, optional): TCP port where the CARLA Simulator instance is running. Defaults to 2000.
            worker_threads (int, optional): Number of working threads used for background updates.
                                            If 0, use all available concurrency. Defaults to 0.
            timeout (float, optional): The maximum time a network call is allowed before blocking it and raising a
                                       timeout exceeded error. Defaults to 5.
            client (Optional[carla.Client], optional): An already existing client, that shall be used explicitly by all
                threads in the current process. When the `CARLAClient` is passed to a subprocess, it will instantiate
                its own client. If `None`, a new client will directly be created.
        """
        self._init(host, port, worker_threads, timeout, client)

    def _init(self, host: str, port: int, worker_threads: int, timeout: float, client: Optional[carla.Client] = None):
        self._host = host
        self._port = port
        self._worker_threads = worker_threads
        self._timeout = timeout

        if client is not None:
            self._client = client
        else:
            self.connect()
        self._world = None

    def connect(self):
        self._client = carla.Client(self._host, self._port, self._worker_threads)
        self._client.set_timeout(self._timeout)
        self._world = None  # When establishing a new connection, we have to reset the world

    def get_world(self) -> carla.World:
        if self._world is None:
            self._world = self._client.get_world()
        return self._world

    def set_world(self, world: Optional[carla.Client]):
        """
        Explicitly set the given world instance to be returned by `get_world`. This enables different objects that share
        this CARLAClient to all interact with the identical world instance, instead of retrieving individual instances
        from the shared client.

        Args:
            world (Optional[carla.Client]): The world instance to be shared. If `None`, the currently set shared
                world instance will be removed.
        """
        self._world = world

    def __getstate__(self):
        # The _client is explicitly not part of the state, so that objects of this class can be pickled
        return self._host, self._port, self._worker_threads, self._timeout

    def __setstate__(self, state):
        # Initialize the unpickled object with the given configuration parameters, reestablishing the saved connection
        self._init(*state)

    def __getattr__(self, name: str):
        # Every attribute access that does not hit attributes of this class are directly forwarded to the client
        return getattr(self._client, name)


def find_ego_vehicle(world: carla.World, role_name: str = 'hero') -> carla.Actor:
    """
    Searches for the ego vehicle by looking for the given `role_name`. If multiple vehicles have the same `role_name`,
    this method returns the first one it finds.

    Args:
        world (carla.World): carla.World instance
        role_name (str, optional): the `role_name` that identifies the ego vehicle. Defaults to 'hero'.

    Raises:
        ValueError: if no actor with the given `role_name` could be found

    Returns:
        carla.Actor: the actor with the given `role_name`
    """
    all_actors = world.get_actors().filter('vehicle.*')
    for actor in all_actors:
        if 'role_name' in actor.attributes and actor.attributes['role_name'] == role_name:
            return actor
    raise ValueError(f'Could not find vehicle with role name "{role_name}".')


def get_map_name(world: carla.World) -> str:
    """
    Returns the name of the current map. If CARLA reports a map name as a path (e.g. Carla/Maps/Town10HD), this method
    returns only the last name (e.g. Town10HD).

    Args:
        world (carla.World): the current CARLA world instance

    Returns:
        str: the name of the current map
    """
    return Path(world.get_map().name).parts[-1]


def get_environment_vehicle_types() -> List[carla.CityObjectLabel]:
    """
    Returns a list of `CityObjectLabel` that can be categorized as vehicles.

    Returns:
        List[carla.CityObjectLabel]: a list of `CityObjectLabel` that can be categorized as vehicles
    """
    return [
        carla.CityObjectLabel.Car,
        carla.CityObjectLabel.Bus,
        carla.CityObjectLabel.Truck,
        carla.CityObjectLabel.Motorcycle,
        carla.CityObjectLabel.Bicycle,
        carla.CityObjectLabel.Train
    ]


def get_environment_vehicles(world: carla.World) -> List[carla.EnvironmentObject]:
    """
    Returns a list of all currently available environment vehicles in the world.
    Here, "vehicles" comprises the types [car, bus, truck, motorcycle, bicycle, train].

    Args:
        world (carla.World): the current CARLA world instance

    Returns:
        env_vehs (List[carla.EnvironmentObject]): a list of all currently available environment vehicles in the world
    """
    vehicle_labels = get_environment_vehicle_types()
    env_vehs = []
    for vehicle_label in vehicle_labels:
        env_vehs += world.get_environment_objects(vehicle_label)
    return env_vehs


def get_rear_axle_offset(vehicle: carla.Vehicle) -> carla.Vector3D:
    """
    Calculates and returns the offset between the vehicle's center point and the midpoint of its rear axle.
    The offset vector points from the center point to the rear axle.

    Note: The vehicle must simulate physics since the last tick, i.e., `vehicle.set_simulate_physics(False)` must not be
          set, when the last tick was performed.

    Args:
        vehicle (carla.Vehicle): the vehicle whose rear axle offset shall be determined

    Returns:
        carla.Vector3D: the offset vector from the vehicle's center point to the midpoint of its rear axle
    """
    vehicle_transform = vehicle.get_transform()
    physics_control = vehicle.get_physics_control()
    # Calculate the offset of the rear-axle midpoint from the vehicle's center position
    # Note:
    #   1) Also for two-wheeled vehicles, there are actually 4 wheels for the physics control
    #   2) The positions are returned in centimeters, not meters
    rear_left_wheel_pos = vehicle_transform.inverse_transform(physics_control.wheels[2].position / 100)
    rear_right_wheel_pos = vehicle_transform.inverse_transform(physics_control.wheels[3].position / 100)
    return (rear_left_wheel_pos + rear_right_wheel_pos) / 2
