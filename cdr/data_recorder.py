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


import abc
import json
import logging
import math
import os
import random
import shutil
import sys
import time
import weakref
from copy import copy
from enum import Enum
from packaging.version import Version
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
from typing_extensions import Concatenate, ParamSpec, Self

import carla
import numpy as np
from pyquaternion import Quaternion

from cdr.utils.camera_utils import camera_parameters_to_intrinsic_matrix
from cdr.utils.carla_utils import CARLAClient, find_ego_vehicle
from cdr.utils.environment_objects import REGISTERED_HANDLER_TYPES, cleanup_environment_object_handlers
from cdr.utils.json_utils import customs_to_json, load_json, save_json
from cdr.utils.logging_utils import CDRColoredFormatter, CDRDefaultFormatter, CDRLogger
from cdr.workers.base import BaseWorker
from cdr.workers.listeners import DataListener, SensorDataListener, WorldSnapshotDataListener
from cdr.workers.processors import DataProcessor, InstanceSegmentationDataProcessor, SemanticLidarDataProcessor
from cdr.workers.watchdog import Watchdog
from cdr.workers.writers import (BB2DDataWriter, BB3DDataWriter, DataWriter, DepthDataWriter, GNSSDataWriter,
                                 IMUDataWriter, InstanceSegmentationDataWriter, LidarDataWriter, RGBDataWriter,
                                 SemanticLidarDataWriter, SemanticSegmentationDataWriter, TrajectoriesDataWriter)


DEFAULT_RECORDING_OUTPUTS: Dict[str, Dict[str, Any]] = {
    'CAMERA:RGB': {'format': 'jpg'},
    'CAMERA:BB2D': {'format': 'json'},
    'CAMERA:BB3D': {'format': 'json'},
    'CAMERA:DEPTH': {'format': 'exr'},
    'CAMERA:INSTSEG': {'format': 'png'},
    'CAMERA:SEMSEG': {'format': 'png'},
    'LIDAR:XYZI': {'format': 'ply'},
    # 'LIDAR:SEMANTIC': {'format': 'ply'},
    'GNSS': {'format': 'json'},
    'IMU': {'format': 'json'},
    'TRAJECTORIES': {'format': 'json'}
}


class Recorder(abc.ABC):
    """
    Abstract base class defining the interface of a recorder.
    """

    @abc.abstractmethod
    def __enter__(self): ...

    @abc.abstractmethod
    def __call__(self, world: Optional[carla.World]) -> Self: ...

    @abc.abstractmethod
    def __exit__(self, exc_type, exc_val, exc_tb): ...

    @abc.abstractmethod
    def start_recording(self, world: Optional[carla.World]): ...

    @abc.abstractmethod
    def stop_recording(self): ...

    @abc.abstractmethod
    def set_num_simulation_runs(self, num_sim_runs: int): ...

    @abc.abstractmethod
    def set_simulation_run_name(self, sim_run_name: str): ...


class SensorType(Enum):
    """
    Enum that specifies the currently supported sensor types.
    """

    CAMERA = 'camera'
    LIDAR = 'lidar'
    GNSS = 'gnss'
    IMU = 'imu'

    @classmethod
    def get(cls, sensor_type: str) -> 'SensorType':
        """
        Returns the corresponding enum-entry for the given sensor type.

        Args:
            sensor_type (str): string representation of the sensor type

        Raises:
            NotImplementedError: if the requested sensor type is currently not supported

        Returns:
            SensorType: the corresponding enum-entry for the given sensor type
        """
        try:
            return SensorType(sensor_type)
        except ValueError:
            raise NotImplementedError(f'Sensor of type "{sensor_type}" is currently not supported. Supported sensors: '
                                      f'{[sensor.value for sensor in SensorType]}')


P = ParamSpec('P')
def catch_log_handle(cdr_func: Callable[Concatenate['CARLADataRecorder', P], None]) -> Callable[Concatenate['CARLADataRecorder', P], None]:
    """
    Decorator for functions of `CARLADataRecorder`, which will wrap the execution of the decorated function in a
    try-except clause to catch any exception, log its information and then kill the recording.
    If the CDR was configured with `raise_exceptions`, the exception will be re-raised, otherwise the wrapped function
    simply returns.

    Args:
        cdr_func (Callable[Concatenate[CARLADataRecorder, P], None]): the CDR-function to decorate
    """
    def wrapper(self: 'CARLADataRecorder', *args: P.args, **kwargs: P.kwargs):
        try:
            cdr_func(self, *args, **kwargs)
        except (Exception, KeyboardInterrupt) as exc:
            self._logger.exception(f'An error occurred when calling CARLADataRecorder.{cdr_func.__name__}')
            self._logger.error('Kill current recording...')
            self._kill_recording()

            if self.raise_exceptions:
                raise exc
    return wrapper


def _find_sensor_config(name_or_filepath: str) -> Path:
    """
    Searches for a sensor configuration based on the given name or filepath. If a single name is specified,
    a sensor configuration part of the CDR-internal sensor configurations with the requested name is returned.
    A filepath can be used to specify sensor configurations external to the CDR.

    Args:
        name_or_filepath (str): name or filepath for the desired sensor configuration

    Raises:
        ValueError: if a name was specified and it could not be found in the CDR-internal sensor configurations
        FileNotFoundError: if a filepath was specified and the requested file could not be found

    Returns:
        Path: `Path`-object to the requested sensor configuration
    """
    if name_or_filepath.endswith('.json'):
        config_path = Path(name_or_filepath)
    else:
        if name_or_filepath in list_available_sensor_configs():
            cdr_dir = Path(os.path.dirname(os.path.abspath(__file__)))
            config_path = cdr_dir / 'sensor_configs' / f'{name_or_filepath}.json'
        else:
            raise ValueError(f'Given name of sensor configuration "{name_or_filepath}" could not be found.')

    if config_path.exists():
        return config_path
    else:
        raise FileNotFoundError(f'The sensor configuration file "{config_path}" could not be found.')


def list_available_sensor_configs() -> List[str]:
    """
    Returns a list with the names of all currently supported CDR-internal sensor configurations

    Returns:
        List[str]: the names of all currently supported CDR-internal sensor configurations
    """
    cdr_dir = Path(os.path.dirname(os.path.abspath(__file__)))
    return [os.path.splitext(config_filename)[0] for config_filename in os.listdir(cdr_dir / 'sensor_configs')]


def load_sensor_config(file_path: Union[str, Path]) -> Dict[str, Any]:
    """
    Loads the stored sensor configuration from the given file path, calculated the intrinsic camera matrix for
    every camera sensor and returns the configuration.

    Args:
        file_path (Union[str, Path]): path to the sensor configuration file

    Returns:
        Dict[str, Any]: the sensor configuration
    """
    sensor_config = load_json(file_path)

    # Derive and insert the intrinsic matrix for every camera sensor
    for sensor, attributes in sensor_config.items():
        if sensor == 'meta':
            continue

        if attributes['type'] == 'camera':
            intrinsic_matrix = camera_parameters_to_intrinsic_matrix(attributes['image_width'],
                                                                     attributes['image_height'],
                                                                     attributes['fov'])
            sensor_config[sensor]['intrinsic_matrix'] = intrinsic_matrix

    return sensor_config


def parse_sensor_config(sensor_config: Dict[str, Any])\
    -> Tuple[List[Tuple[SensorType, str, np.ndarray, Quaternion, Dict[str, Any]]], Dict[str, Any]]:
    """
    Returns sensor properties for the specified sensor configuration.

    Returns:
        sensor_info_w_meta (Tuple[List[Tuple[SensorType, str, np.ndarray, Quaternion, Dict[str, Any]]], Dict[str, Any]]):
            1) a list of tuples that contain the information for each sensor. The first tuple-entry denotes the
                sensor's type, the second entry an identifier for the sensor, the third entry the location of the
                sensor relative to a vehicle center point, the fourth entry the orientation of the sensor relative
                to a vehicle orientation and the last entry some parameters for the sensor.
            2) a dictionary containing some meta information
    """
    sensor_config = copy(sensor_config)
    meta_info = sensor_config.pop('meta', {})

    sensor_configurations: List[Tuple[SensorType, str, np.ndarray, Quaternion, Dict[str, Any]]] = []
    for sensor, attributes in sensor_config.items():
        # Translation and rotation are described in a front-left-up right-handed coordinate system,
        # located at the center of the vehicle.
        location = np.array(attributes['translation'])
        rotation = Quaternion(attributes['rotation'])

        sensor_type = SensorType.get(attributes['type'])
        if sensor_type == SensorType.CAMERA:
            param_names = ['image_width', 'image_height', 'fov']
        elif sensor_type == SensorType.LIDAR:
            param_names = ['num_layers', 'points_per_second', 'range', 'horizontal_fov', 'upper_fov', 'lower_fov',
                           'accuracy']
        elif sensor_type == SensorType.GNSS:
            param_names = ['horizontal_accuracy', 'vertical_accuracy']
        elif sensor_type == SensorType.IMU:
            param_names = ['accel_noise_density', 'gyro_noise_density']
        else:
            raise NotImplementedError(f'Sensor type "{sensor_type}" currently not supported.')
        param_names += ['capture_frequency']
        sensor_parameters = {param_name: attributes[param_name] for param_name in param_names}

        override_params_key = 'carla_parameters_override'
        if override_params_key in attributes:
            sensor_parameters[override_params_key] = attributes[override_params_key]

        sensor_configurations.append((sensor_type, sensor, location, rotation, sensor_parameters))

    return sensor_configurations, meta_info


def merge_sensor_configs(*sensor_configs: Union[Dict[str, Any], None]) -> Dict[str, Any]:
    """
    Merges the given sensor configurations and returns the combined sensor configuration.

    Args:
        *sensor_configs (Union[Dict[str, Any], None]): the sensor configurations to merge (`None`s are ignored)

    Raises:
        ValueError: if there are sensors with identical ID in different sensor configurations
    Returns:
        Dict[str, Any]: the combined sensor configuration
    """
    # Drop None values (supported for convenience of the caller)
    actual_sensor_configs = [cfg for cfg in sensor_configs if cfg is not None]
    assert len(actual_sensor_configs) > 0
    # Merge sensor configurations
    merged_sensor_config = copy(actual_sensor_configs[0])
    merged_sensor_config.pop('meta', None)  # drop the meta stuff (safe, since we made a copy)
    for sensor_config in actual_sensor_configs[1:]:
        duplicate_sensor_ids = set(merged_sensor_config.keys()).intersection(set(sensor_config.keys()))
        if len(duplicate_sensor_ids) > 0:
            raise ValueError('Sensor configurations cannot contain identical sensor IDs, but found collisions: '
                             f'{duplicate_sensor_ids}')
        merged_sensor_config.update(sensor_config)
        merged_sensor_config.pop('meta', None)  # drop the meta stuff again

    return merged_sensor_config


class CARLADataRecorder(Recorder):

    def __init__(self, results_dir: Union[str, Path], client: carla.Client, *,
                 host: str = 'localhost', port: int = 2000, timeout: float = 10.,
                 ego_rolename: str = 'hero', simulation_fps: int = 20,
                 sensor_config: str = 'nuscenes', static_sensor_config: Optional[str] = None,
                 recording_outputs: Dict[str, Dict[str, Any]] = DEFAULT_RECORDING_OUTPUTS,
                 delay_recording_seconds: float = 0., delay_recording_ticks: int = 0,
                 enable_ssaa: bool = False, ssaa_factor: float = 2.,
                 annotations_traverse_translucency: Union[bool, None] = None, prng_seed: int = 42,
                 raise_exceptions: bool = True, override_existing_data: bool = False, delete_on_error: bool = True,
                 allow_user_interrupt: bool = False, verbose: bool = False):
        """
        Instantiates a new CARLA Data Recorder object, that can be used to gather data from a CARLA simulation.
        Spawning and listening to sensors, internal processing of data and writing the results to the disk
        happens all in dedicated processes. Thus, the calling script is not impacted by any direct slowdowns that may
        occur in a threaded fashion due to the GIL.

        All resulting data will be written to the provided `results_dir`. For the file tree structure and data formats,
        have a look into the README of the CARLA Data Recorder:
        https://github.com/DLR-SE/carla-data-recorder/blob/main/README.md

        Args:
            results_dir (Union[str, Path]):
                The directory where simulation results shall be stored.

            client (carla.Client):
                The client instance that will be used by this CARLA Data Recorder object

            host (str):
                IP or hostname running the CARLA server.

            port (int):
                TCP port of the CARLA server.

            timeout (float):
                Timeout for CARLA client connecting to the server.

            ego_rolename (str):
                The rolename of the ego vehicle.

            simulation_fps (int):
                The frame rate of the simulation, i.e. number of ticks per second.

            sensor_config (str):
                Either the name of a builtin sensor configuration or the path to the desired sensor configuration file.
                The list of available builtin sensor configurations can be retrieved by via `list_available_sensor_configs`.

            static_sensor_config (Optional[str]):
                Path to configuration file for static sensors.

            recording_outputs (Dict[str, Dict[str, Any]]):
                Specifies the demanded outputs for every sensor type, as a mapping from the sensor output type identifier
                to the desired output format (key: "format") and optionally additional format parameters
                (key: "format_parameters").
                See the data format specification for more details.

            delay_recording_seconds (float): Delay the recording for the specified amount of simulated seconds.

            delay_recording_ticks (int): Delay the recording for the specified amount of world ticks.

            enable_ssaa (bool): Enables Super Sampling Anti Aliasing (SSAA).

            ssaa_factor (float): Sets the scaling factor to use for SSAA.

            annotations_traverse_translucency (Union[bool, None]):
                Enables/Disables annotations like depth or semantic and instance segmentation to traverse translucent
                materials. If `None`, this setting is kept as currently set. This feature requires CARLA >=0.9.16.

            prng_seed (int):
                Sets a master PRNG seed, that controls randomness across the entire CDR.

            raise_exceptions (bool):
                If `True`, exceptions will be raised, otherwise they will only be logged.

            override_existing_data (bool):
                If `True`, existing data in the directory of a new simulation run would be overridden,
                otherwise the recording will be rejected.

            delete_on_error (bool):
                If `True`, a data recording will be deleted when an error occurs, otherwise the data will be kept,
                likely resulting in inconsistencies between sensors (e.g. missing data).

            allow_user_interrupt (bool):
                If `True` and this object is used as context manager, a KeyboardInterrupt that leads to an exit of the
                context manager is allowed to stop the recording, otherwise the recording is killed. Depending on
                `delete_on_error` this might either delete the data of the recording or likely result in inconsistent data.

            verbose (bool):
                Enables verbose output to stdout. The logfile will always contain every logged message, independent of
                this setting.
        """
        # External variables
        self.results_dir = Path(results_dir)
        self.host = host
        self.port = port
        self.timeout = timeout
        self.ego_rolename = ego_rolename
        self.simulation_fps = simulation_fps
        self.sensor_config = load_sensor_config(_find_sensor_config(sensor_config))
        self.static_sensor_config = None if static_sensor_config is None else load_sensor_config(static_sensor_config)
        self.recording_outputs = recording_outputs
        self.delay_recording_seconds = delay_recording_seconds
        self.delay_recording_ticks = delay_recording_ticks
        self.enable_ssaa = enable_ssaa
        self.ssaa_factor = ssaa_factor
        self.annotations_traverse_translucency = annotations_traverse_translucency
        self.prng_seed = prng_seed
        self.raise_exceptions = raise_exceptions
        self.override_existing_data = override_existing_data
        self.delete_on_error = delete_on_error
        self.allow_user_interrupt = allow_user_interrupt
        self.verbose = verbose

        # Create results dir
        if not self.results_dir.is_dir():
            os.makedirs(self.results_dir)

        # Init logger for whole data recorder
        self._configure_logger()

        # Internal variables
        self._num_sim_runs = 1  # can be set externally via `set_num_simulation_runs`
        self._cur_sim_run = 1
        self._init_variables()

        # Connect CARLA client
        self._client = CARLAClient(self.host, self.port, timeout=self.timeout, client=client)
        self._world = None

        # Register a finalizer (called on object destruction) to undo possible changes made by env-obj handlers
        self._finalizer = weakref.finalize(self, cleanup_environment_object_handlers)

    def _init_variables(self):
        """
        Re-initializes required variables before the start of a new recording.
        """
        self._data_listeners: Dict[str, DataListener] = {}
        self._data_processors: Dict[str, DataProcessor] = {}
        self._data_writers: Dict[str, DataWriter] = {}

        self._is_recording = False
        self._watchdog = Watchdog(self._logger)
        self.prng_generator = random.Random(self.prng_seed)

        self._cur_sim_run_name = None
        self._metadata = None
        self._frame_recording_start = None

    def _configure_logger(self):
        """
        Configures the logger of the CDR.
        """
        # Set our custom Logger class, which adds the feature to specify the timestamp of a log message
        logging.Logger.manager.setLoggerClass(CDRLogger)
        self._logger: CDRLogger = logging.getLogger(f'carla-data-recorder-{id(self)}')  # type: ignore
        self._logger.setLevel(logging.DEBUG)

        log_stdout_handler = logging.StreamHandler(sys.stdout)
        log_stdout_handler.setLevel(logging.DEBUG if self.verbose else logging.INFO)
        log_stdout_handler.setFormatter(CDRColoredFormatter())

        log_timestamp = time.strftime('%Y-%m-%dT%H-%M-%S', time.gmtime(time.time()))
        log_file_handler = logging.FileHandler(self.results_dir / f'recorder_log-{log_timestamp}.txt')
        log_file_handler.setLevel(logging.DEBUG)
        log_file_handler.setFormatter(CDRDefaultFormatter())

        self._logger.addHandler(log_file_handler)
        self._logger.addHandler(log_stdout_handler)

        self._logger.propagate = False

    @classmethod
    def from_config(cls, results_dir: Union[str, Path], client: carla.Client, config_file: Union[str, Path]) -> 'CARLADataRecorder':
        """
        Instantiates a CARLA Data Recorder from the provided config file. This config is required to be a JSON file
        mapping the keyword arguments of the constructor to values. The constructors defaults are used for keywords not
        provided in the config file.

        Args:
            results_dir (Union[str, Path]): The directory where simulation results shall be stored.

            client (carla.Client): The client instance that will be used by this CARLA Data Recorder object

            config_file (Union[str, Path]): The JSON config file.

        Returns:
            CARLADataRecorder: the configured CARLA Data Recorder object
        """
        config_file = Path(config_file)
        with open(config_file) as file:
            config = json.load(file)

        # To allow for relative filepaths pointing to the sensor configs, we first try to find them as absolute paths
        # and then retry with relative paths
        sensor_config_params = ['sensor_config', 'static_sensor_config']
        for sensor_config_param in sensor_config_params:
            sensor_config: Optional[str] = config[sensor_config_param]
            if sensor_config is not None:
                try:
                    _find_sensor_config(sensor_config)
                except FileNotFoundError as exc:
                    try:
                        sensor_config_rel_resolved = str((config_file.parent / sensor_config).resolve())
                        _find_sensor_config(sensor_config_rel_resolved)
                        config[sensor_config_param] = sensor_config_rel_resolved
                    except FileNotFoundError:
                        raise exc

        return cls(results_dir, client, **config)

    def __enter__(self):
        self._logger.debug('Enter of CARLA Data recorder. Start recording.')
        self.start_recording()

    def __call__(self, world: Optional[carla.World]) -> Self:
        """
        Sets the given world instance to be used for recording the next simulation run.

        Args:
            world (Optional[carla.World]): the world instance to be set,
                                           or `None` to remove the currently associated instance

        Returns:
            Self: this CARLADataRecorder instance
        """
        self._client.set_world(world)
        self._world = world
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        # Check if a worker process crashed, leading to the exit.
        # Otherwise, we check, whether an uncaught exception occurred or not.
        # If this exception is a KeyboardInterrupt and `self.allow_user_interrupt` is True, we stop the recording,
        # otherwise the recording will be killed and already recorded data might be deleted.
        if self._watchdog.has_observed_crash():
            self._logger.error('Worker process crashed -> Kill recording.')
            self._kill_recording()
        elif exc_type is None:
            self._logger.debug('Exit of CARLA Data recorder -> Stop recording.')
            self.stop_recording()
        elif exc_type == KeyboardInterrupt:
            if self.allow_user_interrupt:
                self._logger.info('Simulation aborted by user -> Stop recording.')
                self.stop_recording()
            else:
                self._logger.error('Simulation aborted by user without being allowed. If this was intended, you may '
                                   'want to enable `allow_user_interrupt` -> Kill recording.')
                self._kill_recording()
        else:
            self._logger.error(f'Encountered unexpected exception of type "{exc_type.__name__}" -> Kill recording.',
                               exc_info=1)  # type: ignore
            self._kill_recording()

    def set_num_simulation_runs(self, num_sim_runs: int):
        """
        Set the amount of simulation runs that will be executed sequentially.

        Args:
            num_sim_runs (int): amount of simulation runs that will be executed sequentially
        """
        assert self._num_sim_runs > 0
        self._num_sim_runs = num_sim_runs

    def set_simulation_run_name(self, sim_run_name: str):
        """
        Set an explicit name for the current simulation run when executing multiple simulation runs sequentially
        (see `set_num_simulation_runs()`). This must be called before calling `start_record()` or entering the CDR
        as context manager.

        Args:
            sim_run_name (str): the name of the current simulation run
        """
        assert sim_run_name != ''
        self._cur_sim_run_name = sim_run_name

    @property
    def cur_simrun_dir(self) -> Path:
        """
        Returns the path to the directory of the current simulation run.

        Returns:
            Path: the path to the directory of the current simulation run
        """
        if self._num_sim_runs == 1:
            return self.results_dir
        else:
            if self._cur_sim_run_name is not None:
                return self.results_dir / self._cur_sim_run_name
            else:
                return self.results_dir / f'{self._cur_sim_run:06d}'

    def _set_sync_mode(self):
        """
        Sets CARLA to synchronous mode and remembers the previous configuration.
        """
        assert self._world is not None

        settings = self._world.get_settings()
        self._prev_sync_mode = settings.synchronous_mode
        self._prev_fixed_delta_seconds = settings.fixed_delta_seconds

        settings.synchronous_mode = True
        settings.fixed_delta_seconds = 1.0 / self.simulation_fps
        self._world.apply_settings(settings)

    def _reset_sync_mode(self):
        """
        Resets CARLA to its mode before calling `_set_sync_mode()`.
        """
        assert self._world is not None

        if hasattr(self, '_prev_sync_mode'):
            settings = self._world.get_settings()
            settings.synchronous_mode = self._prev_sync_mode
            settings.fixed_delta_seconds = self._prev_fixed_delta_seconds
            self._world.apply_settings(settings)

    def _spawn_sensor_configuration(self, sensor_config: Dict[str, Any],
                                    ego_vehicle: Optional[carla.Actor] = None) -> Dict[SensorType, List[str]]:
        """
        Spawns `DataListener` for all the sensors specified in the given sensor configuration. If `ego_vehicle` is set,
        the sensors are attached to it.

        Args:
            sensor_config (Dict[str, Any]): the sensor configuration
            ego_vehicle (Optional[carla.Actor], optional): an optional actor to attach the sensors to. Defaults to None.

        Raises:
            NotImplementedError: If a requested sensor type in the sensor configuration is currently not supported.

        Returns:
            spawned_sensors (Dict[SensorType, List[str]]):
            a dictionary containing the names of the spawned sensors for every sensor type
        """
        # Keep track of spawned sensors
        spawned_sensor_ids: Dict[SensorType, List[str]] = {}

        # Determine which sensor types shall be used
        spawn_rgb_camera = 'CAMERA:RGB' in self.recording_outputs
        spawn_depth_camera = 'CAMERA:DEPTH' in self.recording_outputs
        spawn_instseg_camera = any([mode in self.recording_outputs for mode in
                                    ['CAMERA:SEMSEG', 'CAMERA:INSTSEG', 'CAMERA:BB2D', 'CAMERA:BB3D']])
        spawn_default_lidar = 'LIDAR:XYZI' in self.recording_outputs
        spawn_semantic_lidar = 'LIDAR:SEMANTIC' in self.recording_outputs
        spawn_gnss = 'GNSS' in self.recording_outputs
        spawn_imu = 'IMU' in self.recording_outputs

        def map_values_to_string(mapping: Dict[str, Any]) -> Dict[str, str]:
            return {key: str(val) for key, val in mapping.items()}

        def check_tickrate(sensor_id: str, sensor_tick_rate: float):
            frac = self.simulation_fps / sensor_tick_rate
            if abs(frac - round(frac)) > 1e-6:
                self._logger.warning(f'Simulation tick rate (={self.simulation_fps}) is not a multiple of '
                                     f'"{sensor_id}"\'s capture_frequency (={sensor_tick_rate}). '
                                     'This leads to frames not being captured at the desired interval.')

        def spawn_camera_listener(sensor_id: str,
                                  location: np.ndarray, rotation: Quaternion,
                                  sensor_parameters: Dict[str, Any], attach_to: Optional[str] = None) -> bool:
            width, height = sensor_parameters['image_width'], sensor_parameters['image_height']
            fov, sensor_tick = sensor_parameters['fov'], 1. / sensor_parameters['capture_frequency']

            check_tickrate(sensor_id, sensor_parameters['capture_frequency'])

            def get_cam_attributes(use_ssaa: bool = False) -> Dict[str, str]:
                image_size_x, image_size_y = width, height
                if use_ssaa:
                    image_scaling_factor = self.ssaa_factor
                    image_size_x = int(width * image_scaling_factor)
                    image_size_y = int(height * image_scaling_factor)

                camera_attributes = {'image_size_x': image_size_x, 'image_size_y': image_size_y,
                                     'fov': fov, 'sensor_tick': sensor_tick}
                override_params_key = 'carla_parameters_override'
                if override_params_key in sensor_parameters:
                    camera_attributes.update(sensor_parameters[override_params_key])
                return map_values_to_string(camera_attributes)

            if spawn_rgb_camera:
                listener = SensorDataListener('sensor.camera.rgb', get_cam_attributes(self.enable_ssaa),
                                              location, rotation, attach_to, self._client, self.simulation_fps)
                self._data_listeners[f'{sensor_id}:RGB'] = listener

            if spawn_depth_camera:
                listener = SensorDataListener('sensor.camera.depth', get_cam_attributes(),
                                              location, rotation, attach_to, self._client, self.simulation_fps)
                self._data_listeners[f'{sensor_id}:DEPTH'] = listener

            if spawn_instseg_camera:
                listener = SensorDataListener('sensor.camera.instance_segmentation', get_cam_attributes(),
                                              location, rotation, attach_to, self._client, self.simulation_fps)
                self._data_listeners[f'{sensor_id}:INSTSEG'] = listener

            return any([spawn_rgb_camera, spawn_depth_camera, spawn_instseg_camera])

        def spawn_lidar_listener(sensor_id: str,
                                 location: np.ndarray, rotation: Quaternion,
                                 sensor_parameters: Dict[str, Any],
                                 attach_to: Optional[str] = None) -> bool:
            lidar_attributes = {
                'channels': sensor_parameters['num_layers'],
                'points_per_second': sensor_parameters['points_per_second'],
                'range': sensor_parameters['range'],
                'horizontal_fov': sensor_parameters['horizontal_fov'],
                'upper_fov': sensor_parameters['upper_fov'],
                'lower_fov': sensor_parameters['lower_fov'],
                'noise_stddev': sensor_parameters['accuracy'],
                'rotation_frequency': self.simulation_fps,  # always cover full horizontal FOV
                'sensor_tick': 1. / sensor_parameters['capture_frequency']
            }
            override_params_key = 'carla_parameters_override'
            if override_params_key in sensor_parameters:
                lidar_attributes.update(sensor_parameters[override_params_key])
            lidar_attributes = map_values_to_string(lidar_attributes)

            check_tickrate(sensor_id, sensor_parameters['capture_frequency'])

            if spawn_default_lidar:
                listener = SensorDataListener('sensor.lidar.ray_cast', lidar_attributes, location, rotation,
                                              attach_to, self._client, self.simulation_fps)
                self._data_listeners[f'{sensor_id}:XYZI'] = listener

            if spawn_semantic_lidar:
                listener = SensorDataListener('sensor.lidar.ray_cast_semantic', lidar_attributes, location, rotation,
                                              attach_to, self._client, self.simulation_fps)
                self._data_listeners[f'{sensor_id}:SEMANTIC'] = listener

            return any([spawn_default_lidar, spawn_semantic_lidar])

        def spawn_gnss_listener(sensor_id: str,
                                location: np.ndarray, rotation: Quaternion,
                                sensor_parameters: Dict[str, Any],
                                attach_to: Optional[str] = None) -> bool:
            if not spawn_gnss:
                return False

            if attach_to is None:
                self._logger.warning(f'Was spawning a static GNSS sensor ({sensor_id}) really intended?')

            # TODO Extract actual georeference from the current CARLA map and use that?
            horizontal_accuracy_degrees = sensor_parameters['horizontal_accuracy'] / 111_111  # approximation at equator
            std_sigma = 3.891  # 3.891σ = 99.99% of all values lie in the requested accuracy
            gnss_attributes = {
                'noise_lat_stddev': horizontal_accuracy_degrees / std_sigma,
                'noise_lon_stddev': horizontal_accuracy_degrees / std_sigma,
                'noise_alt_stddev': sensor_parameters['vertical_accuracy'] / std_sigma,
                'sensor_tick': 1. / sensor_parameters['capture_frequency']
            }
            override_params_key = 'carla_parameters_override'
            if override_params_key in sensor_parameters:
                gnss_attributes.update(sensor_parameters[override_params_key])
            gnss_attributes = map_values_to_string(gnss_attributes)

            check_tickrate(sensor_id, sensor_parameters['capture_frequency'])

            listener = SensorDataListener('sensor.other.gnss', gnss_attributes, location, rotation,
                                          attach_to, self._client, self.simulation_fps)
            self._data_listeners[f'{sensor_id}:GNSS'] = listener
            return True

        def spawn_imu_listener(sensor_id: str,
                               location: np.ndarray, rotation: Quaternion,
                               sensor_parameters: Dict[str, Any],
                               attach_to: Optional[str] = None) -> bool:
            if not spawn_imu:
                return False

            if attach_to is None:
                self._logger.warning(f'Was spawning a static IMU sensor ({sensor_id}) really intended?')

            # accelerometer noise density is specified in µG/√Hz
            G = 9.80665  # earth gravity
            accel_std = sensor_parameters['accel_noise_density'] / 1e6 * G * np.sqrt(self.simulation_fps)
            # gyroscope noise density is specified in (°/s)/√Hz
            gyro_std = np.deg2rad(sensor_parameters['gyro_noise_density']) * np.sqrt(self.simulation_fps)
            imu_attributes = {
                'noise_accel_stddev_x': accel_std,
                'noise_accel_stddev_y': accel_std,
                'noise_accel_stddev_z': accel_std,
                'noise_gyro_stddev_x': gyro_std,
                'noise_gyro_stddev_y': gyro_std,
                'noise_gyro_stddev_z': gyro_std,
                'sensor_tick': 1. / sensor_parameters['capture_frequency']
            }
            override_params_key = 'carla_parameters_override'
            if override_params_key in sensor_parameters:
                imu_attributes.update(sensor_parameters[override_params_key])
            imu_attributes = map_values_to_string(imu_attributes)

            check_tickrate(sensor_id, sensor_parameters['capture_frequency'])

            listener = SensorDataListener('sensor.other.imu', imu_attributes, location, rotation,
                                          attach_to, self._client, self.simulation_fps)
            self._data_listeners[f'{sensor_id}:IMU'] = listener
            return True

        # Parse sensor configuration file
        sensor_configuration, meta_info = parse_sensor_config(sensor_config)
        if ego_vehicle is not None:
            optimal_vehicle_key = 'carla_optimal_vehicle'
            if optimal_vehicle_key in meta_info and ego_vehicle.type_id != meta_info[optimal_vehicle_key]:
                self._logger.warning('Optimal CARLA vehicle for specified sensor configuration is '
                                     f'"{meta_info[optimal_vehicle_key]}" but current ego vehicle is '
                                     f'"{ego_vehicle.type_id}".')

        # Spawn sensors from the configuration file
        for sensor_type, sensor_id, location, rotation, sensor_parameters in sensor_configuration:
            attach_to = self.ego_rolename if ego_vehicle is not None else None
            if sensor_type == SensorType.CAMERA:
                did_spawn = spawn_camera_listener(sensor_id, location, rotation, sensor_parameters, attach_to)
            elif sensor_type == SensorType.LIDAR:
                did_spawn = spawn_lidar_listener(sensor_id, location, rotation, sensor_parameters, attach_to)
            elif sensor_type == SensorType.GNSS:
                did_spawn = spawn_gnss_listener(sensor_id, location, rotation, sensor_parameters, attach_to)
            elif sensor_type == SensorType.IMU:
                did_spawn = spawn_imu_listener(sensor_id, location, rotation, sensor_parameters, attach_to)
            else:
                raise NotImplementedError(f'Sensor type "{sensor_type}" currently not supported.')
            if did_spawn:
                if sensor_type not in spawned_sensor_ids.keys():
                    spawned_sensor_ids[sensor_type] = []
                spawned_sensor_ids[sensor_type].append(sensor_id)

        return spawned_sensor_ids

    @catch_log_handle
    def start_recording(self, world: Optional[carla.World] = None):
        """
        This method
        1) spawns the sensors that were defined when initializing this instance. This currently includes:
            - sensors attached to the ego vehicle (`sensor_config_file`)
            - static cameras placed in the world (`static_camera_poses_file`, `self.static_camera_ids`,
                                                  `self.static_camera_intrinsics`)
        2) spawns all required subprocesses to receive data from the simulator, process and then write it to disk.

        After this setup is done, the method returns the control flow back to the caller. From now on, every tick
        issued by the caller will be recorded. When finished with the simulation, `stop_recording` should be called.

        Note: Instead of using `start_recording` and `stop_recording` manually, it is advised to use the
        CARLA Data Recorder as a context manager:
        ```python
        recorder = CARLADataRecorder(results_dir, client)
        with recorder(world):
            # do something
            world.tick()
        ```

        Args:
            world (Optional[carla.World]): The current world instance, that shall be used by the CDR for this recording.
                                           If `None` (default), the world must be provided via `__call__`,
                                           previously to calling this method.
        """
        _t_recording_init = time.time()

        if self._is_recording:
            raise RuntimeError('Invoked start of a new recording while a recording is still running!')

        if world is not None:  # If the world is given as argument (CDR not used as context manager), we set it here
            self(world)
        if self._world is None:  # Otherwise the world must be given previously via __call__
            raise RuntimeError('Invoked start of a new recording without setting the world. Must either provide the '
                               'world via `__call__(world)` previously to starting a recording, or provide the world '
                               'explicitly as argument of `start_recording(world)`!')

        # Check/Delete/Create directory for simulation run
        if self._num_sim_runs > 1:
            # If there is already a directory with the same name, depending on config we either override or throw error
            if self.cur_simrun_dir.is_dir():
                if self.override_existing_data:
                    shutil.rmtree(self.cur_simrun_dir)
                else:
                    raise RuntimeError(f'Directory for current simulation run "{self.cur_simrun_dir}" already exists, '
                                       'but `override_existing_data` was not set. -> Reject recording.')
            os.mkdir(self.cur_simrun_dir)
        else:
            filenames_in_simrun_dir = os.listdir(self.cur_simrun_dir)
            # If there are files/dirs we need, depending on config we either override or throw error
            for filename in filenames_in_simrun_dir:
                if filename in ['sensor_configuration.json', 'metadata.json']:
                    if self.override_existing_data:
                        os.remove(self.cur_simrun_dir / filename)
                    else:
                        raise RuntimeError(f'File "{filename}" in current simulation run "{self.cur_simrun_dir}" '
                                           'already exists, but `override_existing_data` was not set.'
                                           '-> Reject recording.')
                elif filename in ['sensor', 'ground_truth']:
                    if self.override_existing_data:
                        shutil.rmtree(self.cur_simrun_dir / filename)
                    else:
                        raise RuntimeError(f'Directory "{filename}" in current simulation run "{self.cur_simrun_dir}" '
                                           'already exists, but `override_existing_data` was not set. '
                                           '-> Reject recording.')

        self._logger.info(f'Recording run {self._cur_sim_run: >{len(str(self._num_sim_runs))}}/{self._num_sim_runs}')
        self._is_recording = True

        self._set_sync_mode()
        self._world.tick()  # tick it once, so that everything by scenario_runner is actually spawned

        # Enables/Disables annotations to traverse translucent materials (requires CARLA >=0.9.16)
        client_version = Version(self._client.get_client_version())
        if self.annotations_traverse_translucency is not None:
            if client_version >= Version('0.9.16'):
                self._world.set_annotations_traverse_translucency(self.annotations_traverse_translucency)
            else:
                self._logger.warning('Requested to explicitly set annotations_traverse_translucency='
                                     f'{self.annotations_traverse_translucency}, requiring CARLA >=0.9.16, but CARLA '
                                     f'client version is {client_version}. -> Setting ignored.')
                self.annotations_traverse_translucency = None

        # Write some general metadata about the scenario, assuming that these properties are static
        weather = self._world.get_weather()
        self._metadata = {
            "map": self._world.get_map().name,
            "weather": {
                "cloudiness": weather.cloudiness,
                "precipitation": weather.precipitation,
                "precipitation_deposits": weather.precipitation_deposits,
                "wind_intensity": weather.wind_intensity,
                "sun_azimuth_angle": weather.sun_azimuth_angle,
                "sun_altitude_angle": weather.sun_altitude_angle,
                "fog_density": weather.fog_density,
                "fog_falloff": weather.fog_falloff,
                "wetness": weather.wetness,
                "scattering_intensity": weather.scattering_intensity,
                "mie_scattering_scale": weather.mie_scattering_scale,
                "rayleigh_scattering_scale": weather.rayleigh_scattering_scale
            }
        }
        if self.annotations_traverse_translucency is not None:
            self._metadata['annotations_traverse_translucency'] = self.annotations_traverse_translucency

        # Merge the sensor configurations and store it to the data dir
        merged_sensor_config = merge_sensor_configs(self.sensor_config, self.static_sensor_config)
        save_json(merged_sensor_config, self.cur_simrun_dir / 'sensor_configuration.json', default=customs_to_json)

        # Install handlers for specific environment objects, if user did not interact with them yet
        for env_obj_handler in REGISTERED_HANDLER_TYPES:
            handler, is_new = env_obj_handler.get(self._world)
            handler_seed = self.prng_generator.randint(0, 2**32 - 1)
            handler.set_additional_parameters(self._client, handler_seed)
            if is_new:
                handler.handle(None, True)

        # Get the ego vehicle
        ego_vehicle = find_ego_vehicle(self._world, self.ego_rolename)
        self._logger.debug(f'Ego vehicle is: {ego_vehicle}')

        ####################### CREATE DATA LISTENERS #######################
        self._logger.debug('Create data listeners...')

        # Spawn dynamic sensors that are attached to the ego vehicle
        ego_sensor_ids = self._spawn_sensor_configuration(self.sensor_config, ego_vehicle)
        all_sensor_ids = {sensor_type: ego_sensor_ids.get(sensor_type, []).copy() for sensor_type in SensorType}

        # Spawn static cameras (if specified)
        if self.static_sensor_config is not None:
            static_sensor_ids = self._spawn_sensor_configuration(self.static_sensor_config)
            for sensor_type, sensor_ids in static_sensor_ids.items():
                all_sensor_ids[sensor_type] += sensor_ids

        # Create listener for world snapshots, used by instance segmentation processor, 2D and 3D BBs writers,
        # trajectories writer and the main loop
        self._data_listeners['WORLD_SNAPSHOT'] = WorldSnapshotDataListener(self._client)

        ####################### CREATE DATA PROCESSORS #######################
        self._logger.debug('Create data processors...')

        # Create processors for refining the instance segmentations...
        for sensor_id in all_sensor_ids[SensorType.CAMERA]:
            if f'{sensor_id}:INSTSEG' in self._data_listeners:  # ... if there is any instance segmentation at all
                # Cameras attached to ego must explicitly tag the ego vehicle in the segmentation
                tag_ego = sensor_id in ego_sensor_ids[SensorType.CAMERA]
                processor = InstanceSegmentationDataProcessor(self._data_listeners[f'{sensor_id}:INSTSEG'].subscribe(),
                                                              tag_ego, ego_vehicle.id)
                self._data_processors[f'{sensor_id}:PANOPTSEG'] = processor

        # Create processors for refining the semantic lidar...
        for sensor_id in all_sensor_ids[SensorType.LIDAR]:
            if f'{sensor_id}:SEMANTIC' in self._data_listeners:  # ... if there is any semantic lidar at all
                # Lidars attached to ego must explicitly tag the ego vehicle in the segmentation
                tag_ego = sensor_id in ego_sensor_ids[SensorType.LIDAR]
                processor = SemanticLidarDataProcessor(self._data_listeners[f'{sensor_id}:SEMANTIC'].subscribe(),
                                                       tag_ego, ego_vehicle.id)
                self._data_processors[f'{sensor_id}:SEMANTIC'] = processor

        ####################### CREATE DATA WRITERS #######################
        def get_output_format(sensor_output_type: str) -> Tuple[str, Dict[str, Any]]:
            output_format = self.recording_outputs[sensor_output_type]['format']
            output_format_parameters = self.recording_outputs[sensor_output_type].get('format_parameters', {})
            return output_format, output_format_parameters

        self._logger.debug('Create data writers...')
        # DataWriter for camera sensors
        for sensor_id in all_sensor_ids[SensorType.CAMERA]:
            # Create RGB and depth writers (if not disabled), which directly use the sensor outputs
            if 'CAMERA:RGB' in self.recording_outputs:
                output_format, output_format_parameters = get_output_format('CAMERA:RGB')
                output_format_parameters['fps'] = merged_sensor_config[sensor_id]['capture_frequency']
                rgb_writer = RGBDataWriter(self._data_listeners[f'{sensor_id}:RGB'].subscribe(),
                                           self.cur_simrun_dir, sensor_id, output_format,
                                           self.enable_ssaa, self.ssaa_factor,
                                           **output_format_parameters)
                self._data_writers[f'{sensor_id}:RGB'] = rgb_writer

            if 'CAMERA:DEPTH' in self.recording_outputs:
                output_format, output_format_parameters = get_output_format('CAMERA:DEPTH')
                depth_writer = DepthDataWriter(self._data_listeners[f'{sensor_id}:DEPTH'].subscribe(),
                                               merged_sensor_config[sensor_id]['intrinsic_matrix'],
                                               self.cur_simrun_dir, sensor_id, output_format,
                                               **output_format_parameters)
                self._data_writers[f'{sensor_id}:DEPTH'] = depth_writer

            # Create 3D and 2D BBs writers (if not disabled), which also use the panoptic segmentation to filter the BBs
            if 'CAMERA:BB3D' in self.recording_outputs:
                # Cameras attached to ego must filter the ego vehicle in BBs
                filter_ego = sensor_id in ego_sensor_ids[SensorType.CAMERA]
                output_format, output_format_parameters = get_output_format('CAMERA:BB3D')
                bb_3d_writer = BB3DDataWriter(self._data_listeners['WORLD_SNAPSHOT'].subscribe(),
                                              self._data_processors[f'{sensor_id}:PANOPTSEG'].subscribe(),
                                              self.cur_simrun_dir, sensor_id, output_format,
                                              filter_ego, self.ego_rolename,
                                              **output_format_parameters)
                self._data_writers[f'{sensor_id}:BB3D'] = bb_3d_writer
            if 'CAMERA:BB2D' in self.recording_outputs:
                # Cameras attached to ego must filter the ego vehicle in BBs
                filter_ego = sensor_id in ego_sensor_ids[SensorType.CAMERA]
                output_format, output_format_parameters = get_output_format('CAMERA:BB2D')
                bb_2d_writer = BB2DDataWriter(self._data_listeners['WORLD_SNAPSHOT'].subscribe(),
                                              self._data_processors[f'{sensor_id}:PANOPTSEG'].subscribe(),
                                              merged_sensor_config[sensor_id]['intrinsic_matrix'],
                                              self.cur_simrun_dir, sensor_id, output_format,
                                              filter_ego, self.ego_rolename,
                                              **output_format_parameters)
                self._data_writers[f'{sensor_id}:BB2D'] = bb_2d_writer

            # Create semantic and instance segmentation writers (if not disabled), which use the panoptic segmentation
            if 'CAMERA:SEMSEG' in self.recording_outputs:
                output_format, output_format_parameters = get_output_format('CAMERA:SEMSEG')
                semseg_writer = SemanticSegmentationDataWriter(self._data_processors[f'{sensor_id}:PANOPTSEG'].subscribe(),
                                                               self.cur_simrun_dir, sensor_id, output_format,
                                                               **output_format_parameters)
                self._data_writers[f'{sensor_id}:SEMSEG'] = semseg_writer
            if 'CAMERA:INSTSEG' in self.recording_outputs:
                output_format, output_format_parameters = get_output_format('CAMERA:INSTSEG')
                instseg_writer = InstanceSegmentationDataWriter(self._data_processors[f'{sensor_id}:PANOPTSEG'].subscribe(),
                                                                self.cur_simrun_dir, sensor_id, output_format,
                                                                **output_format_parameters)
                self._data_writers[f'{sensor_id}:INSTSEG'] = instseg_writer

        # DataWriter for lidar sensors
        for sensor_id in all_sensor_ids[SensorType.LIDAR]:
            if 'LIDAR:XYZI' in self.recording_outputs:
                output_format, output_format_parameters = get_output_format('LIDAR:XYZI')
                lidar_writer = LidarDataWriter(self._data_listeners[f'{sensor_id}:XYZI'].subscribe(),
                                               self.cur_simrun_dir, sensor_id, output_format,
                                               **output_format_parameters)
                self._data_writers[f'{sensor_id}:XYZI'] = lidar_writer
            if 'LIDAR:SEMANTIC' in self.recording_outputs:
                output_format, output_format_parameters = get_output_format('LIDAR:SEMANTIC')
                lidar_writer = SemanticLidarDataWriter(self._data_processors[f'{sensor_id}:SEMANTIC'].subscribe(),
                                                       self.cur_simrun_dir, sensor_id, output_format,
                                                       **output_format_parameters)
                self._data_writers[f'{sensor_id}:SEMANTIC'] = lidar_writer

        # DataWriter for gnss sensors
        for sensor_id in all_sensor_ids[SensorType.GNSS]:
            output_format, output_format_parameters = get_output_format('GNSS')
            gnss_writer = GNSSDataWriter(self._data_listeners[f'{sensor_id}:GNSS'].subscribe(),
                                         self.cur_simrun_dir, sensor_id, output_format,
                                         **output_format_parameters)
            self._data_writers[f'{sensor_id}:GNSS'] = gnss_writer

        # DataWriter for imu sensors
        for sensor_id in all_sensor_ids[SensorType.IMU]:
            output_format, output_format_parameters = get_output_format('IMU')
            imu_writer = IMUDataWriter(self._data_listeners[f'{sensor_id}:IMU'].subscribe(),
                                       self.cur_simrun_dir, sensor_id, output_format,
                                       **output_format_parameters)
            self._data_writers[f'{sensor_id}:IMU'] = imu_writer

        # DataWriter for trajectories
        if 'TRAJECTORIES' in self.recording_outputs:
            output_format, output_format_parameters = get_output_format('TRAJECTORIES')
            traj_writer = TrajectoriesDataWriter(self._data_listeners['WORLD_SNAPSHOT'].subscribe(),
                                                self.cur_simrun_dir, output_format,
                                                **output_format_parameters)
            self._data_writers['TRAJECTORIES'] = traj_writer

        ####################### RUN #######################
        # Register all workers at the watchdog, to enable logging and crash monitoring
        self._watchdog.register(*self._data_listeners.values())
        self._watchdog.register(*self._data_processors.values())
        self._watchdog.register(*self._data_writers.values())
        self._watchdog.start()

        # Start all listeners
        self._logger.debug('Start data listeners...')
        for listener in self._data_listeners.values():
            listener.start()

        # Since the listeners may create new CARLA clients, they require one tick to be fully initialized
        self._logger.debug('Wait for data listeners to be waiting for a tick...')
        for listener in self._data_listeners.values():
            listener.wait_until_awaits_tick()

        self._logger.debug('Tick...')
        self._world.tick()

        # Since the listeners do some setup work (especially spawning sensors), we need to wait until they are ready
        self._logger.debug('Wait for data listeners to be ready for the simulation start...')
        for listener in self._data_listeners.values():
            listener.wait_until_ready()

        # In our experiments we found, that when initializing a new map, textures may not be fully loaded.
        # -> We explicitly tick the world 5 times, so that everything is initialized.
        required_init_ticks = 5
        # Since the recording may be delayed by request of the user, these explicit ticks may be reduced.
        delay_recording_frames = math.ceil(self.delay_recording_seconds * self.simulation_fps) + self.delay_recording_ticks
        if delay_recording_frames == 0:
            # To avoid inconsistencies at the start of a scenario (e.g. change of weather), we insert a delay of one
            # tick at the start of the recording.
            delay_recording_frames = 1
        if delay_recording_frames < required_init_ticks:
            self._logger.debug('Execute pre-recording ticks...')
            for _ in range(required_init_ticks - delay_recording_frames):
                self._world.tick()

        # Notify sensor listeners about the start time of the recording
        world_snapshot = self._world.get_snapshot()
        frame_sim_start, time_sim_start = world_snapshot.timestamp.frame, world_snapshot.timestamp.elapsed_seconds
        self._frame_recording_start = frame_sim_start + delay_recording_frames
        time_recording_start = time_sim_start + delay_recording_frames * (1. / self.simulation_fps)
        for listener in self._data_listeners.values():
            listener.set_recording_start(self._frame_recording_start, time_recording_start)

        self._logger.debug('Start worker processes...')
        # Start all processors
        for processor in self._data_processors.values():
            processor.start()
        # Start all writers
        for writer in self._data_writers.values():
            writer.start()
        self._t_recording_start = time.time()
        elapsed_time_rec_init = self._t_recording_start - _t_recording_init
        self._logger.info(f'Recording initialization finished (took {elapsed_time_rec_init:.3f}s). Start scenario.')

    @catch_log_handle
    def stop_recording(self):
        """
        This method stops the current recording. For this, the method blocks until all worker processes of the
        CARLA Data Recorder finished their work.

        Note: Instead of using `start_recording` and `stop_recording` manually, it is advised to use the
        CARLA Data Recorder as a context manager (see `start_recording(...)`).
        """
        if not self._is_recording:
            raise RuntimeError('Invoked stop of the current recording without any recording actually running!')
        assert self._world is not None

        frame_sim_end = self._world.get_snapshot().timestamp.frame
        # Notify sensor listeners about the end time of the recording
        for listener in self._data_listeners.values():
            listener.set_recording_end(frame_sim_end)
        self._logger.info(f'Simulation of the scenario finished at frame {frame_sim_end}!')

        def wait_and_log(worker: BaseWorker, worker_name: str):
            exit_status = worker.wait_until_finished()
            if exit_status is not None:
                msg = f'Error occurred in "{type(worker)} | {worker_name}".'
                exc, tb = exit_status
                self._logger.error(f'{msg}\n{repr(exc)}\n{tb}')
                raise RuntimeError(msg) from exc

        self._logger.info('Wait for all data listeners to terminate...')
        for name, data_listener in self._data_listeners.items():
            self._logger.debug(f'Waiting for listener {name}...')
            wait_and_log(data_listener, name)

        self._logger.info('Wait for all data processors to terminate...')
        for name, data_processor in self._data_processors.items():
            self._logger.debug(f'Waiting for processor {name}...')
            wait_and_log(data_processor, name)

        self._logger.info('Wait for all data writers to terminate...')
        num_written_frames = {}
        for name, data_writer in self._data_writers.items():
            self._logger.debug(f'Waiting for writer {name}...')
            wait_and_log(data_writer, name)
            sensor_name = name.split(':')[0]
            num_written_frames[sensor_name] = data_writer.get_num_iterations()

        # Collect further metadata and write to disk
        assert self._metadata is not None
        assert self._frame_recording_start is not None
        self._metadata['num_simulated_ticks'] = frame_sim_end - self._frame_recording_start + 1
        self._metadata['tick_duration'] = 1.0 / self.simulation_fps
        self._metadata['num_frames'] = num_written_frames
        save_json(self._metadata, self.cur_simrun_dir / 'metadata.json')

        elapsed_time_rec = time.time() - self._t_recording_start
        self._logger.info(f'Finished writing of all data (took {elapsed_time_rec:.3f}s).')

        # Stop the watchdog
        self._watchdog.stop()

        self._reset()

    def _kill_recording(self):
        """
        Kills the current recording by stopping all running workers, the watchdog and deleting the already written data,
        if `delete_on_error` is set.
        """
        # Kill all worker processes and stop the watchdog
        for name, data_listener in self._data_listeners.items():
            self._logger.debug(f'Killing listener {name}...')
            data_listener.stop()
        for name, data_processor in self._data_processors.items():
            self._logger.debug(f'Killing processor {name}...')
            data_processor.stop()
        for name, data_writer in self._data_writers.items():
            self._logger.debug(f'Killing writer {name}...')
            data_writer.stop()

        for data_listener in self._data_listeners.values():
            if data_listener.is_started():
                data_listener.join()
        for data_processor in self._data_processors.values():
            if data_processor.is_started():
                data_processor.join()
        for data_writer in self._data_writers.values():
            if data_writer.is_started():
                data_writer.join()

        self._watchdog.stop()

        # Delete already recorded data if requested and recording actually running
        if self.delete_on_error and self._is_recording:
            shutil.rmtree(self.cur_simrun_dir, ignore_errors=True)
            self._logger.info('Erased dirty data.')

        self._reset()

    def _reset(self):
        """
        Resets the CDR to be ready for the next recording.
        """
        if self._world is not None:
            # Cleanup
            self._reset_sync_mode()
            self(None)

        # Prepare next recording
        self._cur_sim_run += 1
        self._init_variables()


class NoRecorder(Recorder):
    """
    Recorder class recording nothing. This class just exposes the same interface as a regular `Recorder` while
    doing no work. This enables a very easy optional integration of the CARLA Data Recorder as a context manager:

    ```python
    recorder = CARLADataRecorder('<path/to/results/dir', client) if args.enable_recorder else NoRecorder()
    with recorder:
        # do something
        world.tick()
    ```
    """

    def __enter__(self):
        pass

    def __call__(self, world: Optional[carla.World]) -> Self:
        return self  # pragma: no cover

    def __exit__(self, exc_type, exc_val, exc_tb):
        pass

    def start_recording(self, world: Optional[carla.World]):
        pass

    def stop_recording(self):
        pass

    def set_num_simulation_runs(self, num_sim_runs: int):
        pass

    def set_simulation_run_name(self, sim_run_name: str):
        pass
