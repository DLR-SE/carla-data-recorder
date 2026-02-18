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


import contextlib
import logging
import os
import tempfile
import time
from pathlib import Path
from typing import Dict, List, Tuple

import carla
import numpy as np
import pytest
from PIL import Image

import cdr.utils.environment_objects as env_objs
from cdr import CARLADataRecorder
from cdr.data_recorder import DEFAULT_RECORDING_OUTPUTS
from tests.mocks.carla import ActorBlueprintMock, ClientMock, LidarMeasurementMock, SemanticLidarMeasurementMock


def recorder_log_exists(recording_dir: Path):
    for filename in os.listdir(recording_dir):
        if filename.startswith('recorder_log') and filename.endswith('.txt'):
            return True
    return False


@pytest.fixture
def temp_dir():
    with tempfile.TemporaryDirectory() as temp_dir:
        yield Path(temp_dir) / 'test_runs'


@pytest.fixture
def make_client(monkeypatch):
    def _client(actors_w_attributes: List[Tuple[str, Dict[str, str]]]):
        client = ClientMock('', 0)
        world = client.get_world()
        for actor_type, actor_attributes in actors_w_attributes:
            world.spawn_actor(ActorBlueprintMock(actor_type, actor_attributes), carla.Transform())
        monkeypatch.setattr(carla, 'Client', lambda *args: client)
        monkeypatch.setattr(carla, 'SemanticLidarMeasurement', SemanticLidarMeasurementMock)
        monkeypatch.setattr(carla, 'LidarMeasurement', LidarMeasurementMock)
        monkeypatch.setattr(env_objs, 'get_bp_bb_lut', lambda world: ([], np.array([])))
        return client
    return _client


@pytest.fixture
def client_no_actors(make_client) -> ClientMock:
    return make_client([])


@pytest.fixture
def client_two_actors(make_client) -> ClientMock:
    return make_client([
        ('vehicle.citroen.c3', {'role_name': 'hero'}),
        ('vehicle.tesla.model3', None)
    ])


def test_no_world_set_context_manager(temp_dir, client_no_actors):
    with pytest.raises(RuntimeError, match='Invoked start of a new recording without setting the world'):
        cdr = CARLADataRecorder(temp_dir, client_no_actors, timeout=0.001, verbose=True)
        with cdr:
            pass


def test_no_world_set_manual(temp_dir, client_no_actors):
    with pytest.raises(RuntimeError, match='Invoked start of a new recording without setting the world'):
        cdr = CARLADataRecorder(temp_dir, client_no_actors, timeout=0.001, verbose=True)
        cdr.start_recording()


@pytest.mark.parametrize('override_existing_data', [True, False])
@pytest.mark.parametrize('file_or_dir', ['sensor_configuration.json', 'metadata.json', 'sensor', 'ground_truth'])
def test_override_existing_data_single_run(temp_dir, client_two_actors,
                                           override_existing_data: bool, file_or_dir: str):
    world = client_two_actors.get_world()
    temp_dir.mkdir()
    if file_or_dir.endswith('.json'):
        (temp_dir / file_or_dir).touch()
    else:
        (temp_dir / file_or_dir).mkdir()

    if override_existing_data:
        mngr = contextlib.nullcontext()
    else:
        mngr = pytest.raises(RuntimeError, match='current simulation run ".*" already exists')
    with mngr:
        cdr = CARLADataRecorder(temp_dir, client_two_actors, override_existing_data=override_existing_data, verbose=True)
        with cdr(world):
            pass


@pytest.mark.parametrize('override_existing_data', [True, False])
def test_override_existing_data_multiple_runs(temp_dir, client_two_actors, override_existing_data: bool):
    world = client_two_actors.get_world()
    sim_run_name = 'sim_run-0'
    temp_dir.mkdir()
    (temp_dir / sim_run_name).mkdir()

    if override_existing_data:
        mngr = contextlib.nullcontext()
    else:
        mngr = pytest.raises(RuntimeError, match='current simulation run ".*" already exists')
    with mngr:
        cdr = CARLADataRecorder(temp_dir, client_two_actors, override_existing_data=override_existing_data, verbose=True)
        cdr.set_num_simulation_runs(2)
        cdr.set_simulation_run_name(sim_run_name)
        with cdr(world):
            pass


def test_annotations_traverse_translucency_invalid_version(temp_dir, client_two_actors, caplog):
    world = client_two_actors.get_world()
    client_two_actors.client_version = '0.9.15'
    cdr = CARLADataRecorder(temp_dir, client_two_actors, annotations_traverse_translucency=True, verbose=True)
    cdr._logger.propagate = True
    with cdr(client_two_actors.get_world()):
        pass

    expected_log_msg = ('Requested to explicitly set annotations_traverse_translucency=True, requiring CARLA >=0.9.16, '
                        f'but CARLA client version is 0.9.15. -> Setting ignored.')
    assert (cdr._logger.name, logging.WARNING, expected_log_msg) in caplog.record_tuples


def test_no_ego_vehicle(temp_dir, client_no_actors):
    world = client_no_actors.get_world()
    with pytest.raises(ValueError, match='Could not find vehicle'):
        cdr = CARLADataRecorder(temp_dir, client_no_actors, verbose=True)
        with cdr(world):
            pass


def test_non_optimal_ego_vehicle(temp_dir, client_no_actors, caplog):
    world = client_no_actors.get_world()
    world.spawn_actor(ActorBlueprintMock('vehicle.tesla.model3', {'role_name': 'hero'}), carla.Transform())

    cdr = CARLADataRecorder(temp_dir, client_no_actors, sensor_config=str(Path('tests') / 'test_configs' / '1cam.json'),
                            verbose=True)
    cdr._logger.propagate = True
    with cdr(world):
        pass

    expected_log_msg = ('Optimal CARLA vehicle for specified sensor configuration is "vehicle.citroen.c3" but current '
                        'ego vehicle is "vehicle.tesla.model3".')
    assert (cdr._logger.name, logging.WARNING, expected_log_msg) in caplog.record_tuples


def test_sensors_not_found(temp_dir, client_no_actors):
    world = client_no_actors.get_world()
    with pytest.raises(ValueError, match='sensor configuration ".*" could not be found'):
        cdr = CARLADataRecorder(temp_dir, client_no_actors, sensor_config='unknown_sensors')
        with cdr(world):
            pass


def test_interrupt_illegal(temp_dir, client_two_actors):
    world = client_two_actors.get_world()
    with pytest.raises(KeyboardInterrupt):
        cdr = CARLADataRecorder(temp_dir, client_two_actors,
                                sensor_config=str(Path('tests') / 'test_configs' / '1cam.json'),
                                simulation_fps=20, verbose=True)
        with cdr(world):
            world.tick()
            raise KeyboardInterrupt('test: illegally interrupting CDR')

    assert not os.path.isdir(temp_dir) or (
        # it may not be deleted, if the log-file is blocking the deletion
        len(os.listdir(temp_dir)) == 1 and recorder_log_exists(temp_dir)
    )


def test_interrupt_permitted(temp_dir, client_two_actors):
    world = client_two_actors.get_world()
    with pytest.raises(KeyboardInterrupt):
        cdr = CARLADataRecorder(temp_dir, client_two_actors,
                                sensor_config=str(Path('tests') / 'test_configs' / '1cam.json'),
                                simulation_fps=20, allow_user_interrupt=True, verbose=True)
        with cdr(world):
            world.tick()
            raise KeyboardInterrupt('test: interrupting CDR')

    assert recorder_log_exists(temp_dir)
    assert os.path.isdir(temp_dir / 'sensor')
    assert os.path.isdir(temp_dir / 'ground_truth')


def test_sensor_tickrate_not_multiple_warning(temp_dir, client_two_actors, caplog):
    world = client_two_actors.get_world()
    sim_fps = 15
    cdr = CARLADataRecorder(temp_dir, client_two_actors, sensor_config=str(Path('tests') / 'test_configs' / '1cam.json'),
                            simulation_fps=sim_fps, verbose=True)
    cdr._logger.propagate = True
    with cdr(world):
        pass

    expected_log_msg = (f'Simulation tick rate (={sim_fps}) is not a multiple of "CAM_FRONT"\'s capture_frequency (=10). '
                        'This leads to frames not being captured at the desired interval.')
    assert (cdr._logger.name, logging.WARNING, expected_log_msg) in caplog.record_tuples


def test_recording_still_running(temp_dir, client_two_actors):
    world = client_two_actors.get_world()
    cdr = CARLADataRecorder(temp_dir, client_two_actors,
                            sensor_config=str(Path('tests') / 'test_configs' / '0sensors.json'), verbose=True)
    with pytest.raises(RuntimeError, match='recording is still running'):
        with cdr(world):
            cdr.start_recording()


def test_no_recording_running(temp_dir, client_two_actors):
    cdr = CARLADataRecorder(temp_dir, client_two_actors,
                            sensor_config=str(Path('tests') / 'test_configs' / '0sensors.json'), verbose=True)
    with pytest.raises(RuntimeError, match='Invoked stop of the current recording without any recording actually running'):
        cdr.stop_recording()


def test_worker_crashed_ctx_manager(temp_dir, client_two_actors, caplog):
    world = client_two_actors.get_world()
    cdr = CARLADataRecorder(temp_dir, client_two_actors,
                            sensor_config=str(Path('tests') / 'test_configs' / '1cam.json'), verbose=True)
    cdr._logger.propagate = True

    with cdr(world):
        world.tick()
        try:
            world.invalid_tick()
            time.sleep(0.5)  # give the watchdog the required time to catch the error from the worker
        except KeyboardInterrupt:
            pass

    expected_log_msg = 'Worker process crashed -> Kill recording.'
    assert (cdr._logger.name, logging.ERROR, expected_log_msg) in caplog.record_tuples


def test_worker_crashed_manual(temp_dir, client_two_actors, caplog):
    world = client_two_actors.get_world()
    cdr = CARLADataRecorder(temp_dir, client_two_actors,
                            sensor_config=str(Path('tests') / 'test_configs' / '1cam.json'), verbose=True)
    cdr._logger.propagate = True

    cdr.start_recording(world)
    world.tick()
    try:
        world.invalid_tick()
    except KeyboardInterrupt:
        pass

    with pytest.raises(RuntimeError, match='Error occurred in'):
        cdr.stop_recording()

    expected_log_msg = 'An error occurred when calling CARLADataRecorder.stop_recording'
    assert (cdr._logger.name, logging.ERROR, expected_log_msg) in caplog.record_tuples


def assert_data_complete(data_dir: Path, sensor_names: Dict[str, List[str]], sensor_annotations: Dict[str, List[str]],
                         num_ticks: int, check_trajectories: bool = True):
    assert os.path.exists(data_dir / 'sensor_configuration.json')
    assert os.path.exists(data_dir / 'metadata.json')

    dirs_to_check = [data_dir / 'ground_truth' / 'trajectories'] if check_trajectories else []
    for sensor_type, names in sensor_names.items():
        for name in names:
            dirs_to_check.append(data_dir / 'sensor' / sensor_type / name)
    for annotation_type, names in sensor_annotations.items():
        for name in names:
            dirs_to_check.append(data_dir / 'ground_truth' / annotation_type / name)

    for data_dir in dirs_to_check:
        assert os.path.isdir(data_dir)
        assert len(os.listdir(data_dir)) == num_ticks


def assert_data_complete_mono_files(data_dir: Path,
                                    sensor_names: Dict[str, List[str]], sensor_annotations: Dict[str, List[str]],
                                    file_extensions: Dict[str, str], check_trajectories: bool = True):
    assert os.path.exists(data_dir / 'sensor_configuration.json')
    assert os.path.exists(data_dir / 'metadata.json')

    files_to_check = [data_dir / 'ground_truth' / 'trajectories.parquet'] if check_trajectories else []
    for sensor_type, names in sensor_names.items():
        for name in names:
            files_to_check.append(data_dir / 'sensor' / sensor_type / f'{name}.{file_extensions[sensor_type]}')
    for annotation_type, names in sensor_annotations.items():
        for name in names:
            files_to_check.append(data_dir / 'ground_truth' / annotation_type / f'{name}.{file_extensions[annotation_type]}')

    for file in files_to_check:
        assert os.path.exists(file)


@pytest.mark.parametrize('num_ticks', [0, 1, 10, 50, 200])
def test_x_ticks(temp_dir, client_two_actors, num_ticks: int):
    world = client_two_actors.get_world()

    cdr = CARLADataRecorder(temp_dir, client_two_actors,
                           sensor_config=str(Path('tests') / 'test_configs' / '1cam_1lidar.json'),
                           simulation_fps=20, verbose=True)
    with cdr(world):
        for _ in range(num_ticks):
            world.tick()

    expected_cameras = ['CAM_FRONT']
    expected_sensors = {'camera': expected_cameras, 'lidar': ['LIDAR_TOP']}
    excpected_annotations = {annotation_type: expected_cameras for annotation_type in
                             ['2d_bbs', '3d_bbs', 'depth', 'inst_seg', 'sem_seg']}

    assert recorder_log_exists(temp_dir)
    assert_data_complete(temp_dir, expected_sensors, excpected_annotations, num_ticks)


def test_huge_frame_ids(temp_dir, client_two_actors):
    num_ticks = 20
    world = client_two_actors.get_world()
    world.frame = 2**31 - num_ticks // 2

    cdr = CARLADataRecorder(temp_dir, client_two_actors,
                            sensor_config=str(Path('tests') / 'test_configs' / '1cam_1lidar.json'),
                            simulation_fps=20, verbose=True)
    with cdr(world):
        for _ in range(num_ticks):
            world.tick()

    expected_cameras = ['CAM_FRONT']
    expected_sensors = {'camera': expected_cameras, 'lidar': ['LIDAR_TOP']}
    excpected_annotations = {annotation_type: expected_cameras for annotation_type in
                             ['2d_bbs', '3d_bbs', 'depth', 'inst_seg', 'sem_seg']}

    assert recorder_log_exists(temp_dir)
    assert_data_complete(temp_dir, expected_sensors, excpected_annotations, num_ticks)


def test_from_config(temp_dir, client_two_actors):
    num_ticks = 10
    world = client_two_actors.get_world()

    cdr = CARLADataRecorder.from_config(temp_dir, client_two_actors,
                                        Path('tests') / 'test_configs' / 'test_from_config.json')
    with cdr(world):
        for _ in range(num_ticks):
            world.tick()

    expected_cameras = ['CAM_FRONT', 'CAM_BACK']
    expected_sensors = {'camera': expected_cameras, 'lidar': ['LIDAR_TOP'],
                        'gnss': ['GPS'], 'imu': ['IMU']}
    excpected_annotations = {annotation_type: expected_cameras for annotation_type in
                             ['2d_bbs', '3d_bbs', 'depth', 'inst_seg', 'sem_seg']}

    assert recorder_log_exists(temp_dir)
    assert_data_complete(temp_dir, expected_sensors, excpected_annotations, num_ticks)


def test_from_config_sensors_not_found(temp_dir, client_no_actors):
    with pytest.raises(FileNotFoundError, match='sensor configuration file ".*" could not be found'):
        CARLADataRecorder.from_config(temp_dir, client_no_actors,
                                      Path('tests') / 'test_configs' / 'test_from_config_sensors_not_found.json')


def test_from_config_mono_formats(temp_dir, client_two_actors):
    num_ticks = 10
    world = client_two_actors.get_world()

    cdr = CARLADataRecorder.from_config(temp_dir, client_two_actors,
                                        Path('tests') / 'test_configs' / 'test_from_config_mono_formats.json')
    with cdr(world):
        for _ in range(num_ticks):
            world.tick()

    expected_cameras = ['CAM_FRONT', 'CAM_BACK']
    expected_sensors = {'camera': expected_cameras, 'lidar': ['LIDAR_TOP'],
                        'gnss': ['GPS'], 'imu': ['IMU']}
    expected_annotations = {annotation_type: expected_cameras for annotation_type in
                             ['2d_bbs', '3d_bbs', 'depth', 'inst_seg', 'sem_seg']}
    expected_extensions = {k: 'parquet' for k in list(expected_sensors.keys()) + list(expected_annotations.keys())}
    expected_extensions['camera'] = 'mp4'  # all parquet except camera data

    assert recorder_log_exists(temp_dir)
    assert_data_complete_mono_files(temp_dir, expected_sensors, expected_annotations, expected_extensions)


@pytest.mark.parametrize('num_runs', [2, 5])
@pytest.mark.parametrize('with_names', [True, False])
def test_multiple_runs(temp_dir, client_two_actors, num_runs: int, with_names: bool):
    num_ticks = 10
    world = client_two_actors.get_world()

    cdr = CARLADataRecorder(temp_dir, client_two_actors,
                            sensor_config=str(Path('tests') / 'test_configs' / '1cam_1lidar.json'),
                            simulation_fps=20, verbose=True)
    cdr.set_num_simulation_runs(num_runs)

    if with_names:
        sim_run_names = [f'sim_run-{i:03d}' for i in range(num_runs)]
    else:
        sim_run_names = [f'{i+1:06d}' for i in range(num_runs)]

    for sim_run_name in sim_run_names:
        if with_names:
            cdr.set_simulation_run_name(sim_run_name)
        with cdr(world):
            for _ in range(num_ticks):
                world.tick()

    expected_cameras = ['CAM_FRONT']
    expected_sensors = {'camera': expected_cameras, 'lidar': ['LIDAR_TOP']}
    excpected_annotations = {annotation_type: expected_cameras for annotation_type in
                             ['2d_bbs', '3d_bbs', 'depth', 'inst_seg', 'sem_seg']}

    assert recorder_log_exists(temp_dir)
    sim_run_dir_names = sorted(list(filter(lambda path: os.path.isdir(temp_dir / path), os.listdir(temp_dir))))
    assert len(sim_run_dir_names) == num_runs
    assert sorted(sim_run_names) == sim_run_dir_names
    for sim_run_dir_name in sim_run_dir_names:
        assert_data_complete(temp_dir / sim_run_dir_name, expected_sensors, excpected_annotations, num_ticks)


def test_no_sensors_only_trajectories(temp_dir, client_two_actors):
    num_ticks = 10
    world = client_two_actors.get_world()

    cdr = CARLADataRecorder(temp_dir, client_two_actors,
                            sensor_config=str(Path('tests') / 'test_configs' / '0sensors.json'),
                            simulation_fps=20, verbose=True)
    with cdr(world):
        for _ in range(num_ticks):
            world.tick()

    assert recorder_log_exists(temp_dir)
    assert not os.path.isdir(temp_dir / 'sensor')
    assert len(os.listdir(temp_dir / 'ground_truth')) == 1
    assert_data_complete(temp_dir, {}, {}, num_ticks)


@pytest.mark.parametrize('cam_outputs', [
    [],
    ['RGB'],
    ['RGB', 'SEMSEG', 'INSTSEG'],
    ['RGB',                      'DEPTH'],
    ['RGB',                               'BB2D', 'BB3D'],
    ['RGB', 'SEMSEG', 'INSTSEG', 'DEPTH'],
    ['RGB',                      'DEPTH', 'BB2D', 'BB3D'],
    ['RGB', 'SEMSEG', 'INSTSEG', 'DEPTH', 'BB2D', 'BB3D'],
    ['DEPTH'],
])
def test_one_camera_various_outputs(temp_dir, client_two_actors, cam_outputs: List[str]):
    num_ticks = 10
    world = client_two_actors.get_world()

    cam_outputs = [f'CAMERA:{output_type}' for output_type in cam_outputs]
    recording_outputs = {output_type: DEFAULT_RECORDING_OUTPUTS[output_type] for output_type in cam_outputs}

    cdr = CARLADataRecorder(temp_dir, client_two_actors,
                            sensor_config=str(Path('tests') / 'test_configs' / '1cam.json'), simulation_fps=20,
                            recording_outputs=recording_outputs, verbose=True)
    with cdr(world):
        for _ in range(num_ticks):
            world.tick()

    expected_cameras = ['CAM_FRONT']
    expected_sensors = {'camera': expected_cameras} if 'CAMERA:RGB' in cam_outputs else {}
    annotation_types = []
    annotation_types += ['2d_bbs'] if 'CAMERA:BB2D' in cam_outputs else []
    annotation_types += ['3d_bbs'] if 'CAMERA:BB3D' in cam_outputs else []
    annotation_types += ['depth'] if 'CAMERA:DEPTH' in cam_outputs else []
    annotation_types += ['sem_seg'] if 'CAMERA:SEMSEG' in cam_outputs else []
    annotation_types += ['inst_seg'] if 'CAMERA:INSTSEG' in cam_outputs else []
    excpected_annotations = {annotation_type: expected_cameras for annotation_type in annotation_types}

    assert recorder_log_exists(temp_dir)
    assert_data_complete(temp_dir, expected_sensors, excpected_annotations, num_ticks, check_trajectories=False)


def test_ssaa(temp_dir, client_two_actors):
    num_ticks = 10
    world = client_two_actors.get_world()

    cdr = CARLADataRecorder(temp_dir, client_two_actors,
                            sensor_config=str(Path('tests') / 'test_configs' / '1cam.json'), simulation_fps=20,
                            enable_ssaa=True, verbose=True)
    with cdr(world):
        for _ in range(num_ticks):
            world.tick()

    expected_cameras = ['CAM_FRONT']
    expected_sensors = {'camera': expected_cameras}
    excpected_annotations = {annotation_type: expected_cameras for annotation_type in
                             ['2d_bbs', '3d_bbs', 'depth', 'inst_seg', 'sem_seg']}

    assert recorder_log_exists(temp_dir)
    assert_data_complete(temp_dir, expected_sensors, excpected_annotations, num_ticks)
    for cam in expected_cameras:
        cam_dir = temp_dir / 'sensor' / 'camera' / cam
        for image_filename in os.listdir(cam_dir):
            with Image.open(cam_dir / image_filename) as image:
                assert image.width == 10 and image.height == 10


def test_static_camera(temp_dir, client_two_actors):
    num_ticks = 10
    world = client_two_actors.get_world()

    cdr = CARLADataRecorder(temp_dir, client_two_actors,
                            sensor_config=str(Path('tests') / 'test_configs' / '1cam.json'),
                            static_sensor_config=str(Path('tests') / 'test_configs' / 'static_cam.json'),
                            simulation_fps=20, verbose=True)
    with cdr(world):
        for _ in range(num_ticks):
            world.tick()

    expected_cameras = ['CAM_FRONT', 'CAM_STATIC']
    expected_sensors = {'camera': expected_cameras}
    excpected_annotations = {annotation_type: expected_cameras for annotation_type in
                             ['2d_bbs', '3d_bbs', 'depth', 'inst_seg', 'sem_seg']}

    assert recorder_log_exists(temp_dir)
    assert_data_complete(temp_dir, expected_sensors, excpected_annotations, num_ticks)


def test_static_gnss_imu(temp_dir, client_two_actors, caplog):
    num_ticks = 10
    world = client_two_actors.get_world()

    cdr = CARLADataRecorder(temp_dir, client_two_actors,
                            sensor_config=str(Path('tests') / 'test_configs' / '0sensors.json'),
                            static_sensor_config=str(Path('tests') / 'test_configs' / 'static_gnss_imu.json'),
                            simulation_fps=20, verbose=True)
    cdr._logger.propagate = True
    with cdr(world):
        for _ in range(num_ticks):
            world.tick()

    assert recorder_log_exists(temp_dir)
    assert_data_complete(temp_dir, {'gnss': ['GPS'], 'imu': ['IMU']}, {}, num_ticks)

    expected_log_msgs = [
        'Was spawning a static GNSS sensor (GPS) really intended?',
        'Was spawning a static IMU sensor (IMU) really intended?',
    ]
    for expected_log_msg in expected_log_msgs:
        assert (cdr._logger.name, logging.WARNING, expected_log_msg) in caplog.record_tuples


def test_static_config_id_collision(temp_dir, client_two_actors):
    world = client_two_actors.get_world()

    cdr = CARLADataRecorder(temp_dir, client_two_actors,
                            sensor_config=str(Path('tests') / 'test_configs' / '1cam.json'),
                            static_sensor_config=str(Path('tests') / 'test_configs' / '1cam.json'),
                            simulation_fps=20, verbose=True)
    with pytest.raises(ValueError, match='Sensor configurations cannot contain identical sensor IDs, but found collisions'):
        cdr.start_recording(world)
