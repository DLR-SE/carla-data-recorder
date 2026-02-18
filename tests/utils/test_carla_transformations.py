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
import pytest
from pyquaternion import Quaternion

from cdr.utils.carla_transformations import (array_to_carla_location, carla_location_to_array,
                                             carla_rotation_to_quaternion, carla_vector3d_to_array,
                                             array_to_carla_vector3d, decode_instseg_ids, encode_instseg_ids,
                                             gnss_to_dict, image_to_array, image_to_dict, imu_to_dict,
                                             point_cloud_to_dict, quaternion_to_carla_rotation, world_snapshot_to_dict)
from tests.helpers import is_close
from tests.mocks.carla import (ActorBlueprintMock, EnvironmentObjectMock, GNSSMeasurementMock,
                               IMUMeasurementMock, ImageMock, LidarMeasurementMock, SemanticLidarMeasurementMock,
                               WorldMock)


#################################################
#####        CARLA BASIC DATA TYPES         #####
#################################################


##### VECTOR3D AND LOCATION TRANSFORMATIONS #####


@pytest.fixture
def xyz_test_values():
    return [
        [ 0.  ,     0.   ,   0.  ],
        [ 0.1 ,    -0.4  ,   0.  ],
        [-1.4 ,    20.1  ,  13.  ],
        [ 2.71,    -3.213,   3.23],
        [-4.41, -1000.2  , -10.12]
    ]


def test_carla_vector3d_to_array(xyz_test_values):
    for x, y, z in xyz_test_values:
        assert is_close(carla_vector3d_to_array(carla.Vector3D(x, y, z)), np.array((x, y, z)))


def test_array_to_carla_vector3d(xyz_test_values):
    for x, y, z in xyz_test_values:
        assert is_close(array_to_carla_vector3d(np.array((x, y, z))), carla.Vector3D(x, y, z))


def test_carla_vector3d_to_array_to_carla_vector3d(xyz_test_values):
    for x, y, z in xyz_test_values:
        xyz = carla.Vector3D(x, y, z)
        assert is_close(array_to_carla_vector3d(carla_vector3d_to_array(xyz)), xyz)


def test_array_to_carla_vector3d_to_array(xyz_test_values):
    for x, y, z in xyz_test_values:
        xyz = np.array((x, y, z))
        assert is_close(carla_vector3d_to_array(array_to_carla_vector3d(xyz)), xyz)


def test_carla_location_to_array(xyz_test_values):
    for x, y, z in xyz_test_values:
        assert is_close(carla_location_to_array(carla.Location(x, y, z)), np.array((x, -y, z)))


def test_array_to_carla_location(xyz_test_values):
    for x, y, z in xyz_test_values:
        assert is_close(array_to_carla_location(np.array((x, y, z))), carla.Vector3D(x, -y, z))


def test_carla_location_to_array_to_carla_location(xyz_test_values):
    for x, y, z in xyz_test_values:
        xyz = carla.Location(x, y, z)
        assert is_close(array_to_carla_location(carla_location_to_array(xyz)), xyz)


def test_array_to_carla_location_to_array(xyz_test_values):
    for x, y, z in xyz_test_values:
        xyz = np.array((x, y, z))
        assert is_close(carla_location_to_array(array_to_carla_location(xyz)), xyz)


##### ROTATION TRANSFORMATIONS #####


@pytest.fixture
def rpy_test_values():
    return [
        [ 0.  ,   0.  ,    0.  ],
        [ 0.  ,   0.  ,   90.  ],
        [-3.17,  10.21,  -30.94],
        [ 0.54,   0.23,  177.77],
    ]

@pytest.fixture
def wxyz_test_values():
    def _wxyz_test_values(is_camera: bool):
        if is_camera:
            return [
                [ 0.5       , -0.5       ,  0.5       , -0.5       ],
                [ 0.        ,  0.        ,  0.70710678, -0.70710678],
                [ 0.65684243, -0.56659608,  0.30056717, -0.39646725],
                [-0.48874675,  0.49158225,  0.50630008, -0.51296563],
            ]
        else:
            return [
                [ 1.        ,  0.        ,  0.        ,  0.        ],
                [ 0.70710678,  0.        ,  0.        , -0.70710678],
                [ 0.96023647, -0.00282687, -0.09307321,  0.26320204],
                [ 0.01946836, -0.00191503, -0.00475053, -0.99979735],
            ]
    return _wxyz_test_values


@pytest.mark.parametrize('is_camera', [False, True])
def test_carla_rotation_to_quaternion(rpy_test_values, wxyz_test_values, is_camera: bool):
    wxyz_test_values = wxyz_test_values(is_camera)
    for (roll, pitch, yaw), (w, x, y, z) in zip(rpy_test_values, wxyz_test_values):
        assert is_close(carla_rotation_to_quaternion(carla.Rotation(roll=roll, pitch=pitch, yaw=yaw), is_camera),
                        Quaternion(w, x, y, z))


@pytest.mark.parametrize('is_camera', [False, True])
def test_quaternion_to_carla_rotation(rpy_test_values, wxyz_test_values, is_camera: bool):
    wxyz_test_values = wxyz_test_values(is_camera)
    for (w, x, y, z), (roll, pitch, yaw) in zip(wxyz_test_values, rpy_test_values):
        assert is_close(quaternion_to_carla_rotation(Quaternion(w, x, y, z), is_camera),
                        carla.Rotation(roll=roll, pitch=pitch, yaw=yaw))


@pytest.mark.parametrize('is_camera', [False, True])
def test_carla_rotation_to_quaternion_to_carla_rotation(rpy_test_values, is_camera: bool):
    for roll, pitch, yaw in rpy_test_values:
        rot = carla.Rotation(roll=roll, pitch=pitch, yaw=yaw)
        assert is_close(quaternion_to_carla_rotation(carla_rotation_to_quaternion(rot, is_camera), is_camera), rot)


@pytest.mark.parametrize('is_camera', [False, True])
def test_quaternion_to_carla_rotation_to_quaternion(wxyz_test_values, is_camera: bool):
    wxyz_test_values = wxyz_test_values(is_camera)
    for w, x, y, z in wxyz_test_values:
        quat = Quaternion(w, x, y, z)
        assert is_close(carla_rotation_to_quaternion(quaternion_to_carla_rotation(quat, is_camera), is_camera), quat)



#################################################
#####           SENSOR DATA TYPES           #####
#################################################


@pytest.fixture
def frame():
    return 1


@pytest.fixture
def timestamp():
    return 0.5


@pytest.fixture
def transform():
    return carla.Transform()


##### INSTANCE SEGMENTATION TRANSFORMATIONS #####


@pytest.fixture
def instseg_ids():
    return np.array([
        [[  0,  14],
        [ 12,   9]],
        [[  0,  27],
        [237,   0]],
        [[  0,  35],
        [122, 197]]
    ], np.uint8)


@pytest.fixture
def actor_ids():
    return np.array([
        [    0,  8987],
        [31469, 50432]
    ], np.uint16)


def test_decode_instseg_ids(instseg_ids, actor_ids):
    assert np.all(decode_instseg_ids(instseg_ids[1], instseg_ids[2]) == actor_ids)


def test_encode_instseg_ids(instseg_ids, actor_ids):
    green, blue = encode_instseg_ids(actor_ids)
    assert np.all(green == instseg_ids[1])
    assert np.all(blue == instseg_ids[2])


def test_decode_encode_instseg_ids(instseg_ids):
    green, blue = encode_instseg_ids(decode_instseg_ids(instseg_ids[1], instseg_ids[2]))
    assert np.all(green == instseg_ids[1])
    assert np.all(blue == instseg_ids[2])


def test_encode_decode_instseg_ids(actor_ids):
    assert np.all(decode_instseg_ids(*encode_instseg_ids(actor_ids)) == actor_ids)


##### IMAGE TRANSFORMATIONS #####


@pytest.fixture
def bgra() -> np.ndarray:
    return np.array([
        [[0, 1, 2],
         [3, 4, 5]],
        [[10, 20, 30],
         [40, 50, 60]],
        [[0, 100, 200],
         [0, 100, 200]],
        [[255, 255, 255],
         [255, 255, 255]]
    ], np.uint8).transpose((1, 2, 0))


@pytest.fixture
def rgb() -> np.ndarray:
    return np.array([
        [[0, 100, 200],
         [0, 100, 200]],
        [[10, 20, 30],
         [40, 50, 60]],
        [[0, 1, 2],
         [3, 4, 5]],
    ], np.uint8).transpose((1, 2, 0))


def test_image_to_array(frame, timestamp, transform,
                        bgra, rgb):
    assert np.all(image_to_array(ImageMock(bgra, frame, timestamp, transform)) == rgb)


def test_image_to_dict(frame, timestamp, transform,
                       bgra, rgb):
    image = ImageMock(bgra, frame, timestamp, transform)
    res = image_to_dict(image)
    assert sorted(res.keys()) == sorted(['frame_id', 'timestamp', 'image_array', 'location', 'rotation'])
    assert res['frame_id'] == frame
    assert res['timestamp'] == timestamp
    assert is_close(res['image_array'], rgb)
    assert is_close(res['location'], np.zeros(3))
    assert is_close(res['rotation'], Quaternion(0.5, -0.5, 0.5, -0.5))


##### LIDAR TRANSFORMATIONS #####


@pytest.fixture
def xyz() -> np.ndarray:
    return np.array([
        [ 0.   ,   0.  ,  0. ],
        [ 0.1  ,  10.2 , -0.5],
        [65.231, -22.67,  9.2],
    ], np.float32)


@pytest.fixture
def xyz_expected() -> np.ndarray:
    return np.array([
        [ 0.   ,   0.  ,  0. ],
        [ 0.1  , -10.2 , -0.5],
        [65.231,  22.67,  9.2],
    ], np.float32)


@pytest.fixture
def intensity() -> np.ndarray:
    return np.array([0., 1., 0.2], np.float32)


@pytest.fixture
def intensity_expected() -> np.ndarray:
    return np.array([0., 1., 0.2], np.float32)


def test_point_cloud_to_dict_default_lidar(frame, timestamp, transform,
                                           xyz, intensity, xyz_expected, intensity_expected,
                                           monkeypatch):
    monkeypatch.setattr(carla, 'LidarMeasurement', LidarMeasurementMock)

    lidar_measurement = LidarMeasurementMock(xyz, intensity, frame, timestamp, transform)
    res = point_cloud_to_dict(lidar_measurement)

    assert sorted(res.keys()) == sorted(['frame_id', 'timestamp', 'location', 'rotation', 'vertices', 'vertex_attributes'])
    assert sorted(res['vertex_attributes'].keys()) == ['intensity']
    assert res['frame_id'] == frame
    assert res['timestamp'] == timestamp
    assert is_close(res['location'], np.zeros(3))
    assert is_close(res['rotation'], Quaternion())
    assert is_close(res['vertices'], xyz_expected)
    assert is_close(res['vertex_attributes']['intensity'], intensity_expected)


@pytest.fixture
def cosine_incident_angle() -> np.ndarray:
    return np.array((0., -10.2, 0.564), np.float32)


@pytest.fixture
def cosine_incident_angle_expected() -> np.ndarray:
    return np.array((0., -10.2, 0.564), np.float32)


@pytest.fixture
def actor_id() -> np.ndarray:
    return np.array((0, 10, 154), np.uint32)


@pytest.fixture
def actor_id_expected() -> np.ndarray:
    return np.array((0, 10, 154), np.uint32)


@pytest.fixture
def semseg_id() -> np.ndarray:
    return np.array((1, 14, 13), np.uint32)


@pytest.fixture
def semseg_id_expected() -> np.ndarray:
    return np.array((1, 14, 13), np.uint32)


def test_point_cloud_to_dict_semantic_lidar(frame, timestamp, transform,
                                            xyz, cosine_incident_angle, actor_id, semseg_id,
                                            xyz_expected, cosine_incident_angle_expected, actor_id_expected, semseg_id_expected,
                                            monkeypatch):
    monkeypatch.setattr(carla, 'SemanticLidarMeasurement', SemanticLidarMeasurementMock)

    semantic_lidar_measurement = SemanticLidarMeasurementMock(xyz, cosine_incident_angle, actor_id, semseg_id,
                                                              frame, timestamp, transform)
    res = point_cloud_to_dict(semantic_lidar_measurement)

    assert sorted(res.keys()) == sorted(['frame_id', 'timestamp', 'location', 'rotation', 'vertices', 'vertex_attributes'])
    assert sorted(res['vertex_attributes'].keys()) == sorted(['cosine_incident_angle', 'actor_id', 'semseg_id'])
    assert res['frame_id'] == frame
    assert res['timestamp'] == timestamp
    assert is_close(res['location'], np.zeros(3))
    assert is_close(res['rotation'], Quaternion())
    assert is_close(res['vertices'], xyz_expected)
    assert is_close(res['vertex_attributes']['cosine_incident_angle'], cosine_incident_angle_expected)
    assert is_close(res['vertex_attributes']['actor_id'], actor_id_expected)
    assert is_close(res['vertex_attributes']['semseg_id'], semseg_id_expected)


##### OTHER SENSOR TRANSFORMATIONS #####


@pytest.fixture
def latitude():
    return 0.00013231


@pytest.fixture
def longitude():
    return 0.0058642


@pytest.fixture
def altitude():
    return 12.3213


def test_gnss_to_dict(frame, timestamp, transform,
                      latitude, longitude, altitude):
    gnss_measurement = GNSSMeasurementMock(latitude, longitude, altitude, frame, timestamp, transform)
    res = gnss_to_dict(gnss_measurement)

    assert sorted(res.keys()) == sorted(['frame_id', 'timestamp', 'location', 'rotation', 'latitude', 'longitude', 'altitude'])
    assert res['frame_id'] == frame
    assert res['timestamp'] == timestamp
    assert is_close(res['location'], np.zeros(3))
    assert is_close(res['rotation'], Quaternion())
    assert is_close(res['latitude'], latitude)
    assert is_close(res['longitude'], longitude)
    assert is_close(res['altitude'], altitude)


@pytest.fixture
def acceleration():
    return carla.Vector3D(5.312, -1.23954, 9.845412)


@pytest.fixture
def acceleration_expected():
    return np.array((5.312, 1.23954, 9.845412))


@pytest.fixture
def angular_velocity():
    return carla.Vector3D(np.deg2rad(0.0054123), np.deg2rad(-1.041531), np.deg2rad(4.29512))


@pytest.fixture
def angular_velocity_expected():
    return np.array((np.deg2rad(-0.0054123), np.deg2rad(-1.041531), np.deg2rad(-4.29512)))


@pytest.fixture
def compass():
    return np.deg2rad(45)


@pytest.fixture
def compass_expected():
    return np.deg2rad(45)


def test_imu_to_dict(frame, timestamp, transform,
                     acceleration, angular_velocity, compass,
                     acceleration_expected, angular_velocity_expected, compass_expected):
    imu_measurement = IMUMeasurementMock(acceleration, angular_velocity, compass, frame, timestamp, transform)
    res = imu_to_dict(imu_measurement)

    assert sorted(res.keys()) == sorted(['frame_id', 'timestamp', 'location', 'rotation', 'acceleration', 'angular_velocity', 'compass'])
    assert res['frame_id'] == frame
    assert res['timestamp'] == timestamp
    assert is_close(res['location'], np.zeros(3))
    assert is_close(res['rotation'], Quaternion())
    assert is_close(res['acceleration'], acceleration_expected)
    assert is_close(res['angular_velocity'], angular_velocity_expected)
    assert is_close(res['compass'], compass_expected)


##### OBJECT TRANSFORMATIONS #####


@pytest.fixture
def actor_properties():
    return {
        # expected_actor_id: (type_id, role_name, class_id)
        1: ('spectator', '', ''),
        2: ('traffic.traffic_light', '', ''),
        3: ('traffic.traffic_light', '', ''),
        4: ('traffic.traffic_light', '', ''),
        5: ('vehicle.tesla.model3', '', 'car'),
        6: ('vehicle.citroen.c3', 'hero', 'car'),
        7: ('static.prop.barrel', '', ''),
        8: ('walker.pedestrian.0001', '', 'person'),
        9: ('vehicle.mitsubishi.fusorosa', '', 'bus'),
        10: ('vehicle.carlamotors.european_hgv', '', 'truck'),
        11: ('vehicle.kawasaki.ninja', '', 'motorcycle'),
        12: ('vehicle.gazelle.omafiets', '', 'bicycle'),
    }


@pytest.fixture
def world(frame, timestamp, actor_properties):
    world = WorldMock(frame, timestamp)
    for type_id, role_name, _ in actor_properties.values():
        world.spawn_actor(ActorBlueprintMock(type_id, {'role_name': role_name} if role_name != '' else None),
                          carla.Transform())
    return world


def test_world_snapshot_to_dict(world, actor_properties):
    world.tick()
    world_snapshot = world.get_snapshot()
    frame, timestamp = world_snapshot.frame, world_snapshot.timestamp.elapsed_seconds
    res = world_snapshot_to_dict(world_snapshot, world)

    assert sorted(res.keys()) == sorted(['frame_id', 'timestamp', 'actors'])
    assert res['frame_id'] == frame
    assert res['timestamp'] == timestamp

    expected_actor_ids = sorted([actor_id for actor_id, props in actor_properties.items() if props[-1] != ''])
    assert sorted(res['actors'].keys()) == expected_actor_ids
    for actor_id in expected_actor_ids:
        type_id, role_name, class_id = actor_properties[actor_id]
        assert res['actors'][actor_id]['type_id'] == type_id
        assert res['actors'][actor_id]['role_name'] ==  role_name
        assert res['actors'][actor_id]['class_id'] == class_id
        # TODO Also test: center, size, rot, velocity, acceleration, angular_velocity?


def test_world_snapshot_to_dict_unknown_actor(world, actor_properties):
    world.tick()
    world_snapshot = world.get_snapshot()
    world.get_actor(list(actor_properties.keys())[-1]).destroy()

    with pytest.raises(ValueError, match='Failed to find actor'):
        world_snapshot_to_dict(world_snapshot, world)
