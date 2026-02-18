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

from cdr.utils.bounding_boxes import bounding_box_contains_points, get_bounding_box, get_global_3d_bounding_box
from tests.helpers import is_close
from tests.mocks.carla import ActorMock, ActorSnapshotMock, WorldMock


@pytest.fixture
def bounding_boxes():
    return [
        carla.BoundingBox(carla.Location(0.5,   0., 0.4), carla.Vector3D(1.5 , 1.  , 1.1)),
        carla.BoundingBox(carla.Location(0.3, -0.1, 0.3), carla.Vector3D(2.  , 1.2 , 1. )),
        carla.BoundingBox(carla.Location(0.5, -0.5, 0.2), carla.Vector3D(0.17, 0.25, 0.9)),
        carla.BoundingBox(carla.Location(), carla.Vector3D(2., 3., 1.)),
    ]


@pytest.fixture
def world():
    return WorldMock()


@pytest.fixture
def actors(world, bounding_boxes):
    return [
        ActorMock(world, 1, 'vehicle.citroen.c3', bounding_box=bounding_boxes[0]),
        ActorMock(world, 2, 'vehicle.tesla.model3', bounding_box=bounding_boxes[1]),
        ActorMock(world, 3, 'walker.pedestrian.0001', bounding_box=bounding_boxes[2]),
        ActorMock(world, 4, 'traffic.traffic_light', bounding_box=bounding_boxes[3]),
    ]


@pytest.fixture
def actor_snapshots():
    return [
        ActorSnapshotMock(1, carla.Transform(carla.Location(  10.,  10., 0.))),
        ActorSnapshotMock(2, carla.Transform(carla.Location(-100., -50., 0.))),
        ActorSnapshotMock(3, carla.Transform(carla.Location(  20., -10., 0.), carla.Rotation(yaw=90.))),
        ActorSnapshotMock(4, carla.Transform(carla.Location( 200., 200., 0.))),
    ]


@pytest.fixture
def expected_global_bbs():
    return [
        (carla.Vector3D( 10.5,  10. , 0.4), carla.Rotation(),        np.array((3.  , 2. , 2.2))),
        (carla.Vector3D(-99.7, -50.1, 0.3), carla.Rotation(),        np.array((4.  , 2.4, 2. ))),
        (carla.Vector3D( 20.5,  -9.5, 0.2), carla.Rotation(yaw=90.), np.array((0.34, 0.5, 1.8))),
        (carla.Vector3D(200. , 200. , 0. ), carla.Rotation(),        np.array((4.  , 6. , 2. )))
    ]


def test_get_bounding_box_part1(actors, bounding_boxes):
    assert all([get_bounding_box(actor) == bounding_box for actor, bounding_box in zip(actors, bounding_boxes)])


def test_get_bounding_box_part2(world):
    bounding_box = carla.BoundingBox(carla.Location(), carla.Vector3D(2.5, 2., 1.6))
    actor = ActorMock(world, 1, 'vehicle.tesla.cybertruck', bounding_box=bounding_box)
    assert get_bounding_box(actor) == bounding_box


def test_get_global_3d_bounding_box(actors, actor_snapshots, expected_global_bbs):
    for actor, actor_snapshot, expected_bb in zip(actors, actor_snapshots, expected_global_bbs):
        ret = get_global_3d_bounding_box(actor, actor_snapshot)
        expected_loc, expected_rot, expected_size = expected_bb
        actual_loc, actual_rot, actual_size = ret
        assert is_close(expected_loc, actual_loc)
        assert is_close(expected_rot, actual_rot)
        assert is_close(expected_size, actual_size)


def test_bounding_box_contains_points_local():
    bb_center = np.array((0., 0., 0.))
    bb_size = np.array((4., 2., 2.))
    points = np.array([
        (  0. , 0. ,  0. ),  # True
        (  2. , 1. ,  1. ),  # True
        (  2. , 1. ,  1.1),  # False
        ( -1. , 0. , -1. ),  # True
        (-10. , 0. ,  0. ),  # False
    ])
    expected_result = np.array([True, True, False, True, False])
    assert np.all(bounding_box_contains_points(bb_center, bb_size, Quaternion(), points) == expected_result)
    assert np.all(bounding_box_contains_points(bb_center, bb_size, Quaternion(axis=np.array((1., 0., 0.)), degrees=180), points) == expected_result)
    assert np.all(bounding_box_contains_points(bb_center, bb_size, Quaternion(axis=np.array((0., 1., 0.)), degrees=180), points) == expected_result)
    assert np.all(bounding_box_contains_points(bb_center, bb_size, Quaternion(axis=np.array((0., 0., 1.)), degrees=180), points) == expected_result)


def test_bounding_box_contains_points_global():
    bb_center = np.array((100., -50., 5.))
    bb_size = np.array((4., 2., 2.))
    points = np.array([
        (  0. ,  0.,  0. ),  # False
        (100., -50.,  5. ),  # True
        (102., -51.,  6. ),  # True
        (100., -50., -5. ),  # False
    ])
    expected_result = np.array([False, True, True, False])
    assert np.all(bounding_box_contains_points(bb_center, bb_size, Quaternion(), points) == expected_result)
    assert np.all(bounding_box_contains_points(bb_center, bb_size, Quaternion(axis=np.array((1., 0., 0.)), degrees=180), points) == expected_result)
    assert np.all(bounding_box_contains_points(bb_center, bb_size, Quaternion(axis=np.array((0., 1., 0.)), degrees=180), points) == expected_result)
    assert np.all(bounding_box_contains_points(bb_center, bb_size, Quaternion(axis=np.array((0., 0., 1.)), degrees=180), points) == expected_result)
