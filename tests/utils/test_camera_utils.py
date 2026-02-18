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


import numpy as np
import pytest
from pyquaternion import Quaternion

from cdr.utils.camera_utils import apply_transformation, camera_parameters_to_intrinsic_matrix, get_all_pixel_coordinates, pose_to_transformation_matrix, project_image_to_world_coordinates, project_world_to_image_coordinates, transform_depth_to_z_depth, transform_z_depth_to_depth, transformation_matrix_to_pose
from tests.helpers import is_close



@pytest.fixture
def cameras_parameters():
    return [
        (400, 300,  90.),
        (600, 600, 135.)
    ]


@pytest.fixture
def intrinsic_matrices():
    return [
        np.array([[200.,   0., 200.],
                  [  0., 200., 150.],
                  [  0.,   0.,   1.]]),
        np.array([[124.264,   0.   , 300.],
                  [  0.   , 124.264, 300.],
                  [  0.   ,   0.   ,   1.]]),
    ]


def test_camera_parameters_to_intrinsic_matrix(cameras_parameters, intrinsic_matrices):
    for camera_parameters, intrinsic_matrix in zip(cameras_parameters, intrinsic_matrices):
        assert np.all(np.isclose(camera_parameters_to_intrinsic_matrix(*camera_parameters), intrinsic_matrix))


@pytest.fixture
def poses():
    return [
        (np.array([ 0.,   0., 0.]), Quaternion()),
        (np.array([10., -10., 1.]), Quaternion()),
        (np.array([ 1.,   2., 3.]), Quaternion(axis=[1., 0., 0], degrees=90.)),
        (np.array([ 1.,   2., 3.]), Quaternion(axis=[0., 1., 0], degrees=-90.)),
        (np.array([ 1.,   2., 3.]), Quaternion(axis=[0., 0., 1], degrees=180.)),
    ]


@pytest.fixture
def transformation_matrices():
    return [
        np.eye(4),
        np.array([ [ 1.,  0.,  0.,  10.],
                   [ 0.,  1.,  0., -10.],
                   [ 0.,  0.,  1.,   1.],
                   [ 0.,  0.,  0.,   1.]]),
        np.array([ [ 1.,  0.,  0.,   1.],
                   [ 0.,  0., -1.,   2.],
                   [ 0.,  1.,  0.,   3.],
                   [ 0.,  0.,  0.,   1.]]),
        np.array([ [ 0.,  0., -1.,   1.],
                   [ 0.,  1.,  0.,   2.],
                   [ 1.,  0.,  0.,   3.],
                   [ 0.,  0.,  0.,   1.]]),
        np.array([ [-1.,  0.,  0.,   1.],
                   [ 0., -1.,  0.,   2.],
                   [ 0.,  0.,  1.,   3.],
                   [ 0.,  0.,  0.,   1.]]),
    ]


def test_pose_to_transformation_matrix(poses, transformation_matrices):
    for pose, transformation_matrix in zip(poses, transformation_matrices):
        assert np.all(np.isclose(pose_to_transformation_matrix(*pose), transformation_matrix))


def test_transformation_matrix_to_pose(transformation_matrices, poses):
    for transformation_matrix, pose in zip(transformation_matrices, poses):
        ret = transformation_matrix_to_pose(transformation_matrix)
        assert is_close(ret[0], pose[0])
        assert is_close(ret[1], pose[1])


def test_pose_to_transformation_matrix_to_pose(poses):
    for pose in poses:
        ret = transformation_matrix_to_pose(pose_to_transformation_matrix(*pose))
        assert is_close(ret[0], pose[0])
        assert is_close(ret[1], pose[1])


def test_transformation_matrix_to_pose_to_transformation_matrix(transformation_matrices):
    for transformation_matrix in transformation_matrices:
        ret = pose_to_transformation_matrix(*transformation_matrix_to_pose(transformation_matrix))
        assert is_close(ret, transformation_matrix)


@pytest.fixture
def transformation_matrix(transformation_matrices):
    return transformation_matrices[-1]


@pytest.fixture
def points():
    return np.array([
        [ 0.,  0.,  0.],
        [ 1.,  1.,  1.],
        [-1., -1., -1.],
        [-1., -2., -3.],
        [ 1.,  2., -3.],
    ])


@pytest.fixture
def points_transformed():
    return np.array([
        [ 1.,  2.,  3.],
        [ 0.,  1.,  4.],
        [ 2.,  3.,  2.],
        [ 2.,  4.,  0.],
        [ 0.,  0.,  0.],
    ])


@pytest.mark.parametrize('sample_axis', [None, 0, 1, 2])
def test_apply_transformation_Nx3(transformation_matrix, points, points_transformed, sample_axis):
    if sample_axis in [1, 2]:
        with pytest.raises(ValueError):
            apply_transformation(points, transformation_matrix, sample_axis)
    else:
        assert np.all(np.isclose(apply_transformation(points, transformation_matrix, sample_axis), points_transformed))


@pytest.mark.parametrize('sample_axis', [None, 0, 1, 2])
def test_apply_transformation_3xN(transformation_matrix, points, points_transformed, sample_axis):
    if sample_axis in [0, 2]:
        with pytest.raises(ValueError):
            apply_transformation(points.T, transformation_matrix, sample_axis)
    else:
        assert np.all(np.isclose(apply_transformation(points.T, transformation_matrix, sample_axis), points_transformed.T))


def test_apply_transformation_infer_sample_axis_3x3(transformation_matrix, points):
    with pytest.raises(ValueError):
        apply_transformation(points[:3], transformation_matrix)


@pytest.fixture
def camera_transformation_matrix():
    return np.array(
        [[ 1.,  0.,  0.,   1.],
         [ 0.,  0.,  1.,  -2.],
         [ 0., -1.,  0.,  -1.],
         [ 0.,  0.,  0.,   1.]]
    )


@pytest.fixture
def intrinsic_matrix(intrinsic_matrices):
    return intrinsic_matrices[0]


@pytest.fixture
def image_coords():
    return np.array([
        [ 100.,   50.      ],
        [ 200.,   16. + 2/3],
        [-200.,  150.      ],
        [   0.,    0.      ],
        [ 200.,  250.      ]
    ])


@pytest.fixture
def z_depths():
    return np.array([2., 3., 1., 0., 4.])


@pytest.fixture
def z_depths_2d():
    return np.array([
        [5. , 4.8, 5. , 5.1],
        [4. , 4. , 3.7, 3.8],
        [3.1, 2.1, 1.9, 2. ]
    ])


@pytest.fixture
def depths():
    return np.array([2.44948974, 3.60555128, 2.23606798, 0.        , 4.47213595])


@pytest.fixture
def depths_2d():
    return np.array([
        [7.99024796, 7.65568024, 7.9591339 , 8.10252218],
        [6.38286769, 6.3703846 , 5.88109442, 6.02825717],
        [4.93952924, 3.33956953, 3.0155954 , 3.1680988 ]
    ])


@pytest.mark.parametrize('sample_axis', [None, 0, 1, 2])
def test_project_world_to_image_coordinates(points, camera_transformation_matrix, intrinsic_matrix, image_coords,
                                            sample_axis, z_depths):
    if sample_axis in [1, 2]:
        with pytest.raises(ValueError):
            project_world_to_image_coordinates(points, intrinsic_matrix, camera_transformation_matrix, c2w=True,
                                               sample_axis=sample_axis)
    else:
        ret = project_world_to_image_coordinates(points, intrinsic_matrix, camera_transformation_matrix, c2w=True,
                                                 sample_axis=sample_axis)
        assert np.all(np.isclose(ret[0], image_coords))
        assert np.all(np.isclose(ret[1], z_depths))


@pytest.mark.parametrize('sample_axis', [None, 0, 1, 2])
def test_project_image_to_world_coordinates(image_coords, z_depths, camera_transformation_matrix, intrinsic_matrix,
                                            points, sample_axis):
    if sample_axis in [1, 2]:
        with pytest.raises(ValueError):
            project_image_to_world_coordinates(image_coords, intrinsic_matrix, camera_transformation_matrix, c2w=True,
                                               z_depths=z_depths, sample_axis=sample_axis)
    else:
        with pytest.warns(RuntimeWarning, match='divide by zero encountered in divide'):
            with pytest.warns(RuntimeWarning, match='invalid value encountered in multiply'):
                ret = project_image_to_world_coordinates(image_coords, intrinsic_matrix, camera_transformation_matrix,
                                                         c2w=True, z_depths=z_depths, sample_axis=sample_axis)
        mask = z_depths != 0.
        assert np.all(np.isclose(ret[mask], points[mask]))
        assert np.all(np.isnan(ret[~mask]))


def test_project_image_to_world_coordinates_zdepth_misshaped(image_coords, camera_transformation_matrix, intrinsic_matrix):
    with pytest.raises(ValueError):
        project_image_to_world_coordinates(image_coords, intrinsic_matrix, camera_transformation_matrix, c2w=True,
                                            z_depths=np.ones((image_coords.shape[0], 2)), sample_axis=0)


@pytest.fixture
def expected_pixel_coords():
    return np.array([
        [0., 0.], [1., 0.], [2., 0.], [3., 0.], [4., 0.], [5., 0.],
        [0., 1.], [1., 1.], [2., 1.], [3., 1.], [4., 1.], [5., 1.],
        [0., 2.], [1., 2.], [2., 2.], [3., 2.], [4., 2.], [5., 2.],
        [0., 3.], [1., 3.], [2., 3.], [3., 3.], [4., 3.], [5., 3.],
    ])


@pytest.mark.parametrize('midpoints', [True, False])
def test_get_all_pixel_coordinates(expected_pixel_coords, midpoints):
    pixel_coordinates = get_all_pixel_coordinates(4, 6, midpoints)
    if midpoints:
        expected_pixel_coords += 0.5
    assert np.all(np.isclose(pixel_coordinates, expected_pixel_coords))


def test_transform_depth_to_z_depth_1d(depths, intrinsic_matrix, image_coords, z_depths):
    ret = transform_depth_to_z_depth(depths, intrinsic_matrix, image_coords)
    assert np.all(np.isclose(ret, z_depths))


def test_transform_depth_to_z_depth_1d_missing_image_coords(depths, intrinsic_matrix):
    with pytest.raises(ValueError):
        transform_depth_to_z_depth(depths, intrinsic_matrix)


def test_transform_depth_to_z_depth_1d_invalid_number_image_coords(depths, intrinsic_matrix, image_coords):
    with pytest.raises(ValueError):
        transform_depth_to_z_depth(depths, intrinsic_matrix, np.concatenate((image_coords, image_coords), axis=0))


def test_transform_depth_to_z_depth_2d(depths_2d, intrinsic_matrix, z_depths_2d):
    ret = transform_depth_to_z_depth(depths_2d, intrinsic_matrix)
    assert np.all(np.isclose(ret, z_depths_2d))


def test_transform_depth_to_z_depth_3d(depths_2d, intrinsic_matrix, z_depths_2d):
    depths_3d = depths_2d.reshape(depths_2d.shape + (1,))
    z_depths_3d = z_depths_2d.reshape(z_depths_2d.shape + (1,))
    ret = transform_depth_to_z_depth(depths_3d, intrinsic_matrix)
    assert np.all(np.isclose(ret, z_depths_3d))


def test_transform_depth_to_z_depth_3d_invalid(depths_2d, intrinsic_matrix):
    depths_3d = np.broadcast_to(depths_2d[Ellipsis, None], depths_2d.shape + (3,))
    with pytest.raises(ValueError):
        transform_depth_to_z_depth(depths_3d, intrinsic_matrix)


def test_transform_z_depth_to_depth_1d(z_depths, intrinsic_matrix, image_coords, depths):
    ret = transform_z_depth_to_depth(z_depths, intrinsic_matrix, image_coords)
    assert np.all(np.isclose(ret, depths))


def test_transform_z_depth_to_depth_1d_missing_image_coords(z_depths, intrinsic_matrix):
    with pytest.raises(ValueError):
        transform_z_depth_to_depth(z_depths, intrinsic_matrix)


def test_transform_z_depth_to_depth_2d(z_depths_2d, intrinsic_matrix, depths_2d):
    ret = transform_z_depth_to_depth(z_depths_2d, intrinsic_matrix)
    assert np.all(np.isclose(ret, depths_2d))


def test_transform_z_depth_to_depth_3d(z_depths_2d, intrinsic_matrix, depths_2d):
    z_depths_3d = z_depths_2d.reshape(z_depths_2d.shape + (1,))
    depths_3d = depths_2d.reshape(depths_2d.shape + (1,))
    ret = transform_z_depth_to_depth(z_depths_3d, intrinsic_matrix)
    assert np.all(np.isclose(ret, depths_3d))


def test_transform_z_depth_to_depth_3d_invalid(z_depths_2d, intrinsic_matrix):
    z_depths_3d = np.broadcast_to(z_depths_2d[Ellipsis, None], z_depths_2d.shape + (3,))
    with pytest.raises(ValueError):
        transform_depth_to_z_depth(z_depths_3d, intrinsic_matrix)


def test_transform_depth_to_z_depth_to_depth(depths, intrinsic_matrix, image_coords):
    ret = transform_z_depth_to_depth(transform_depth_to_z_depth(depths, intrinsic_matrix, image_coords),
                                     intrinsic_matrix, image_coords)
    assert np.all(np.isclose(ret, depths))


def test_transform_z_depth_to_depth_to_z_depth(z_depths, intrinsic_matrix, image_coords):
    ret = transform_depth_to_z_depth(transform_z_depth_to_depth(z_depths, intrinsic_matrix, image_coords),
                                     intrinsic_matrix, image_coords)
    assert np.all(np.isclose(ret, z_depths))
