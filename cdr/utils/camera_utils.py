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


"""
Collection of utility methods to handle projections from 3D world space to 2D image space and vice versa.
"""

from enum import Enum
from typing import Literal, Optional, Tuple, Union

import numpy as np
from pyquaternion import Quaternion


def camera_parameters_to_intrinsic_matrix(width: float, height: float, fov: float) -> np.ndarray:
    """
    Returns the intrinsic matrix for the given camera parameters. The given parameters should all have the same unit
    (e.g. pixel or mm).

    Args:
        width (float): the width of the image
        height (float): the height of the image
        fov (float): the horizontal FOV of the image

    Returns:
        intrinsic_matrix (np.ndarray): the corresponding intrinsic matrix `[3, 3]`
    """
    intrinsic_matrix = np.eye(3)
    focal_length = width / 2. / np.tan(np.deg2rad(fov / 2))
    intrinsic_matrix[0, 0] = intrinsic_matrix[1, 1] = focal_length
    intrinsic_matrix[0, 2] = width / 2
    intrinsic_matrix[1, 2] = height / 2
    return intrinsic_matrix


def pose_to_transformation_matrix(translation: np.ndarray, rotation: Quaternion) -> np.ndarray:
    """
    Translates an object's pose representation to the corresponding transformation matrix `[4, 4]`. If the pose is
    specified in a world-to-object frame (how the world has to be transformed relative to the object), the returned
    transformation matrix will also be in a world-to-object frame. Analogously, a object-to-world transformation matrix
    will be returned, if the pose is given in a object-to-world frame.

    Args:
        translation (np.ndarray): the translation vector of the pose
        rotation (Quaternion): the rotation (as Quaternion) of the pose

    Returns:
        transformation_matrix (np.ndarray): the corresponding transformation matrix `[4, 4]`
    """
    transformation_matrix = np.eye(4, dtype=np.float32)
    transformation_matrix[:3, :3] = rotation.rotation_matrix
    transformation_matrix[:3, 3] = translation
    return transformation_matrix


def transformation_matrix_to_pose(transformation_matrix: np.ndarray) -> Tuple[np.ndarray, Quaternion]:
    """
    Translates a transformation matrix (either `[3, 4]` or `[4, 4]`) to the corresponding pose representation. If the
    transformation matrix is specified in a world-to-object frame (how the world has to be transformed relative to
    the object), the returned pose will also be in a world-to-object frame. Analogously, a object-to-world pose will
    be returned, if the transformation matrix is given in a object-to-world frame.

    Args:
        transformation_matrix (np.ndarray): the transformation_matrix to translate (either `[3, 4]` or `[4, 4]`)

    Returns:
        translation,rotation (Tuple[np.ndarray, Quaternion]):
            a tuple of the translation vector and the rotation (as Quaternion) of the pose
    """
    rotation = Quaternion(matrix=transformation_matrix[:3, :3])
    translation = transformation_matrix[:3, 3]
    return translation, rotation


class PointsLayout(Enum):

    N_x_DIM = 'Nx[2|3]'
    DIM_x_N = '[2|3]xN'


def _infer_points_layout(points: np.ndarray, sample_axis: Union[None, Literal[0], Literal[1]],
                         point_dim: Union[Literal[2], Literal[3]]) -> PointsLayout:
    if sample_axis is None:
        # If sample axis is not provided, we try to infer it
        if points.shape[0] == point_dim and points.shape[1] != point_dim:
            return PointsLayout.DIM_x_N
        elif points.shape[1] == point_dim and points.shape[0] != point_dim:
            return PointsLayout.N_x_DIM
        else:
            raise ValueError(f'Passed matrix of points must be of shape [{point_dim}], n] or [n, {point_dim}], '
                             f'but was of shape {points.shape} which is ambiguous without providing a `sample_axis`.')
    else:
        if sample_axis == 0:
            return PointsLayout.N_x_DIM
        elif sample_axis == 1:
            return PointsLayout.DIM_x_N
        else:
            raise ValueError(f'Provided sample axis must be 0 or 1, but was {sample_axis}.')


def apply_transformation(points: np.ndarray, transformation_matrix: np.ndarray,
                         sample_axis: Union[None, Literal[0], Literal[1]] = None) -> np.ndarray:
    """
    Applies the given transformation to all points.

    Args:
        points (np.ndarray): points to transform
        transformation_matrix (np.ndarray): transformation to apply
        sample_axis (Union[None, Literal[0], Literal[1]], optional): Indicates the axis along which the samples are
            arranged, i.e. `0` indicates that `points` is of shape `[n, 3]`, while `1` indicates a shape of `[3, n]`.
            If `None`, it is tried to infer the `sample_axis`, which fails, if there are exactly 3 samples, i.e. the
            shape is `[3, 3]`. Defaults to None.

    Raises:
        ValueError: if `sample_axis=None` and the points have the ambiguous shape `[3, 3]`

    Returns:
        points (np.ndarray): transformed points
    """
    # We support coordinates in both layouts [3, n] or [n, 3]
    points_layout = _infer_points_layout(points, sample_axis, 3)
    if points_layout == PointsLayout.N_x_DIM:
        points = points.T

    points_homogeneous = transform_cartesian_to_homogeneous_coordinates(points)
    points_transformed = np.dot(transformation_matrix, points_homogeneous)
    points_transformed = transform_homogeneous_to_cartesian_coordinates(points_transformed)

    if points_layout == PointsLayout.N_x_DIM:
        points_transformed = points_transformed.T
    return points_transformed


def project_world_to_image_coordinates(world_coordinates: np.ndarray,
                                       intrinsic_matrix: np.ndarray, transformation_matrix: np.ndarray,
                                       c2w: bool = True,
                                       sample_axis: Union[None, Literal[0], Literal[1]] = None) -> Tuple[np.ndarray, np.ndarray]:
    """
    Projects the given world coordinates to image coordinates of the camera, described by it's transformation and
    intrinsic matrices.

    Args:
        world_coordinates (np.ndarray): the world coordinates in the shape `[3, n]` or `[n, 3]` that shall be projected
            to image coordinates
        intrinsic_matrix (np.ndarray): the intrinsic matrix of the camera
        transformation_matrix (np.ndarray): the transformation matrix of the camera
        c2w (bool, optional): a boolean that indicates if the given transformation matrix is specified as
            camera-to-world transformation (True), or as world-to-camera transformation (False).
        sample_axis (Union[None, Literal[0], Literal[1]], optional): Indicates the axis along which the samples are
            arranged, i.e. `0` indicates that `world_coordinates` is of shape `[n, 3]`, while `1` indicates a shape of
            `[3, n]`. If `None`, it is tried to infer the `sample_axis`, which fails, if there are exactly 3 samples,
            i.e. the shape is `[3, 3]`. Defaults to None.

    Raises:
        ValueError: if `sample_axis=None` and the image coordinates have the ambiguous shape `[3, 3]`

    Returns:
        image_coordinates_cartesian,z_depths (Tuple[np.ndarray, np.ndarray]):
            a tuple of the image coordinates and the z-depth to the point for each image coordinate
    """
    # We support coordinates in both layouts [3, n] or [n, 3]
    points_layout = _infer_points_layout(world_coordinates, sample_axis, 3)
    if points_layout == PointsLayout.N_x_DIM:
        world_coordinates = world_coordinates.T

    assert 3 <= transformation_matrix.shape[0] <= 4
    assert transformation_matrix.shape[1] == 4
    extrinsic_matrix = transformation_matrix
    if c2w:
        extrinsic_matrix = np.linalg.pinv(extrinsic_matrix)
    extrinsic_matrix = extrinsic_matrix[:3, :]

    # Transform the world coordinates from Euclidean space to homogeneous space.
    world_coordinates_homogeneous = transform_cartesian_to_homogeneous_coordinates(world_coordinates)
    # Apply the projection from world coordinates to image coordinates.
    camera_matrix = np.dot(intrinsic_matrix, extrinsic_matrix)
    image_coordinates_homogeneous = np.dot(camera_matrix, world_coordinates_homogeneous)
    # Keep a copy of the z-depth values.
    z_depths = image_coordinates_homogeneous[-1, :].copy()
    # Transform the image coordinates from homogeneous space to Euclidean space.
    image_coordinates_cartesian = transform_homogeneous_to_cartesian_coordinates(image_coordinates_homogeneous)

    # If the world coordinates were provided as (n, 3), return the image coordinates as (n, 2).
    if points_layout == PointsLayout.N_x_DIM:
        image_coordinates_cartesian = image_coordinates_cartesian.T

    return image_coordinates_cartesian, z_depths


def project_image_to_world_coordinates(image_coordinates: np.ndarray,
                                       intrinsic_matrix: np.ndarray, transformation_matrix: np.ndarray,
                                       c2w: bool = True, z_depths: Union[float, np.ndarray] = 1.,
                                       sample_axis: Union[None, Literal[0], Literal[1]] = None) -> np.ndarray:
    """
    Projects the given image coordinates of the camera, described by it's transformation and intrinsic matrices,
    to the corresponding world coordinates.

    Args:
        image_coordinates (np.ndarray): the image coordinates in the shape `[2, n]` or `[n, 2]` that shall be projected
            to world coordinates
        intrinsic_matrix (np.ndarray): the intrinsic matrix of the camera
        transformation_matrix (np.ndarray): the transformation matrix of the camera
        c2w (bool, optional): a boolean that indicates if the given transformation matrix is specified as
            camera-to-world transformation (True), or as world-to-camera transformation (False).
        z_depths (Union[float, np.ndarray], optional): Either a scalar or an array with as many entries as
            `image_coordinates`-entries. The z_depth values will be used to project the image coordinates to world
            coordinates. Defaults to 1..
        sample_axis (Union[None, Literal[0], Literal[1]], optional): Indicates the axis along which the samples are
            arranged, i.e. `0` indicates that `image_coordinates` is of shape `[n, 2]`, while `1` indicates a shape of
            `[2, n]`. If `None`, it is tried to infer the `sample_axis`, which fails, if there are exactly 2 samples,
            i.e. the shape is `[2, 2]`. Defaults to None.

    Raises:
        ValueError: if `sample_axis=None` and the image coordinates have the ambiguous shape `[2, 2]`
        ValueError: if `z_depths` is neither a scalar nor a 1D-array, or if the 1D-array does not have as many entries
                    as image coordinates

    Returns:
        world_coordinates (np.ndarray): the world coordinates of the given image coordinates
    """
    # We support coordinates in both layouts [2, n] or [n, 2]
    points_layout = _infer_points_layout(image_coordinates, sample_axis, 2)
    if points_layout == PointsLayout.N_x_DIM:
        image_coordinates = image_coordinates.T

    # Check if z-depth is either a scalar (apply same z-depth for each point)
    # or a 1D-array with as many values as image coordinates (apply individual z-depth for each point)
    if np.ndim(z_depths) == 0:
        pass
    elif isinstance(z_depths, np.ndarray) and np.ndim(z_depths) == 1:
        if image_coordinates.shape[1] != z_depths.shape[0]:
            raise ValueError(f'Shape mismatch: Provided {image_coordinates.shape[1]} image coordinates but '
                             f'{z_depths.shape[0]} z-depth values!')
    else:
        raise ValueError('"z_depth" must be either a scalar or a 1D-array with as many values as image coordinates!')

    assert 3 <= transformation_matrix.shape[0] <= 4
    assert transformation_matrix.shape[1] == 4
    extrinsic_matrix_4x4 = np.eye(4)
    extrinsic_matrix_4x4[:3, :4] = transformation_matrix[:3, :4]
    if c2w:
        extrinsic_matrix_4x4 = np.linalg.inv(extrinsic_matrix_4x4)

    intrinsic_matrix_4x4 = np.eye(4)
    intrinsic_matrix_4x4[:3, :3] = intrinsic_matrix

    # Transform image coordinates from Euclidean to homogeneous space
    image_coordinates_homogeneous = transform_cartesian_to_homogeneous_coordinates(image_coordinates)
    # Concatenate disparities to the homogeneous coordinates
    disparities = np.ones(image_coordinates_homogeneous.shape[1], dtype=np.float32) / z_depths
    image_coordinates_homogeneous = np.vstack((image_coordinates_homogeneous, disparities))
    # Build camera matrix and invert it for projection from image to world coordinates
    camera_matrix = np.dot(intrinsic_matrix_4x4, extrinsic_matrix_4x4)
    camera_matrix_inv = np.linalg.inv(camera_matrix)
    # Reproject the camera coordinates to world coordinates
    world_coordinates_homogeneous = z_depths * np.dot(camera_matrix_inv, image_coordinates_homogeneous)
    # Transform homogeneous space to Euclidean space
    world_coordinates_cartesian = transform_homogeneous_to_cartesian_coordinates(world_coordinates_homogeneous)

    # If the image coordinates were provided as (n, 2), return the world coordinates as (n, 3).
    if points_layout == PointsLayout.N_x_DIM:
        world_coordinates_cartesian = world_coordinates_cartesian.T

    return world_coordinates_cartesian


def get_all_pixel_coordinates(image_height: int, image_width: int, midpoints: bool = True) -> np.ndarray:
    """
    Creates an array of shape `[image_height * image_width, 2]` containing image coordinates for all pixels.

    Args:
        image_height (int): the height of the image
        image_width (int): the width of the image
        midpoints (bool, optional): Whether the pixel coordinates shall represent the upper-left corner (False)
            or the midpoint (True) of the pixel. Defaults to True.

    Returns:
        image_coordinates (np.ndarray): array of shape `[image_height * image_width, 2]` containing the image
            coordinates of all pixels
    """
    offset = 0.
    if midpoints:
        offset = 0.5
    x_range = np.arange(offset, image_width + offset)
    y_range = np.arange(offset, image_height + offset)
    xx, yy = np.meshgrid(x_range, y_range, indexing='xy')
    pixels_in_2d = np.stack([xx.flatten(), yy.flatten()], axis=1)
    return pixels_in_2d


def _depths_shape_alignment(depths: np.ndarray, image_coordinates: Optional[np.ndarray] = None,
                            sample_axis: Union[None, Literal[0], Literal[1]] = None) -> Tuple[np.ndarray, np.ndarray, Tuple[int, ...]]:
    """
    Aligns the shapes of depth values and their corresponding image coordinates so that they can be used in vectorized
    operations.

    This helper accepts depth data either in shape `[n,]` or `[height, width]` or `[height, width, 1]`. It converts
    the data into a depth vector and an array of 2D image coordinates that are both of shape `[n]` and `[2, n]`
    respectively. The original shape of the input depth array is returned as the third element of the tuple. If
    `depths` is in an image-shape, the `image_coordinates` can be automatically determined, otherwise, they must be
    given by the caller.

    Args:
        depths (np.ndarray): the depth values, either in shape `[n,]` or `[height, width]` or `[height, width, 1]`.
            If shape `[n,]` is given, `image_coordinates` must be given too.
        image_coordinates (Optional[np.ndarray], optional): the image coordinates to use. Must be set if `depths`
            is of shape `[n,]`. Defaults to None.
        sample_axis (Union[None, Literal[0], Literal[1]], optional): Indicates the axis along which the samples are
            arranged, i.e. `0` indicates that `image_coordinates` is of shape `[n, 2]`, while `1` indicates a shape of
            `[2, n]`. If `None`, it is tried to infer the `sample_axis`, which fails, if there are exactly 2 samples,
            i.e. the shape is `[2, 2]`. Only used if `image_coordinates` is given. Defaults to None.

    Raises:
        ValueError: if `depths` is of shape `[n,]` and `image_coordinates` is None
        ValueError: if `depths` is neither in shape `[n,]` nor `[height, width]` nor `[height, width, 1]`
        ValueError: if `image_coordinates` are given but their number of samples does not align with `depths`

    Returns:
        depths,image_coordinates,original_shape (Tuple[np.ndarray, np.ndarray, Tuple[int, ...]]):
            depths and image_coordinates in shape `[n,]` and `[2, n]`, respectively. The original shape of `depths` is
            the third component of the returned tuple.
    """
    # We support depths either as [n,] or [height, width] or [height, width, 1]
    original_shape = depths.shape
    if depths.ndim == 1:
        if image_coordinates is None:
            raise ValueError('When providing flattened `depths` values, `image_coordinates` must be set explicitly!')
    elif depths.ndim == 2 or (depths.ndim == 3 and depths.shape[-1] == 1):
        if image_coordinates is None:
            image_height, image_width = depths.shape[:2]
            image_coordinates = get_all_pixel_coordinates(image_height, image_width)
            sample_axis = 0
        depths = depths.ravel()
    else:
        raise ValueError(f'`depths` of shape {depths.shape} are not supported.')

    # We support coordinates in both layouts [2, n] or [n, 2]
    points_layout = _infer_points_layout(image_coordinates, sample_axis, 2)
    if points_layout == PointsLayout.N_x_DIM:
        image_coordinates = image_coordinates.T

    # Last consistency check, if both `depths` and `image_coordinates` were given
    if depths.shape[0] != image_coordinates.shape[1]:
        raise ValueError('Provided number of depth values does not match number of image coordinates: '
                         f'{depths.shape[0]} vs {image_coordinates.shape[1]}')

    return depths, image_coordinates, original_shape


def transform_depth_to_z_depth(depths: np.ndarray, intrinsic_matrix: np.ndarray,
                               image_coordinates: Optional[np.ndarray] = None,
                               sample_axis: Union[None, Literal[0], Literal[1]] = None) -> np.ndarray:
    """
    Transforms the given `depths` values to corresponding `z_depths` values according to the intrinsic matrix
    and image coordinates.

    This function accepts depth data either in shape `[n,]` or `[height, width]` or `[height, width, 1]`. If `depths`
    is in an image-shape, the `image_coordinates` can be automatically determined, otherwise, they must be given
    by the caller.

    Args:
        depths (np.ndarray): the depth values, either in shape `[n,]` or `[height, width]` or `[height, width, 1]`.
            If shape `[n,]` is given, `image_coordinates` must be given too.
        intrinsic_matrix (np.ndarray): the intrinsic matrix of the camera
        image_coordinates (Optional[np.ndarray], optional): the image coordinates to use. Must be set if `depths`
            is of shape `[n,]`. Defaults to None.
        sample_axis (Union[None, Literal[0], Literal[1]], optional): Indicates the axis along which the samples are
            arranged, i.e. `0` indicates that `image_coordinates` is of shape `[n, 2]`, while `1` indicates a shape of
            `[2, n]`. If `None`, it is tried to infer the `sample_axis`, which fails, if there are exactly 2 samples,
            i.e. the shape is `[2, 2]`. Only used if `image_coordinates` is given. Defaults to None.

    Raises:
        ValueError: if `depths` is of shape `[n,]` and `image_coordinates` is None
        ValueError: if `depths` is neither in shape `[n,]` nor `[height, width]` nor `[height, width, 1]`
        ValueError: if `image_coordinates` are given but their number of samples does not align with `depths`

    Returns:
        z_depths (np.ndarray): the corresponding z_depths
    """
    depths, image_coordinates, original_shape = _depths_shape_alignment(depths, image_coordinates, sample_axis)
    image_coordinates_homogeneous = transform_cartesian_to_homogeneous_coordinates(image_coordinates)
    camera_rays = np.linalg.inv(intrinsic_matrix) @ image_coordinates_homogeneous
    z_depths = depths / np.linalg.norm(camera_rays, axis=0) * camera_rays[-1, :]
    return z_depths.reshape(original_shape)


def transform_z_depth_to_depth(z_depths: np.ndarray, intrinsic_matrix: np.ndarray,
                               image_coordinates: Optional[np.ndarray] = None,
                               sample_axis: Union[None, Literal[0], Literal[1]] = None) -> np.ndarray:
    """
    Transforms the given `z_depths` values to corresponding `depths` values according to the intrinsic matrix
    and image coordinates.

    This function accepts depth data either in shape `[n,]` or `[height, width]` or `[height, width, 1]`. If `z_depths`
    is in an image-shape, the `image_coordinates` can be automatically determined, otherwise, they must be given
    by the caller.

    Args:
        z_depths (np.ndarray): the z_depth values, either in shape `[n,]` or `[height, width]` or `[height, width, 1]`.
            If shape `[n,]` is given, `image_coordinates` must be given too.
        intrinsic_matrix (np.ndarray): the intrinsic matrix of the camera
        image_coordinates (Optional[np.ndarray], optional): the image coordinates to use. Must be set if `depths`
            is of shape `[n,]`. Defaults to None.
        sample_axis (Union[None, Literal[0], Literal[1]], optional): Indicates the axis along which the samples are
            arranged, i.e. `0` indicates that `image_coordinates` is of shape `[n, 2]`, while `1` indicates a shape of
            `[2, n]`. If `None`, it is tried to infer the `sample_axis`, which fails, if there are exactly 2 samples,
            i.e. the shape is `[2, 2]`. Only used if `image_coordinates` is given. Defaults to None.

    Raises:
        ValueError: if `depths` is of shape `[n,]` and `image_coordinates` is None
        ValueError: if `depths` is neither in shape `[n,]` nor `[height, width]` nor `[height, width, 1]`
        ValueError: if `image_coordinates` are given but their number of samples does not align with `depths`

    Returns:
        depths (np.ndarray): the corresponding depths
    """
    z_depths, image_coordinates, original_shape = _depths_shape_alignment(z_depths, image_coordinates, sample_axis)
    image_coordinates_homogeneous = transform_cartesian_to_homogeneous_coordinates(image_coordinates)
    camera_rays = np.linalg.inv(intrinsic_matrix) @ image_coordinates_homogeneous
    depths = z_depths * np.linalg.norm(camera_rays, axis=0) / camera_rays[-1, :]
    return depths.reshape(original_shape)


def transform_cartesian_to_homogeneous_coordinates(cartesian_coordinates: np.ndarray) -> np.ndarray:
    """
    Transforms the given cartesian coordinates to homogeneous coordinates.

    Args:
        cartesian_coordinates (np.ndarray): the cartesian coordinates that shall be transformed in shape `[3, n]`

    Returns:
        np.ndarray: array containing the homogeneous coordinates in shape `[4, n]`
    """
    # cartesian coordinates can be transformed to homogeneous coordinates by appending a 1 to the coordinate vector
    return np.vstack((cartesian_coordinates, np.ones(cartesian_coordinates.shape[1], dtype=np.float32)))


def transform_homogeneous_to_cartesian_coordinates(homogeneous_coordinates: np.ndarray) -> np.ndarray:
    """
    Transforms the given homogeneous coordinates to cartesian coordinates.

    Args:
        homogeneous_coordinates (np.ndarray): the homogeneous coordinates that shall be transformed in shape `[4, n]`

    Returns:
        np.ndarray: array containing the cartesian coordinates in shape `[3, n]`
    """
    # If x_n = 0, this projects to coordinates (0., 0., 0.)
    mask = np.isclose(homogeneous_coordinates[-1], 0.)  # get binary mask for points with x_n = 0
    homogeneous_coordinates[:, ~mask] /= homogeneous_coordinates[-1, ~mask]  # perform the perspective projection for points with x_n != 0
    homogeneous_coordinates[:, mask] = 0.  # set all coordinates for points with x_n = 0 to 0
    return homogeneous_coordinates[:-1]
