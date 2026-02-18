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


import json
import os
from multiprocessing import Queue
from pathlib import Path
from typing import Any, Dict, Tuple, Union

import numpy as np
import pyarrow as pa
from pyquaternion import Quaternion

from cdr.utils.camera_utils import (apply_transformation, pose_to_transformation_matrix,
                                    project_world_to_image_coordinates)
from cdr.utils.json_utils import customs_to_json
from .abstract import DataWriter, ParquetWriterMixin


FIELDS_2D_BBS = [pa.field('class_id', pa.string()), pa.field('c_x', pa.float32()), pa.field('c_y', pa.float32()),
                 pa.field('w', pa.float32()), pa.field('h', pa.float32()), pa.field('type_id', pa.string()),
                 pa.field('role_name', pa.string())]


FIELDS_3D_BBS = [pa.field('class_id', pa.string()), pa.field('center', pa.list_(pa.float32())),
                 pa.field('size', pa.list_(pa.float32())), pa.field('rot', pa.list_(pa.float32())),
                 pa.field('velocity', pa.list_(pa.float32())), pa.field('acceleration', pa.list_(pa.float32())),
                 pa.field('angular_velocity', pa.list_(pa.float32())), pa.field('type_id', pa.string()),
                 pa.field('role_name', pa.string())]


class BB3DDataWriter(ParquetWriterMixin, DataWriter):

    def __init__(self, snapshot_queue: Queue, panopt_queue: Queue,
                 recording_dir: Union[str, Path], sensor_id: str, output_format: str,
                 filter_ego: bool, ego_role_name: str,
                 required_img_frac_per_actor: float = 0.000075,
                 **output_format_kwargs: Any):
        """
        Writes 3D Bounding Box information to the disk.

        Args:
            snapshot_queue (Queue): a queue that receives CARLA's world snapshots
            panopt_queue (Queue): a queue that receives the refined panoptic segmentation of a
                `InstanceSegmentationDataProcessor`
            recording_dir (Union[str, Path]): path to root directory, where the recording shall be stored
            output_format (str): the output format to use
            filter_ego (bool): whether to filter ego from the bounding boxes (True) or not (False)
            ego_role_name (str): the `role_name` of the ego vehicle
            required_img_frac_per_actor (float, optional): the fraction of an image that an actor must at least
                occupy in order to be annotated (i.e. small, far away actors are discarded). Defaults to 0.000075.
            **output_format_kwargs (Any): additional parameters for output formats
        """
        super().__init__(recording_dir, sensor_id, output_format, output_format_kwargs,
                         main_queue=panopt_queue, secondary_queues=(snapshot_queue,))
        self.filter_ego = filter_ego
        self.ego_role_name = ego_role_name
        self.required_img_frac_per_actor = required_img_frac_per_actor

    def _work_iteration(self):
        received_data = self.receive()
        if received_data is None:
            self._reached_end = True
            return
        panopt_queue_item, (snapshot_queue_item,) = received_data

        frame_id = panopt_queue_item['frame_id']

        instseg_actor_ids = panopt_queue_item['inst_seg']
        actors_info: Dict[int, Dict[str, Any]] = snapshot_queue_item['actors']
        sensor_location = panopt_queue_item['location']
        sensor_rotation = panopt_queue_item['rotation']

        actor_ids, pixel_counts = np.unique(instseg_actor_ids, return_counts=True)
        required_pixels_per_actor = int(np.size(instseg_actor_ids) * self.required_img_frac_per_actor)
        # Filter those actors in the instance segmentation, which are just barely visible
        valid_actor_ids = []
        for i in range(len(actor_ids)):
            actor_id = actor_ids[i]
            if actor_id == 0:  # skip non-actor pixels
                continue
            if pixel_counts[i] >= required_pixels_per_actor:  # must be visible to some degree
                if not self.filter_ego or actors_info[actor_id]['role_name'] != self.ego_role_name:
                    valid_actor_ids.append(actor_id)

        # Keep only those actors, which are visible in the instance segmentation
        def _filter_valid_actors(id_w_info: Tuple[int, Dict[str, Any]]) -> bool:
            return id_w_info[0] in valid_actor_ids
        bb_3d_dict = dict(filter(_filter_valid_actors, actors_info.items()))

        # Post-process the BBs if necessary (relevant for subclasses)
        bb_3d_dict = self._post_process(bb_3d_dict, sensor_location, sensor_rotation)

        if self.output_format == 'json':
            with open(os.path.join(self.data_dir, f'{frame_id:06d}.json'), 'wt') as file:
                json.dump(bb_3d_dict, file, default=customs_to_json)
        elif self.output_format == 'parquet':
            self.write_to_parquet(frame_id, {'actors': bb_3d_dict})
        else:
            raise NotImplementedError('Forgot to implement writing logic after adding new supported output format?!')

    def _post_process(self, bb_dict: Dict[int, Dict[str, Any]],
                      sensor_location: np.ndarray, sensor_rotation: Quaternion) -> Dict[int, Dict[str, Any]]:
        return bb_dict

    @classmethod
    def get_data_directory(cls, recording_dir: Path) -> Path:
        return recording_dir / 'ground_truth' / '3d_bbs'

    @classmethod
    def get_supported_output_formats(cls) -> Dict[str, bool]:
        return {'json': True, 'parquet': False}

    @classmethod
    def get_parquet_schema(cls) -> pa.Schema:
        return pa.schema([('actors', pa.map_(pa.uint32(), pa.struct(FIELDS_3D_BBS)))])

    @classmethod
    def get_parquet_compression(cls) -> str:
        return 'zstd'


class BB2DDataWriter(BB3DDataWriter):

    def __init__(self, snapshot_queue: Queue, panopt_queue: Queue, intrinsic_matrix: np.ndarray,
                 recording_dir: Union[str, Path], sensor_id: str, output_format: str,
                 filter_ego: bool, ego_role_name: str,
                 required_img_frac_per_actor: float = 0.000075,
                 **output_format_kwargs: Any):
        """
        Writes 2D Bounding Box information to the disk.

        Args:
            snapshot_queue (Queue): a queue that receives CARLA's world snapshots
            panopt_queue (Queue): a queue that receives the refined panoptic segmentation of a
                `InstanceSegmentationDataProcessor`
            intrinsic_matrix (np.ndarray): the intrinsic calibration of the camera, associated with this writer
            recording_dir (Union[str, Path]): path to root directory, where the recording shall be stored
            sensor_id (str): the ID of the sensor this writer belongs to
            output_format (str): the output format to use
            filter_ego (bool): whether to filter ego from the bounding boxes (True) or not (False)
            ego_role_name (str): the `role_name` of the ego vehicle
            required_img_frac_per_actor (float, optional): the fraction of an image that an actor must at least
                occupy in order to be annotated (i.e. small, far away actors are discarded). Defaults to 0.000075.
            **output_format_kwargs (Any): additional parameters for output formats
        """
        super().__init__(snapshot_queue, panopt_queue, recording_dir, sensor_id, output_format,
                         filter_ego, ego_role_name, required_img_frac_per_actor, **output_format_kwargs)
        self.intrinsic_matrix = intrinsic_matrix

    def _post_process(self, bb_dict: Dict[int, Dict[str, Any]],
                      sensor_location: np.ndarray, sensor_rotation: Quaternion) -> Dict[int, Dict[str, Any]]:
        """
        Projects the given 3D bounding boxes to 2D, using the camera's location and rotation as well as the intrinsic
        calibration of the camera.

        Args:
            bb_dict (Dict[int, Dict[str, Any]]): a dict of 3D bounding boxes, that will be processed
            sensor_location (np.ndarray): the location of the camera
            sensor_rotation (Quaternion): the rotation of the camera

        Returns:
            Dict[int, Dict[str, Any]]: a dict of 2D bounding boxes
        """
        sensor_transformation_matrix = pose_to_transformation_matrix(sensor_location, sensor_rotation)
        processed_bb_dict = {}
        for actor_id, bb_info in bb_dict.items():
            half_length, half_width, half_height = bb_info['size'] / 2
            bb_3d_points_local = np.array([
                [+half_length, +half_width, +half_height],
                [+half_length, +half_width, -half_height],
                [+half_length, -half_width, +half_height],
                [+half_length, -half_width, -half_height],
                [-half_length, +half_width, +half_height],
                [-half_length, +half_width, -half_height],
                [-half_length, -half_width, +half_height],
                [-half_length, -half_width, -half_height]
            ])
            # Transform the vertices of the local bounding box to their global positions
            bb_transformation_matrix = pose_to_transformation_matrix(bb_info['center'], bb_info['rot'])
            bb_3d_points_global = apply_transformation(bb_3d_points_local, bb_transformation_matrix, sample_axis=0)
            # Project the global vertices to the image plane
            bb_2d_points, _ = project_world_to_image_coordinates(bb_3d_points_global, self.intrinsic_matrix,
                                                                 sensor_transformation_matrix, sample_axis=0)

            x_min, y_min = np.min(bb_2d_points, axis=0)
            x_max, y_max = np.max(bb_2d_points, axis=0)

            processed_bb_dict[actor_id] = {
                'class_id': bb_info['class_id'],
                'c_x': x_min + (x_max - x_min) / 2,
                'c_y': y_min + (y_max - y_min) / 2,
                'w': x_max - x_min,
                'h': y_max - y_min,
                # 'occlusion': ?,  # TODO
                # 'truncation': ?,  # TODO
                'type_id': bb_info['type_id'],
                'role_name': bb_info['role_name'],
            }
        return processed_bb_dict

    @classmethod
    def get_data_directory(cls, recording_dir: Path) -> Path:
        return recording_dir / 'ground_truth' / '2d_bbs'

    @classmethod
    def get_parquet_schema(cls) -> pa.Schema:
        return pa.schema([('actors', pa.map_(pa.uint32(), pa.struct(FIELDS_2D_BBS)))])
