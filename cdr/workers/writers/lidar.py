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


import os
from multiprocessing import Queue
from pathlib import Path
from typing import Any, Dict, Optional, Union

import DracoPy
import numpy as np
import pyarrow as pa
import trimesh

from .abstract import DataWriter, ParquetWriterMixin


def encode_point_cloud_draco(vertices: np.ndarray, vertex_attributes: Optional[Dict[str, np.ndarray]] = None,
                             quantization_bits: int = 14) -> bytes:
    """
    Encodes the given vertices and optional vertex_attributes into Draco format.

    Args:
        vertices (np.ndarray): vertices (x, y, z) of points in a point cloud with shape [n, 3]
        vertex_attributes (Optional[Dict[str, np.ndarray]]): Optional additional attributes for each vertex.
            Must be provided as dictionary, that maps the attribute name to a numpy array of shape [n,] or [n, 1].
        quantization_bits (int): amount of quantization bits used by Draco compression

    Returns:
        bytes: Draco-encoded binary blob
    """
    n = vertices.shape[0]
    if vertex_attributes is not None:
        vertex_attributes = {attr_name: attr_vals.reshape((n, 1)) for attr_name, attr_vals in vertex_attributes.items()}

    # Compression level does not really gain anything for point clouds -> set to 1, to reduce performance impact
    return DracoPy.encode(points=vertices, generic_attributes=vertex_attributes,
                          compression_level=1, quantization_bits=quantization_bits)


class LidarDataWriter(ParquetWriterMixin, DataWriter):

    def __init__(self, lidar_queue: Queue,
                 recording_dir: Union[str, Path], sensor_id: str, output_format: str,
                 **output_format_kwargs: Any):
        """
        Writes lidar point clouds (xyz + intensity) to the disk.

        Args:
            lidar_queue (Queue): a queue that receives CARLA's rgb images
            recording_dir (Union[str, Path]): path to root directory, where the recording shall be stored
            sensor_id (str): the ID of the sensor this writer belongs to
            output_format (str): the output format to use
            **output_format_kwargs (Any): additional parameters for output formats
        """
        super().__init__(recording_dir, sensor_id, output_format, output_format_kwargs, main_queue=lidar_queue)

    def _work_iteration(self):
        received_data = self.receive()
        if received_data is None:
            self._reached_end = True
            return
        queue_items, _ = received_data

        frame_id = queue_items['frame_id']
        vertices = queue_items['vertices']
        vertex_attributes = queue_items['vertex_attributes']

        if self.output_format == 'ply':
            sensor_data_filename = os.path.join(self.data_dir, f'{frame_id:06d}.ply')
            mesh = trimesh.Trimesh(vertices=vertices, vertex_attributes=vertex_attributes)
            mesh.export(sensor_data_filename)
        elif self.output_format == 'parquet':
            encoded = encode_point_cloud_draco(vertices, vertex_attributes)
            self.write_to_parquet(frame_id, {'pointcloud_draco_bytes': encoded})
        else:
            raise NotImplementedError('Forgot to implement writing logic after adding new supported output format?!')

    @classmethod
    def get_data_directory(cls, recording_dir: Path) -> Path:
        return recording_dir / 'sensor' / 'lidar'

    @classmethod
    def get_supported_output_formats(cls) -> Dict[str, bool]:
        return {'ply': True, 'parquet': False}

    @classmethod
    def get_parquet_schema(cls) -> pa.Schema:
        return pa.schema([('pointcloud_draco_bytes', pa.binary())])

    @classmethod
    def get_parquet_compression(cls) -> str:
        return 'zstd'


class SemanticLidarDataWriter(LidarDataWriter):

    @classmethod
    def get_data_directory(cls, recording_dir: Path) -> Path:
        return recording_dir / 'ground_truth' / 'sem_lidar'
