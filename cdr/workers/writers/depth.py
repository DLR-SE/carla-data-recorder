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


from multiprocessing import Queue
from pathlib import Path
from typing import Any, Dict, Union

import imageio
imageio.plugins.freeimage.download()  # required for EXR
import imageio.v3 as iio
import numpy as np
import pyarrow as pa
from PIL import Image

from .abstract import DataWriter, ParquetWriterMixin
from cdr.utils.camera_utils import transform_z_depth_to_depth


def write_depth_exr(depth_meters: np.ndarray, uri: Union[str, Path]) -> Union[None, bytes]:
    # PXR24_COMPRESSION is not lossless, but a very good compromise between filesize and information loss.
    # More details on this can be found in the documentation describing the depth data.
    # Flags: https://github.com/imageio/imageio-freeimage/blob/master/imageio_freeimage/_freeimage.py
    return iio.imwrite(uri, depth_meters.astype(np.float32), extension='.exr',
                       flags=0x0001 | 0x0010)  # EXR_FLOAT | EXR_PXR24


class DepthDataWriter(ParquetWriterMixin, DataWriter):

    def __init__(self, depth_queue: Queue, intrinsic_matrix: np.ndarray,
                 recording_dir: Union[str, Path], sensor_id: str, output_format: str,
                 **output_format_kwargs: Any):
        """
        Writes depth information to the disk.

        Args:
            depth_queue (Queue): a queue that receives CARLA's depth images
            intrinsic_matrix (np.ndarray): the intrinsic calibration of the camera, associated with this writer
            recording_dir (Union[str, Path]): path to root directory, where the recording shall be stored
            sensor_id (str): the ID of the sensor this writer belongs to
            output_format (str): the output format to use
            **output_format_kwargs (Any): additional parameters for output formats
        """
        super().__init__(recording_dir, sensor_id, output_format, output_format_kwargs, main_queue=depth_queue)
        self.intrinsic_matrix = intrinsic_matrix

    def _work_iteration(self):
        received_data = self.receive()
        if received_data is None:
            self._reached_end = True
            return
        queue_items, _ = received_data

        frame_id = queue_items['frame_id']
        image_array = queue_items['image_array'].astype(np.uint32)

        z_depths = image_array[Ellipsis, 0] + image_array[Ellipsis, 1] * 256 + image_array[Ellipsis, 2] * 256**2
        z_depths = z_depths / (256**3 - 1) * 1000
        depth_in_meters = transform_z_depth_to_depth(z_depths, self.intrinsic_matrix)

        sensor_data_filename = self.data_dir / f'{frame_id:06d}.{self.output_format}'
        if self.output_format == 'png':
            depth_image = depth_in_meters * 100
            depth_image[depth_image > 2**16 - 1] = 2 ** 16 - 1
            depth_image = depth_image.astype(np.uint16)
            # compress_type=3 == RLE; works best in terms of compression for depth
            Image.fromarray(depth_image).save(sensor_data_filename, compress_type=3)
        elif self.output_format == 'exr':
            write_depth_exr(depth_in_meters, sensor_data_filename)
        elif self.output_format == 'parquet':
            depth_exr = write_depth_exr(depth_in_meters, '<bytes>')
            self.write_to_parquet(frame_id, {'depth_meters_exr': depth_exr})
        else:
            raise NotImplementedError('Forgot to implement writing logic after adding new supported output format?!')

    @classmethod
    def get_data_directory(cls, recording_dir: Path) -> Path:
        return recording_dir / 'ground_truth' / 'depth'

    @classmethod
    def get_supported_output_formats(cls) -> Dict[str, bool]:
        return {'png': True, 'exr': True, 'parquet': False}

    @classmethod
    def get_parquet_schema(cls) -> pa.Schema:
        return pa.schema([('depth_meters_exr', pa.binary())])

    @classmethod
    def get_parquet_compression(cls) -> str:
        return 'none'  # already compressed with PXR24 -> no further compression
