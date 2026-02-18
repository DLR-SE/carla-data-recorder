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
from typing import Any, Dict, Union

from PIL import Image
import numpy as np

from .abstract import DataWriter, VideoWriterMixin


class RGBDataWriter(VideoWriterMixin, DataWriter):

    def __init__(self, rgb_queue: Queue,
                 recording_dir: Union[str, Path], sensor_id: str, output_format: str,
                 ssaa_enabled: bool, image_scaling_factor: float,
                 **output_format_kwargs: Any):
        """
        Writes default camera images to the disk.

        Args:
            rgb_queue (Queue): a queue that receives CARLA's rgb images
            recording_dir (Union[str, Path]): path to root directory, where the recording shall be stored
            sensor_id (str): the ID of the sensor this writer belongs to
            output_format (str): the output format to use
            ssaa_enabled (bool): whether or not super-sampling aliasing is enabled
            image_scaling_factor (float): the scaling factor of the images, if super-sampling aliasing is enabled
            **output_format_kwargs (Any): additional parameters for output formats
        """
        super().__init__(recording_dir, sensor_id, output_format, output_format_kwargs, main_queue=rgb_queue)
        self.ssaa_enabled = ssaa_enabled
        self.image_scaling_factor = image_scaling_factor

    def _work_iteration(self):
        received_data = self.receive()
        if received_data is None:
            self._reached_end = True
            return
        queue_items, _ = received_data

        frame_id = queue_items['frame_id']
        image_array = queue_items['image_array']

        image = Image.fromarray(image_array)
        if self.ssaa_enabled:
            image = image.resize((int(image_array.shape[1] / self.image_scaling_factor),
                                  int(image_array.shape[0] / self.image_scaling_factor)),
                                  Image.Resampling.LANCZOS)

        if self.output_format in ['jpg', 'png']:
            sensor_data_filename = os.path.join(self.data_dir, f'{frame_id:06d}.{self.output_format}')
            # In experiments, we found compress_type=2 to work well in terms of quality and speed, both for PNG and JPG
            image.save(sensor_data_filename, compress_type=2)
        elif self.output_format == 'mp4':
            self.write_to_video(frame_id, np.array(image))
        else:
            raise NotImplementedError('Forgot to implement writing logic after adding new supported output format?!')

    @classmethod
    def get_data_directory(cls, recording_dir: Path) -> Path:
        return recording_dir / 'sensor' / 'camera'

    @classmethod
    def get_supported_output_formats(cls) -> Dict[str, bool]:
        return {'jpg': True, 'png': True, 'mp4': False}
