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
from multiprocessing import Queue
from pathlib import Path
from typing import Any, Dict, Union

import numpy as np
import pyarrow as pa
from PIL import Image

from cdr.utils.carla_classes import SEGMENTATION_PIL_PALETTE
from .abstract import DataWriter, ParquetWriterMixin


class SegmentationDataWriter(ParquetWriterMixin, DataWriter):

    def __init__(self, panopt_queue: Queue, segmentation_key: str,
                 recording_dir: Union[str, Path], sensor_id: str, output_format: str,
                 **output_format_kwargs: Any):
        """
        Writes segmentation images to the disk.

        Args:
            panopt_queue (Queue): a queue that receives the refined panoptic segmentation of a
                `InstanceSegmentationDataProcessor`
            segmentation_key (str): the key in the dictionary of the panopt_queue items, that is used to retrieve
                the desired segmentation item
            recording_dir (Union[str, Path]): path to root directory, where the recording shall be stored
            sensor_id (str): the ID of the sensor this writer belongs to
            output_format (str): the output format to use
        """
        super().__init__(recording_dir, sensor_id, output_format, output_format_kwargs, main_queue=panopt_queue)
        self.segmentation_key = segmentation_key

    def _work_iteration(self):
        received_data = self.receive()
        if received_data is None:
            self._reached_end = True
            return
        queue_items, _ = received_data

        frame_id = queue_items['frame_id']
        segmentation = queue_items[self.segmentation_key]

        if self.output_format == 'png':
            sensor_data_filename = self.data_dir / f'{frame_id:06d}.png'
            self.write_to_png(segmentation, sensor_data_filename)
        elif self.output_format == 'parquet':
            data = {'shape': list(segmentation.shape), 'dtype': str(segmentation.dtype), 'array_bytes': segmentation.tobytes()}
            self.write_to_parquet(frame_id, data)
        else:
            raise NotImplementedError('Forgot to implement writing logic after adding new supported output format?!')

    @abc.abstractmethod
    def write_to_png(self, segmentation: np.ndarray, file_path: Union[str, Path]):
        """
        Writes the given segmentation array to a PNG file.

        Args:
            segmentation (np.ndarray): the segmentation array (e.g. semantic segmentation, instance segmentation)
            file_path (Union[str, Path]): the file path to the resulting PNG file
        """
        ...

    @classmethod
    def get_supported_output_formats(cls) -> Dict[str, bool]:
        return {'png': True, 'parquet': False}

    @classmethod
    def get_parquet_schema(cls) -> pa.Schema:
        return pa.schema([('shape', pa.list_(pa.int32())), ('dtype', pa.string()), ('array_bytes', pa.binary())])

    @classmethod
    def get_parquet_compression(cls) -> str:
        return 'zstd'


class SemanticSegmentationDataWriter(SegmentationDataWriter):

    def __init__(self, panopt_queue: Queue,
                 recording_dir: Union[str, Path], sensor_id: str, output_format: str,
                 **output_format_kwargs: Any):
        """
        Writes semantic segmentation images to the disk.

        Args:
            panopt_queue (Queue): a queue that receives the refined panoptic segmentation of a
                `InstanceSegmentationDataProcessor`
            recording_dir (Union[str, Path]): path to root directory, where the recording shall be stored
            sensor_id (str): the ID of the sensor this writer belongs to
            output_format (str): the output format to use
            **output_format_kwargs (Any): additional parameters for output formats
        """
        super().__init__(panopt_queue, 'sem_seg', recording_dir, sensor_id, output_format, **output_format_kwargs)

    def write_to_png(self, segmentation: np.ndarray, file_path: Union[str, Path]):
        sem_seg_image = Image.fromarray(segmentation)
        sem_seg_image = sem_seg_image.convert('P')
        sem_seg_image.putpalette(SEGMENTATION_PIL_PALETTE)
        sem_seg_image.save(file_path)  # will be written as indexed 8-bit png

    @classmethod
    def get_data_directory(cls, recording_dir: Path) -> Path:
        return recording_dir / 'ground_truth' / 'sem_seg'


class InstanceSegmentationDataWriter(SegmentationDataWriter):

    def __init__(self, panopt_queue: Queue,
                 recording_dir: Union[str, Path], sensor_id: str, output_format: str,
                 **output_format_kwargs: Any):
        """
        Writes instance segmentation images to the disk.

        Args:
            panopt_queue (Queue): a queue that receives the refined panoptic segmentation of a
                `InstanceSegmentationDataProcessor`
            recording_dir (Union[str, Path]): path to root directory, where the recording shall be stored
            sensor_id (str): the ID of the sensor this writer belongs to
            output_format (str): the output format to use
            **output_format_kwargs (Any): additional parameters for output formats
        """
        super().__init__(panopt_queue, 'inst_seg', recording_dir, sensor_id, output_format, **output_format_kwargs)

    def write_to_png(self, segmentation: np.ndarray, file_path: Union[str, Path]):
        inst_seg_image = Image.fromarray(segmentation.astype(np.uint16))
        inst_seg_image.save(file_path)  # will be written as 16-bit png

    @classmethod
    def get_data_directory(cls, recording_dir: Path) -> Path:
        return recording_dir / 'ground_truth' / 'inst_seg'
