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
import os
from pathlib import Path
from typing import Any, Dict, Union

import numpy as np
import pyarrow as pa

from cdr.utils.parquet import ParquetWriter
from cdr.utils.video import VideoWriter
from cdr.workers.base import BaseWorker, IterativeWorkerMixin, ReceiverMixin


class DataWriter(IterativeWorkerMixin, ReceiverMixin, BaseWorker, metaclass=abc.ABCMeta):

    def __init__(self, recording_dir: Union[str, Path], sensor_id: str, output_format: str,
                 output_format_parameters: Dict[str, Any] = {}, *args, **kwargs):
        """
        A `DataWriter` is a specialized `BaseWorker` responsible to write data received from `DataListeners` and/or
        `DataProcessors` to a persistent storage.

        Args:
            recording_dir (Union[str, Path]): path to root directory, where the recording shall be stored
            sensor_id (str): the ID of the sensor this writer belongs to
            output_format (str): the output format to use
            output_format_parameters (Dict[str, Any]): additional parameters for output formats
        """
        super().__init__(*args, **kwargs)
        self.data_dir = self.get_data_directory(Path(recording_dir))
        self.sensor_id = sensor_id
        os.makedirs(self.data_dir, exist_ok=True)

        # Verify output format
        if output_format not in self.get_supported_output_formats().keys():
            raise ValueError(f'Currently unsupported output format "{output_format}" requested for "{type(self).__name__}".')
        self.output_format = output_format
        self.output_format_parameters = output_format_parameters

        # Check if we need a sub-directory for the selected output format
        if self.get_supported_output_formats()[output_format]:
            self.data_dir = self.data_dir / self.sensor_id
            os.makedirs(self.data_dir, exist_ok=True)

    @classmethod
    @abc.abstractmethod
    def get_data_directory(cls, recording_dir: Path) -> Path:
        """
        Returns the path to the data directory of this writer in the given recording directory.

        Args:
            recording_dir (Path): path to root directory of the recording

        Returns:
            Path: path to the data directory of this writer in the given recording directory
        """
        ...

    @classmethod
    @abc.abstractmethod
    def get_supported_output_formats(cls) -> Dict[str, bool]:
        """
        Returns a mapping from supported output formats of this writer, to a boolean indicating if the output format
        requires a sub-directory for individual files, or not.
        """
        ...


class ParquetWriterMixin(DataWriter, metaclass=abc.ABCMeta):

    def write_to_parquet(self, frame_id: int, data_item: Dict[str, Any]):
        """
        Writes the given data item according to this writers schema to the target parquet file, associated to the
        given frame_id.

        Args:
            frame_id (int): the frame ID of the data frame
            data_item (Dict[str, Any]): the data to write, which must conform to the used schema
        """
        assert self.parquet_writer is not None  # method should only be called, if actually writing parquet files
        self.parquet_writer.write_data_item(frame_id, data_item)

    def pre_work(self):
        self.parquet_writer = None
        if self.output_format == 'parquet':
            self.parquet_writer = ParquetWriter(self.data_dir / f'{self.sensor_id}.parquet',
                                                self.get_parquet_schema(), self.get_parquet_compression(),
                                                self.output_format_parameters.get('keyframe_every', 1))
        super().pre_work()

    def post_work(self):
        if self.output_format == 'parquet':
            assert self.parquet_writer is not None
            self.parquet_writer.close()
        super().post_work()

    def on_graceful_exit(self):
        if self.parquet_writer is not None:
            self.parquet_writer.close()
        super().on_graceful_exit()

    @classmethod
    @abc.abstractmethod
    def get_parquet_schema(cls) -> pa.Schema:
        """
        Returns the table schema for the parquet file, that can be written by this DataWriter.
        """
        ...

    @classmethod
    @abc.abstractmethod
    def get_parquet_compression(cls) -> str:
        """
        Returns the compression type for the parquet file, that can be written by this DataWriter.
        """
        ...


class VideoWriterMixin(DataWriter, metaclass=abc.ABCMeta):

    def write_to_video(self, frame_id: int, frame: np.ndarray):
        """
        Writes the given image frame according to the target video file. Frame IDs will be stored in the metadata
        of the video.

        Args:
            frame_id (int): the frame ID of the data frame
            frame (np.ndarray): the image data
        """
        assert self.video_writer is not None  # method should only be called, if actually writing video files
        self.video_writer.write_frame(frame_id, frame)

    def pre_work(self):
        self.video_writer = None
        if self.output_format == 'mp4':
            self.video_writer = VideoWriter(self.data_dir / f'{self.sensor_id}.mp4',
                                            self.output_format_parameters['fps'],
                                            self.output_format_parameters.get('keyframe_every', 1))
        super().pre_work()

    def post_work(self):
        if self.output_format == 'mp4':
            assert self.video_writer is not None
            self.video_writer.close()
        super().post_work()

    def on_graceful_exit(self):
        if self.video_writer is not None:
            self.video_writer.close()
        super().on_graceful_exit()
