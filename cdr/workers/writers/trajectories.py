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
from typing import Any, Dict, Union

import pyarrow as pa

from .abstract import DataWriter, ParquetWriterMixin
from cdr.utils.json_utils import customs_to_json
from cdr.workers.writers.bounding_boxes import FIELDS_3D_BBS


class TrajectoriesDataWriter(ParquetWriterMixin, DataWriter):

    def __init__(self, snapshot_queue: Queue,
                 recording_dir: Union[str, Path], output_format: str,
                 **output_format_kwargs: Any):
        """
        Writes trajectory information to the disk.

        Args:
            snapshot_queue (Queue): a queue that receives CARLA's world snapshots
            recording_dir (Union[str, Path]): path to root directory, where the recording shall be stored
            output_format (str): the output format to use
            **output_format_kwargs (Any): additional parameters for output formats
        """
        super().__init__(recording_dir, 'trajectories', output_format, output_format_kwargs, main_queue=snapshot_queue)

    def _work_iteration(self):
        received_data = self.receive()
        if received_data is None:
            self._reached_end = True
            return
        queue_items, _ = received_data

        frame_id = queue_items['frame_id']
        timestamp = queue_items['timestamp']
        actors = queue_items['actors']
        trajectory_data = {
            'timestamp': timestamp,
            'actors': actors
        }

        if self.output_format == 'json':
            with open(os.path.join(self.data_dir, f'{frame_id:06d}.json'), 'wt') as file:
                json.dump(trajectory_data, file, default=customs_to_json)
        elif self.output_format == 'parquet':
            self.write_to_parquet(frame_id, trajectory_data)
        else:
            raise NotImplementedError('Forgot to implement writing logic after adding new supported output format?!')

    @classmethod
    def get_data_directory(cls, recording_dir: Path) -> Path:
        return recording_dir / 'ground_truth'

    @classmethod
    def get_supported_output_formats(cls) -> Dict[str, bool]:
        return {'json': True, 'parquet': False}

    @classmethod
    def get_parquet_schema(cls) -> pa.Schema:
        return pa.schema([('timestamp', pa.float32()), ('actors', pa.map_(pa.uint32(), pa.struct(FIELDS_3D_BBS)))])

    @classmethod
    def get_parquet_compression(cls) -> str:
        return 'zstd'
