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


from pathlib import Path
from typing import Any, Dict, List, Union

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from pyquaternion import Quaternion


TYPE_MAPPING = {
    Quaternion: lambda quaternion: quaternion.elements
}

def _map_leaf_types(data_dict: Dict[str, Any]) -> Dict[str, Any]:
    data_dict_mapped = {}
    for key, val in data_dict.items():
        if isinstance(val, dict):
            val = _map_leaf_types(val)
        elif type(val) in TYPE_MAPPING:
            val = TYPE_MAPPING[type(val)](val)
        data_dict_mapped[key] = val
    return data_dict_mapped


class ParquetWriter:

    def __init__(self, parquet_file: Union[str, Path], schema: pa.Schema, compression: str,
                 row_group_size: int, write_index: bool = True) -> None:
        self.parquet_file = Path(parquet_file)
        # Append frame_id to the end of the schema, which is consistent with assigning this column as index,
        # which will also move it to the end when converting a DataFrame to a PyArrow-Table.
        self.schema = schema.append(pa.field('frame_id', pa.uint32()))
        self.row_group_size = row_group_size
        self.write_index = write_index

        self.cur_row_group = 0
        self.cur_data_batch = []
        self.metadata = None

        self.index_frame_ids: List[int] = []
        self.index_row_groups: List[int] = []

        self.parquet_writer = pq.ParquetWriter(self.parquet_file, self.schema, compression=compression)

    def write_data_item(self, frame_id: int, data_item: Dict[str, Any]):
        """
        Writes the given data item according to this writers schema to the target parquet file, associated to the
        given frame_id. If a row_group_size > 1 is used, data items are cached until the enough items can be written
        as batch with the desired row_group_size.

        Args:
            frame_id (int): the frame ID of the data frame
            data_item (Dict[str, Any]): the data to write, which must conform to the used schema
        """
        # Add the frame_id and current row group to the index, which will be written when closing this writer,
        # which will enable faster access to individual row groups based on frame_ids.
        self.index_frame_ids.append(frame_id)
        self.index_row_groups.append(self.cur_row_group)

        # Map types, that cannot be written directly to parquet
        data_item = _map_leaf_types(data_item)
        # Combine actual data with frame_id
        data = {**data_item, 'frame_id': frame_id}
        self.cur_data_batch.append(data)

        # Write the batch, if we hit the determined row group size
        if len(self.cur_data_batch) == self.row_group_size:
            self._write_batch()
            self.cur_data_batch = []
            self.cur_row_group += 1

    def _write_batch(self):
        """
        Writes the current batch of data rows with 'frame_id' column as index to the target parquet file.
        The number of rows in the batch determines the size of the written row group.
        """
        df = pd.DataFrame(self.cur_data_batch).set_index('frame_id')
        table = pa.Table.from_pandas(df, schema=self.schema, preserve_index=True)
        self.parquet_writer.write(table)

        if self.metadata is None:
            self.metadata = table.schema.metadata

    def close(self):
        # Write the last batch, if non-empty
        if len(self.cur_data_batch) > 0:
            self._write_batch()
            self.cur_data_batch = []
        # Add metadata, which is necessary to reconstruct DataFrame correctly when reading the parquet file back in
        self.parquet_writer.add_key_value_metadata(self.metadata)
        self.parquet_writer.close()

        if self.write_index:
            # Write the index file next to the actual parquet file
            index_path = self.parquet_file.parent / (self.parquet_file.stem + '_index.parquet')
            index = pd.DataFrame({'frame_id': self.index_frame_ids, 'row_group': self.index_row_groups}, dtype=np.uint32)
            index.set_index('frame_id', inplace=True)
            index.to_parquet(index_path, compression=None, index=True)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
