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
from typing import List, Union

import imageio.v3 as iio
import numpy as np
import tagpy


class VideoWriter:

    def __init__(self, video_file: Union[str, Path], fps: int, keyframe_every: int = 1):
        self.video_file = Path(video_file)

        self.index_frame_ids: List[str] = []

        self.video_writer = iio.imopen(self.video_file, 'w', plugin='pyav')
        self.video_writer.init_video_stream(codec='libx264', fps=fps, pixel_format='yuv420p',
                                            max_keyframe_interval=keyframe_every, force_keyframes=True)

    def write_frame(self, frame_id: int, frame: np.ndarray):
        self.video_writer.write_frame(frame)
        self.index_frame_ids.append(str(frame_id))

    def close(self):
        self.video_writer.close()

        # Store the frame_ids as CSV in the comment of the video container.
        # ImageIO and PyAV do not support to write a container comment after writing frames to the stream,
        # so we use tagpy do add the comment, after the file is written.
        file_ref = tagpy.FileRef(str(self.video_file.resolve()), False)
        file_ref.tag().comment = ','.join(self.index_frame_ids)
        file_ref.save()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
