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


import argparse
import json
import os
import shutil
from concurrent.futures import Future, ProcessPoolExecutor
from pathlib import Path
from typing import Any, Callable, Dict, List, Tuple, Union

import imageio.v3 as iio
import numpy as np
import pyarrow as pa
import tqdm
from trimesh.exchange.ply import load_ply

from cdr.utils.parquet import ParquetWriter
from cdr.utils.video import VideoWriter
from cdr.workers.writers import (BB2DDataWriter, BB3DDataWriter, DepthDataWriter, GNSSDataWriter, IMUDataWriter,
                                 LidarDataWriter, SegmentationDataWriter, TrajectoriesDataWriter)
from cdr.workers.writers.depth import write_depth_exr
from cdr.workers.writers.lidar import encode_point_cloud_draco


def write_files_to_parquet(input_files: List[Path], output_file: Path, schema: pa.Schema, compression: str,
                           parse_func: Callable[[Path], dict[str, Any]], *,
                           row_group_size: int = 1):
    with ParquetWriter(output_file, schema, compression, row_group_size) as writer:
        for input_file in input_files:
            # Get current frame_id as int and parse the data from the file
            frame_id = int(input_file.stem)
            data = parse_func(input_file)

            # Write to parquet file
            writer.write_data_item(frame_id, data)


def images_to_video(data_dir: Union[str, Path], output_file: Union[str, Path], fps: int, *,
                    keyframe_every: int = 1):
    """
    Reads all image files (.png or .jpg) in the given `data_dir` (sorted by filename) and creates a
    video (.mp4 with H264 codec) using the given frame rate, which will be stored to `output_file`.

    The video will use every second frame as keyframe, resulting in faster access of individual frames of the video
    in non-sequential access scenarios.

    Args:
        data_dir (Union[str, Path]): directory containing the input images
        output_file (Union[str, Path]): path to resulting video file
        fps (int): frames per second of the video
        keyframe_every (int): specifies the GOP size, i.e. after how many frames a new keyframe is inserted
    """
    data_dir = Path(data_dir)
    output_file = Path(output_file)
    assert fps > 0

    image_files = sorted(list(data_dir.glob('*.png')) + list(data_dir.glob('*.jpg')))

    with VideoWriter(output_file, fps, keyframe_every) as writer:
        # Write the frames
        for img_path in image_files:
            frame_id = int(img_path.stem)
            frame = iio.imread(img_path)
            writer.write_frame(frame_id, frame)


def video_to_images(video_file: Union[str, Path], output_dir: Union[str, Path]):
    """
    Extracts every frame from the given video file and stores them as images
    in `output_dir`, ordered by frame index.

    Args:
        video_file (Union[str, Path]): path to the input video file
        output_dir (Union[str, Path]): directory where extracted frames will be stored
        image_format (str): image format to use ("png" or "jpg")
    """
    video_file = Path(video_file)
    output_dir = Path(output_dir)

    output_dir.mkdir(parents=True, exist_ok=True)

    total_frames = iio.improps(video_file, plugin="pyav").shape[0]
    with iio.imopen(video_file, 'r', plugin='pyav') as reader:
        # Read the CSV-encoded frame_ids from the comment of the container
        frame_ids = reader.container_metadata['comment'].split(',')
        for idx in range(total_frames):
            filename = output_dir / f'{frame_ids[idx]}.jpg'
            frame = reader.read(index=idx)
            iio.imwrite(filename, frame, quality=95)


def depths_to_parquet(data_dir: Union[str, Path], output_file: Union[str, Path], *,
                      keyframe_every: int = 1):
    data_dir = Path(data_dir)
    output_file = Path(output_file)

    schema = DepthDataWriter.get_parquet_schema()

    def parse_func(input_file: Path):
        depth = iio.imread(input_file)
        depth_meters = depth.astype(np.float32) / 100 if input_file.suffix == '.png' else depth
        depth_exr = write_depth_exr(depth_meters, '<bytes>')
        return {'depth_meters_exr': depth_exr}

    # already compressed with PXR24 -> no further compression
    write_files_to_parquet(sorted(list(data_dir.glob('*.exr')) + list(data_dir.glob('*.png'))),
                           output_file, schema, 'none', parse_func, row_group_size=keyframe_every)


def segmentations_to_parquet(data_dir: Union[str, Path], output_file: Union[str, Path], *,
                             keyframe_every: int = 1):
    data_dir = Path(data_dir)
    output_file = Path(output_file)

    schema = SegmentationDataWriter.get_parquet_schema()

    def parse_func(input_file: Path):
        # Store the image-array as binary encoding. The parquet compression will take care of reducing the file size.
        seg_array = iio.imread(input_file, pilmode='P')
        return {'shape': list(seg_array.shape), 'dtype': str(seg_array.dtype), 'array_bytes': seg_array.tobytes()}

    write_files_to_parquet(sorted(data_dir.glob('*.png')), output_file, schema, 'zstd', parse_func,
                           row_group_size=keyframe_every)


def pointclouds_to_parquet(data_dir: Union[str, Path], output_file: Union[str, Path], *,
                           keyframe_every: int = 1):
    """
    Reads all point cloud files (.ply) in the given `data_dir` (sorted by filename) and creates a single parquet file
    at `output_file`, which stores the frame_id (the basename of the file) and a Draco-encoded binary representation
    for every frame in the directory.

    Stored .ply files may contain additional attributes next to the x,y,z points:
    - either an additional 'intensity' value
    - or 'cosine_incident_angle' + 'actor_id' + 'semseg_id' values

    Args:
        data_dir (Union[str, Path]): directory containing the input point clouds
        output_file (Union[str, Path]): path to resulting parquet file
    """
    data_dir = Path(data_dir)
    output_file = Path(output_file)

    schema = LidarDataWriter.get_parquet_schema()

    def parse_func(input_file: Path):
        # Load the PLY file and encode it using Draco to reduce file sizes, then store the binary blob
        with open(input_file, 'rb') as f:
            ply = load_ply(f)
        pc = ply['metadata']['_ply_raw']['vertex']['data']
        # Get the xyz-points...
        vertices = np.stack((pc['x'], pc['y'], pc['z']), axis=-1).astype(np.float32)
        # ... and the further generic attributes
        vertex_attributes = {}
        for name in pc.dtype.names:
            if name in {'x', 'y', 'z'}:
                continue
            vertex_attributes[name] = pc[name].reshape((-1, 1))
        encoded = encode_point_cloud_draco(vertices, vertex_attributes)

        return {'pointcloud_draco_bytes': encoded}

    write_files_to_parquet(sorted(data_dir.glob('*.ply')), output_file, schema, 'zstd', parse_func,
                           row_group_size=keyframe_every)


def jsons_to_parquet(data_dir: Union[str, Path], output_file: Union[str, Path], *,
                     keyframe_every: int = 1):
    """
    Reads all 2D/3D bounding box files (.json) in the given `data_dir` (sorted by filename) and creates a single parquet
    file at `output_file`, which stores the frame_id (the basename of the file) with the 2D/3D bounding box attributes
    of all actors for every frame in the directory.

    Args:
        data_dir (Union[str, Path]): directory containing the input point clouds
        output_file (Union[str, Path]): path to resulting parquet file
    """
    data_dir = Path(data_dir)
    output_file = Path(output_file)

    # Depending on the actual JSON schema, we have to determine the corresponding PyArrow schema
    embed_actors = False
    if f'ground_truth{os.path.sep}2d_bbs' in str(data_dir.resolve()):
        schema = BB2DDataWriter.get_parquet_schema()
        embed_actors = True
    elif f'ground_truth{os.path.sep}3d_bbs' in str(data_dir.resolve()):
        schema = BB3DDataWriter.get_parquet_schema()
        embed_actors = True
    elif f'ground_truth{os.path.sep}trajectories' in str(data_dir.resolve()):
        schema = TrajectoriesDataWriter.get_parquet_schema()
    elif f'sensor{os.path.sep}gnss' in str(data_dir.resolve()):
        schema = GNSSDataWriter.get_parquet_schema()
    elif f'sensor{os.path.sep}imu' in str(data_dir.resolve()):
        schema = IMUDataWriter.get_parquet_schema()
    else:
        raise NotImplementedError()

    def parse_func(json_path: Path):
        with json_path.open('r', encoding='utf-8') as f:
            content = json.load(f)

        data: Dict[str, Any]
        if embed_actors:
            # For 2d_bbs and 3d_bbs, JSON has no top-level column name -> insert it
            data = {'actors': content}
        else:
            data = content

        # If there are actors, turn the IDs from string (due to JSON) back to int
        if 'actors' in data.keys():
            data['actors'] = dict(map(lambda item: (int(item[0]), item[1]), data['actors'].items()))

        return data

    write_files_to_parquet(sorted(data_dir.glob('*.json')), output_file, schema, 'zstd', parse_func,
                           row_group_size=keyframe_every)


SENSOR_CONVERSIONS: Dict[str, Tuple[Callable, str]] = {
    f'sensor{os.path.sep}camera{os.path.sep}': (images_to_video, '.mp4'),
    f'sensor{os.path.sep}lidar{os.path.sep}': (pointclouds_to_parquet, '.parquet'),
    f'sensor{os.path.sep}gnss{os.path.sep}': (jsons_to_parquet, '.parquet'),
    f'sensor{os.path.sep}imu{os.path.sep}': (jsons_to_parquet, '.parquet'),
    f'ground_truth{os.path.sep}2d_bbs{os.path.sep}': (jsons_to_parquet, '.parquet'),
    f'ground_truth{os.path.sep}3d_bbs{os.path.sep}': (jsons_to_parquet, '.parquet'),
    f'ground_truth{os.path.sep}depth{os.path.sep}': (depths_to_parquet, '.parquet'),
    f'ground_truth{os.path.sep}sem_seg{os.path.sep}': (segmentations_to_parquet, '.parquet'),
    f'ground_truth{os.path.sep}inst_seg{os.path.sep}': (segmentations_to_parquet, '.parquet'),
}

def dataset_to_blobs(root_dir: Union[str, Path], keyframes_every: Dict[str, int] = {},
                     remove_converted_dirs: bool = False):
    pool = ProcessPoolExecutor()
    converted_dirs: List[str] = []
    futures: List[Future] = []

    default_keyframes = keyframes_every.get('all', 1)

    for sensor_dir, subdirs, _ in os.walk(root_dir):
        if len(subdirs) > 0:
            continue  # directories containing actual data have no further subdirectories

        future = None
        for subpath, (convert_func, file_extension) in SENSOR_CONVERSIONS.items():
            if subpath in sensor_dir:
                recording_root, sensor_name = sensor_dir.split(subpath)
                data_type = subpath.split(os.path.sep)[1]

                kwargs = {'keyframe_every': keyframes_every.get(data_type, default_keyframes)}
                if subpath == f'sensor{os.path.sep}camera{os.path.sep}':
                    with open(Path(recording_root) / 'sensor_configuration.json') as config_file:
                        kwargs['fps'] = json.load(config_file)[sensor_name]['capture_frequency']

                future = pool.submit(convert_func, sensor_dir, Path(sensor_dir).parent / (sensor_name + file_extension),
                                     **kwargs)
                break  # no need to continue searching for this subdir

        if future is None and f'ground_truth{os.path.sep}trajectories' in sensor_dir:
            kwargs = {'keyframe_every': keyframes_every.get('trajectories', default_keyframes)}
            future = pool.submit(jsons_to_parquet, sensor_dir, Path(sensor_dir).parent / 'trajectories.parquet',
                                 **kwargs)

        if future is not None:
            futures.append(future)
            converted_dirs.append(sensor_dir)

    for future in tqdm.tqdm(futures):
        future.result()

    if remove_converted_dirs:
        for converted_dir in converted_dirs:
            shutil.rmtree(converted_dir)


def main():
    args_parser = argparse.ArgumentParser()
    args_parser.add_argument('dataset_dir', type=str,
                             help='Path to either a single recording directory or a dataset directory that contains '
                             'an arbitrary number of recordings.')
    args_parser.add_argument('-k', '--keyframes_every', nargs='*', type=str,
                             help='Specifies to use an explicit amount of keyframes per data-type. Mappings have to '
                             'be specified as "<data-type>=<keyframes>" pair, where <data-type> is one of "camera", '
                             '"lidar", "gnss", "imu", "2d_bbs", "3d_bbs", "depth", '
                             '"sem_seg", "inst_seg", "trajectories" or "all" (which is '
                             'applied to every data-type). <keyframes> has to be an an integer > 0. '
                             'If no mapping is provided for a data-type, every frame will be a keyframe.')
    args_parser.add_argument('-r', '--remove', action='store_true',
                             help='Remove the source data directories, if entire conversion was successful.')
    args = args_parser.parse_args()

    keyframes_every = {}
    for keyframe_mapping in args.keyframes_every:
        data_type, keyframe_every = keyframe_mapping.split('=')
        keyframes_every[data_type] = int(keyframe_every)

    dataset_to_blobs(args.dataset_dir, keyframes_every, remove_converted_dirs=args.remove)


if __name__ == '__main__':
    main()
