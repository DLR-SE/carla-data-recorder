# Dataset Format

The resulting directory structure and data formats of the individual files are explained in the following.

- [Directory Structure](#directory-structure)
- [Data Types and Formats](#data-types-and-formats)
    - [Sensors](#sensors)
        - [Camera](#camera)
        - [Lidar](#lidar)
        - [GNSS](#gnss)
        - [IMU](#imu)
    - [Ground Truth Annotations](#ground-truth-annotations)
        - [2D Bounding Boxes](#2d-bounding-boxes)
        - [3D Bounding Boxes](#3d-bounding-boxes)
        - [Depth](#depth)
        - [Semantic Segmentation](#semantic-segmentation)
        - [Instance Segmentation](#instance-segmentation)
        - [Semantic Lidar](#semantic-lidar)
        - [Trajectories](#trajectories)

## Directory Structure

For each single recorded simulation, the following directory structure will be created that holds the data for the different sensors and ground truth annotations.

```
|
|-- metadata.json  # information about the simulation run (map, weather parameters, num simulation ticks, num frames)
|-- sensor_configuration.json  # information about the sensors and their configurations
|-- ground_truth
    |-- 2d_bbs  # 2D bounding boxes for every sensor
        |-- ...
    |-- 3d_bbs  # 3D bounding boxes for every sensor
        |-- ...
    |-- depth  # depth for every sensor
        |-- ...
    |-- inst_seg  # instance segmentation for every sensor
        |-- ...
    |-- sem_lidar  # semantic lidar for every sensor
        |-- ...
    |-- sem_seg  # semantic segmentation for every sensor
        |-- ...
    |-- trajectories  # trajectories with velocities and accelerations for every actor in the world
        |-- ...
|-- sensor
    |-- camera
        |-- ...
    |-- gnss
        |-- ...
    |-- imu
        |-- ...
    |-- lidar
        |-- ...
```

The CARLA Data Recorder supports various data formats for the different data types. These are configured using the `recording_outputs` parameter in the CDR configuration.
* If monolithic file formats are selected (e.g. MP4-Videos for camera image sequences or Parquet-Files for other data types), these are stored in the above specified leaf-directories of their corresponding data type, where the file names correspond to the sensor IDs of the used sensor configuration.
  * For Parquet files, the CDR stores an additional Parquet file with suffix `_index` next to the data file, which maps `frame_id` to `row_group`. This can be used for fast lookup of the row group that has to be read from the data file that holds the desired frame, when randomly accessing frames.
* If distributed file formats are selected (e.g. individual JPG or PNG images), an additional directory layer is added in the above specified leaf-directories, where the directory names correspond to the sensor IDs of the used sensor configuration.
  * The filenames of the individual sensor-related files denote the ID of the frame, as reported by CARLA. They can be used to associate the different files with each other. If the actual timestamp of a frame is required, it can be looked up in the trajectories information (see [Trajectories](#trajectories)) or manually inferred from the tick rate of the sensor/world.

## Data Types and Formats

### Sensors

#### Camera

The camera sensors create individual images per frame, which are either stored individually (`png`, `jpg`) or combined (`mp4`).

* **key**: `CAMERA:RGB`
* **format:**
  * `jpg` (default)
  * `png`
  * `mp4`
* **format_parameters:**
  * `keyframe_every` (int, default=1): Specifies the frequency of keyframes in the video. Lower values increase decoding speed when randomly accessing frames in the video, typically at the cost of larger file sizes. < *`mp4` only*

##### Individual Files (JPG, PNG)

An image file can easily be loaded with ImageIO via:

```python
import imageio.v3 as iio

frame = iio.imread('path/to/image')
```

##### Combined File (MP4)

Additionally to the image data stored in the video, the MP4-metadata's comment contains all frame IDs as a comma-separated string. Reading a specific frame by ID from a video can thus be accomplished by:

```python
import imageio.v3 as iio

frame_id = 123
with iio.imopen('path/to/video.mp4', 'r', plugin='pyav') as reader:
    # Read the CSV-encoded frame_ids from the comment of the container
    frame_ids = reader.container_metadata['comment'].split(',')
    frame_index = frame_ids.index(str(frame_id))
    frame = reader.read(index=frame_index)
```

#### Lidar

The lidar sensors create point clouds with X,Y,Z positions + intensity of each target hit, which are either stored individually (`ply`) or combined (`parquet`).

* **key**: `LIDAR:XYZI`
* **format:**
  * `ply` (default)
  * `parquet`
* **format_parameters:**
  * `keyframe_every` (int, default=1): Specifies the row group size in Parquet files. Lower values increase decoding speed when randomly accessing frames in the Parquet file, typically at the cost of larger file sizes. < *`parquet` only*

##### Individual Files (PLY)

Point clouds stored as PLY file can easily be loaded with the lightweight `trimesh` library via:

```python
from trimesh.exchange.ply import load_ply

with open('path/to/file.ply', 'rb') as file:
    xyzi = load_ply(file)['metadata']['_ply_raw']['vertex']['data']
    # xyzi is a structured NumPy array with columns:
    # x, y, z, intensity
```

##### Combined File (Parquet)

Point clouds stored in a Parquet file are encoded using the Draco library, for a good compression/performance tradeoff.
The Parquet scheme has the columns: `frame_id`, `pointcloud_draco_bytes`.
A point cloud can be loaded and decoded via:

```python
import pandas as pd
import DracoPy

frame_id = 123
df = pd.read_parquet('path/to/file.parquet')
pc = DracoPy.decode(df.loc[frame_id]['pointcloud_draco_bytes'])
xyz = pc.points
intensity = pc.get_attribute_by_name('intensity')['data']
```

#### GNSS

Latitude, longitude and altitude measured by GNSS sensors are stored for every tick of the simulation, including the corresponding timestamp. The reported values are relative to the geo-reference of the CARLA map.

* **key**: `GNSS`
* **format:**
  * `json` (default)
  * `parquet`
* **format_parameters:**
  * `keyframe_every` (int, default=1): Specifies the row group size in Parquet files. Lower values increase decoding speed when randomly accessing frames in the Parquet file, typically at the cost of larger file sizes. < *`parquet` only*

##### Individual Files (JSON)

GNSS data in a JSON file is stored as a dictionary with the following structure, which can be directly loaded via Python's `json` module.

```json
{
    "timestamp": 8.692344621863413e-10,
    "altitude": -0.05305445194244385,    // the altitude [meters]
    "latitude": -0.0013818080414722544,  // the latitude [°]
    "longitude": 8.745345336884815e-05   // the longitude [°]
}
```

##### Combined File (Parquet)

GNSS data in a Parquet file is stored as a simple table and can easily be read via pandas.
The Parquet scheme has the columns: `frame_id`, `timestamp`, `altitude`, `latitude`, `longitude`, with the same meaning and units as the JSON schema.

#### IMU

Accelerations and angular velocities measured by IMU sensors are stored for every tick of the simulation, including the corresponding timestamp.

* **key**: `IMU`
* **format:**
  * `json` (default)
  * `parquet`
* **format_parameters:**
  * `keyframe_every` (int, default=1): Specifies the row group size in Parquet files. Lower values increase decoding speed when randomly accessing frames in the Parquet file, typically at the cost of larger file sizes. < *`parquet` only*

##### Individual Files (JSON)

IMU data in a JSON file is stored as a dictionary with the following structure, which can be directly loaded via Python's `json` module.

```json
{
    "timestamp": 8.692344621863413e-10,
    "acceleration": [                    // the acceleration (x,y,z) in the sensor's reference frame [meters/second^2]
        -0.2842060327529907,
        1.233566403388977,
        9.872149467468262
    ],
    "angular_velocity": [                // the angular velocity (x,y,z) in the sensor's reference frame [radians/second]
        -0.0008405359112657607,
        0.0005320724449120462,
        -0.06337276101112366
    ]
}
```

##### Combined File (Parquet)

IMU data in a Parquet file is stored as a simple table and can easily be read via pandas.
The Parquet scheme has the columns: `frame_id`, `timestamp`, `acceleration`, `angular_velocity`, with the same meaning and units as the JSON schema.

### Ground Truth Annotations

#### 2D Bounding Boxes

2D bounding boxes are stored for every tick of each camera sensor. They are filtered based on visibility in the image.

* **key**: `CAMERA:BB2D`
* **format:**
  * `json` (default)
  * `parquet`
* **format_parameters:**
  * `keyframe_every` (int, default=1): Specifies the row group size in Parquet files. Lower values increase decoding speed when randomly accessing frames in the Parquet file, typically at the cost of larger file sizes. < *`parquet` only*

##### Individual Files (JSON)

2D bounding box data in a JSON file is stored as a dictionary that maps actor IDs to a set of attributes. It uses the following structure, which can be directly loaded via Python's `json` module (be aware, that JSON stores the keys (actor IDs) as string, not as int!).

```json
{
    "2511": {                                    // actor ID
        "class_id": "car",                       // label as in CityScapes
        "c_x": 736.493899784827,                 // the horizontal midpoint [pixels]
        "c_y": 473.2811669626476,                // the vertical midpoint [pixels]
        "w": 49.277735593836496,                 // the overall width [pixels]
        "h": 33.73778913968209,                  // the overall height [pixels]
        "type_id": "vehicle.bmw.grandtourer",    // the CARLA blueprint name of the actor
        "role_name": "background"                // the role name of the actor
    }
}
```

*Note:* The `role_name` of environment vehicles (parking vehicles in a map) is always reported as "parking".

##### Combined File (Parquet)

2D bounding box data in a Parquet file are stored as a nested dictionary.
The Parquet scheme has the columns: `frame_id`, `actors`, where actors is a mapping identical to the JSON schema.

#### 3D Bounding Boxes

3D bounding boxes are stored for every tick of each camera sensor. They are filtered based on visibility in the image.

* **key**: `CAMERA:BB3D`
* **format:**
  * `json` (default)
  * `parquet`
* **format_parameters:**
  * `keyframe_every` (int, default=1): Specifies the row group size in Parquet files. Lower values increase decoding speed when randomly accessing frames in the Parquet file, typically at the cost of larger file sizes. < *`parquet` only*

##### Individual Files (JSON)

3D bounding box data in a JSON file is stored as a dictionary that maps actor IDs to a set of attributes. It uses the following structure, which can be directly loaded via Python's `json` module (be aware, that JSON stores the keys (actor IDs) as string, not as int!).

```json
{
    "2511": {                                    // actor ID
        "class_id": "car",                       // label as in CityScapes
        "center": [                              // the center point (x,y,z) in world coordinates [meters]
            -7.363250255584717,
            -119.24063110351562,
            1.0486390590667725
        ],
        "size": [                                // the length, width, height [meters]
            4.611005783081055,
            2.241713285446167,
            1.6672759056091309
        ],
        "rot": [                                 // the orientation as quaternion (w,x,y,z) in world coordinates
            0.7063115086346321,
            0.0,
            0.0,
            -0.7079011603114307
        ],
        "velocity": [                            // the velocity (x,y,z) in world coordinates [meters/second]
            -4.488067872898682e-08,
            -1.1511712472156432e-07,
            0.017194384709000587
        ],
        "acceleration": [                        // the acceleration (x,y,z) in world coordinates [meters/second^2]
            -8.847708272696764e-07,
            -3.1618124012311455e-06,
            0.1350652575492859
        ],
        "angular_velocity": [                    // the angular velocity (x,y,z) in world coordinates [radians/second]
            -1.7477592336945236e-05,
            -1.4797206858929712e-05,
            1.899147328288109e-08
        ],
        "type_id": "vehicle.bmw.grandtourer",    // the CARLA blueprint name of the actor
        "role_name": "background"                // the role name of the actor
    }
}
```

*Note:* The `role_name` of environment vehicles (parking vehicles in a map) is always reported as "parking".

##### Combined File (Parquet)

3D bounding box data in a Parquet file are stored as a nested dictionary.
The Parquet scheme has the columns: `frame_id`, `actors`, where actors is a mapping identical to the JSON schema.

#### Depth

Depth data is stored for every tick of every camera sensor. The depth of every pixels corresponds to the actual metric depth of the geometry at that pixel to the camera (as used in depth estimation). This is different to the z-depth, which is normally exported by CARLA.

* **key**: `CAMERA:DEPTH`
* **format:**
  * `exr` (default)
  * `png`
  * `parquet`
* **format_parameters:**
  * `keyframe_every` (int, default=1): Specifies the row group size in Parquet files. Lower values increase decoding speed when randomly accessing frames in the Parquet file, typically at the cost of larger file sizes. < *`parquet` only*

##### Individual Files (EXR)

The depth is stored absolutely in meters when using the `.exr` format using the lossy PXR24 compression. This compression is a very good compromise between filesize and information loss. As nice property, the deviations (from the compression) grow with the distance to the camera, i.e. are very small for near points and get bigger for far-away points (due to the quantization from 32 to 24 bits). Our experiments support this, where we measured the following deviations:
| Distance [m] | Deviation [mm] |
| ------------ | -------------- |
| <= 10        | <= 0.122       |
| <= 20        | <= 0.244       |
| <= 40        | <= 0.488       |
| <= 80        | <= 0.977       |
| <= 128       | <= 1.0         |
| <= 278       | <= 1.95        |
| <= 512       | <= 4.          |

EXR files can be loaded in Python using ImageIO (with freeimage plugin) with the following snippet:

```python
import imageio
imageio.plugins.freeimage.download()  # required for EXR
import imageio.v3 as iio

depth_in_meters = iio.imread('path/to/file.exr')
```

##### Individual Files (PNG)

If depth maps are stored as `.png`, then the depth is stored as single-channel 16-bit PNG files, where the pixel values encode the depth in centimeters. Thus, the maximum possible depth is clamped to `2^16-1 = 655.35m`. In this case, the depth can be loaded via:

```python
import numpy as np
from PIL import Image
depth_in_meters = np.array(Image.open('path/to/file.png')) / 100
```

##### Combined File (Parquet)

Depth stored in a Parquet file is encoded using as PXR24 compressed EXR blob, for a good compression/performance tradeoff. This is basically identical to the individually stored EXR files (see [Individual Files (EXR)](#individual-files-exr)).
The Parquet scheme has the columns: `frame_id`, `depth_meters_exr`.
A depth image can be loaded and decoded via:

```python
import imageio.v3 as iio
import pandas as pd

frame_id = 123
df = pd.read_parquet('path/to/file.parquet')
encoded = df.loc[frame_id]['depth_meters_exr']
depth_in_meters = iio.imread(encoded, extension='.exr')
```

#### Semantic Segmentation

The semantic segmentation images use the CityScapes labels as defined in [labels.py](https://github.com/mcordts/cityscapesScripts/blob/master/cityscapesscripts/helpers/labels.py) and augments these labels by three additional classes:

| name      | id  | color           |
| --------- | --- | --------------- |
| road line | 34  | (157, 234,  50) |
| water     | 35  | ( 45,  60, 150) |
| other     | 36  | ( 55,  90,  80) |

*Note*: The semantic segmentation stores the `id`, not the `trainId`! If you want to use the `trainId`, it has to be mapped accordingly.

* **key**: `CAMERA:SEMSEG`
* **format:**
  * `png` (default)
  * `parquet`
* **format_parameters:**
  * `keyframe_every` (int, default=1): Specifies the row group size in Parquet files. Lower values increase decoding speed when randomly accessing frames in the Parquet file, typically at the cost of larger file sizes. < *`parquet` only*

##### Individual Files (PNG)

In order to accurately store the `id`, while 1. saving disk space and 2. making it easy to visually inspect the segmentations, we store them as indexed single-channel 8-bit PNGs, where the color palette reflects the CityScapes colors. Thus, image viewers display the images using the well-known color scheme, while the actual `id`-array can be loaded via:

```python
import numpy as np
from PIL import Image
semseg = np.array(Image.open('path/to/file.png'))
```

The only drawback is that it is not possible to load the `id`-array using OpenCV, since it applies the color palette to the image, when reading it. We recommend to use PIL or ImageIO instead.

##### Combined File (Parquet)

Semantic segmentation stored in a Parquet file is stored as an array blob.
The Parquet scheme has the columns: `frame_id`, `shape`, `dtype` and `array_bytes`.
A semantic segmentation image can be loaded via:

```python
import imageio.v3 as iio
import pandas as pd

frame_id = 123
df = pd.read_parquet('path/to/file.parquet')
row = df.loc[frame_id]
semseg = np.frombuffer(row['array_bytes'], dtype=row['dtype']).reshape(row['shape'])
```

#### Instance Segmentation

The actor IDs in the instance segmentation are stored as 16-bit images, thus enabling to differentiate $2^{16}=65536$ actors (which CARLA cannot simulate simultaneously). The encoded IDs directly refer to the IDs stored in bounding box annotations or the trajectories, i.e. they can be used also a tracking IDs.

* **key**: `CAMERA:INSTSEG`
* **format:**
  * `png` (default)
  * `parquet`
* **format_parameters:**
  * `keyframe_every` (int, default=1): Specifies the row group size in Parquet files. Lower values increase decoding speed when randomly accessing frames in the Parquet file, typically at the cost of larger file sizes. < *`parquet` only*

##### Individual Files (PNG)

If instance segmentation images are stored as `.png`, then it is stored as single-channel 16-bit PNGs.
The per-pixel actor IDs can easily be loaded via:

```python
import numpy as np
from PIL import Image
instseg = np.array(Image.open('path/to/file.png'))
```

##### Combined File (Parquet)

Instance segmentation stored in a Parquet file is stored as an array blob.
The Parquet scheme has the columns: `frame_id`, `shape`, `dtype` and `array_bytes`.
An instance segmentation image can be loaded via:

```python
import imageio.v3 as iio
import pandas as pd

frame_id = 123
df = pd.read_parquet('path/to/file.parquet')
row = df.loc[frame_id]
instseg = np.frombuffer(row['array_bytes'], dtype=row['dtype']).reshape(row['shape'])
```

#### Semantic Lidar

The semantic lidar sensors create point clouds with X,Y,Z positions + the cosine of the incident angle + actor IDs + semantic IDs of each target hit. They are either stored individually (`ply`) or combined (`parquet`).

* **key**: `LIDAR:SEMANTIC`
* **format:**
  * `ply` (default)
  * `parquet`
* **format_parameters:**
  * `keyframe_every` (int, default=1): Specifies the row group size in Parquet files. Lower values increase decoding speed when randomly accessing frames in the Parquet file, typically at the cost of larger file sizes. < *`parquet` only*

##### Individual Files (PLY)

Point clouds stored as PLY file can easily be loaded with the lightweight `trimesh` library via:

```python
from trimesh.exchange.ply import load_ply

with open('path/to/file.ply', 'rb') as file:
    xyzcas = load_ply(file)['metadata']['_ply_raw']['vertex']['data']
    # xyzcas is a structured NumPy array with columns:
    # x, y, z, cosine_incident_angle, actor_id, semseg_id
```

##### Combined File (Parquet)

Point clouds stored in a Parquet file are encoded using the Draco library, for a good compression/performance tradeoff.
The Parquet scheme has the columns: `frame_id`, `pointcloud_draco_bytes`.
A point cloud can be loaded and decoded via:

```python
import pandas as pd
import DracoPy

frame_id = 123
df = pd.read_parquet('path/to/file.parquet')
pc = DracoPy.decode(df.loc[frame_id]['pointcloud_draco_bytes'])
xyz = pc.points
cosine_incident_angles = pc.get_attribute_by_name('cosine_incident_angle')['data']
actor_ids = pc.get_attribute_by_name('actor_id')['data']
semseg_ids = pc.get_attribute_by_name('semseg_id')['data']
```

#### Trajectories

The trajectories of all actors in the world are stored for every tick of the simulation, i.e. potentially at a higher frequency than other sensors. Each file contains the corresponding timestamp and information about every actor, which has the same attributes as [3D Bounding Boxes](#3d).

* **key**: `TRAJECTORIES`
* **format:**
  * `json` (default)
  * `parquet`
* **format_parameters:**
  * `keyframe_every` (int, default=1): Specifies the row group size in Parquet files. Lower values increase decoding speed when randomly accessing frames in the Parquet file, typically at the cost of larger file sizes. < *`parquet` only*

##### Individual Files (JSON)

Trajectory data in a JSON file is stored as a dictionary that contains the `timestamp` and a nested dictionary `actors` that maps actor IDs to a set of attributes. It uses the following structure, which can be directly loaded via Python's `json` module (be aware, that JSON stores the keys (actor IDs) as string, not as int!).

<details>
    <summary><b>Expand example</b></summary>

```json
{
    "timestamp": 5.034544445574284,
    "actors": {
        "2466": {
            "class_id": "bicycle",
            "center": [
                37.5545768737793,
                -209.21054077148438,
                1.2159111499786377
            ],
            "size": [
                1.659999966621399,
                0.5,
                1.8200000524520874
            ],
            "rot": [
                0.9999967281814588,
                -1.0966588418769709e-07,
                0.0025576919184199516,
                4.287675329113571e-05
            ],
            "velocity": [
                0.0,
                0.0,
                0.0
            ],
            "acceleration": [
                0.0,
                0.0,
                0.0
            ],
            "angular_velocity": [
                0.0,
                0.0,
                0.0
            ],
            "type_id": "vehicle.diamondback.century",
            "role_name": "scenario"
        },
        "2447": {
            "class_id": "car",
            "center": [
                -3.456117630004883,
                -179.2355194091797,
                0.972186803817749
            ],
            "size": [
                4.901683330535889,
                2.128324270248413,
                1.5107464790344238
            ],
            "rot": [
                0.7070270257414433,
                0.0,
                0.0,
                0.7071865276369512
            ],
            "velocity": [
                0.0,
                0.0,
                0.0
            ],
            "acceleration": [
                0.0,
                0.0,
                0.0
            ],
            "angular_velocity": [
                0.0,
                0.0,
                0.0
            ],
            "type_id": "vehicle.lincoln.mkz_2017",
            "role_name": "hero"
        },
        "2511": {
            "class_id": "car",
            "center": [
                -7.363250255584717,
                -119.24063110351562,
                1.1504039764404297
            ],
            "size": [
                4.611005783081055,
                2.241713285446167,
                1.6672759056091309
            ],
            "rot": [
                0.7063115086346321,
                0.0,
                0.0,
                -0.7079011603114307
            ],
            "velocity": [
                -0.0,
                0.0,
                -0.4899999797344208
            ],
            "acceleration": [
                -0.0,
                0.0,
                -9.799999237060547
            ],
            "angular_velocity": [
                0.0,
                0.0,
                0.0
            ],
            "type_id": "vehicle.bmw.grandtourer",
            "role_name": "background"
        }
    }
}
```
</details>

##### Combined File (Parquet)

Trajectories data in a Parquet file is stored as a nested dictionary.
The Parquet scheme has the columns: `frame_id`, `timestamp`, `actors`, where actors is a mapping identical to the JSON schema.
