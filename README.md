# CARLA Data Recorder

The **CARLA Data Recorder (CDR)** is a tool that enables the recording of structured datasets based on simulation runs in the simulator CARLA. It is compatible with the latest builds of the UE4 version of CARLA (≥0.9.16).

## Features
- Fully parallelized data listening, processing and writing, resulting in a high throughput without blocking the simulator
- Clone of the [nuScenes sensor setup](https://www.nuscenes.org/nuscenes#data-collection) (cameras + lidar + GPS + IMU)
- Supported sensor types
  - Camera
  - Lidar
  - GNSS
  - IMU
- Supported ground truth annotations
  - Depth
    - Instead of writing the z-buffer (as CARLA does), the actual metric depth to each pixel is computed (as expected for depth estimation)
  - Semantic segmentation
  - Instance segmentation
    - Our refined instance segmentation also provides actor IDs for environment vehicles, which are "random" by default (see below)
  - 3D Bounding Boxes and 2D Bounding Boxes
    - Only actors (vehicles, pedestrians) are included, which are actually visible in the corresponding image
  - Semantic lidar
    - Our refined semantic lidar also provides actor IDs for environment vehicles, which are 0 by default (see below)
  - Trajectories
    - Full trajectories with velocities and accelerations for every actor in the world (i.e. no filtering based on visibility)
- Writing of data either as single files per frame (images, JSON, ...) or as monolithic files (videos, Parquet)
- Handling of "problematic" environment objects
  - Vegetation
    - In many CARLA maps, there is vegetation that has black-flickery material on the leaves, which is visible in the RGB and instance segmentation cameras.
    - The CDR detects these objects using a heuristic and disables them.
  - Environment vehicles
    - Each map contains environment vehicles (parking), so that it does not look empty. These vehicles are not actual `carla.Actor`s, so that they need to be processed differently.
    - This brings some problems, so that the CDR substitutes every such environment vehicle with an actual `carla.Actor`, unifying all logic related to vehicles (whether they are usual dynamic actors or "parking").
    - Please read [Usage->Notes->Environment Vehicles](#environment-vehicles) to ensure, that this works properly!

The CARLA Data Recorder does not steer the simulation on its own (e.g. spawning actors, steering the ego vehicle, ticking the world). Instead, it is embedded into a Python script, which is responsible for the actual simulation.
This repository provides a patched version of the [ScenarioRunner for CARLA](https://github.com/carla-simulator/scenario_runner), having the CARLA Data Recorder patched in. Installation und usage instructions for this combination are described in [Installation](#installation) and [Usage](#usage). In case you want to connect your own simulation script to the CARLA Data Recorder instead of using the ScenarioRunner, look [here](#connect-custom-simulation-script-to-carla-data-recorder).

## Installation

We provide a docker compose setup that connects a CARLA container with a dedicated container that hosts the CARLA Data Recorder. Installation and Usage instructions can be found [here](./docker/README.md).

For a manual installation of the CARLA Data Recorder, do the following:

1) Download the latest CARLA version
2) (Optional) Install CARLA ScenarioRunner (use corresponding tag for your CARLA version), if needed
3) Install the CARLA Data Recorder by
   1) cloning this repository
   2) installing it via pip: `pip install .` (inside the root of the cloned repository)
4) (Optional) Patch CARLA ScenarioRunner: Copy `patches/scenario_runner.py` to the root directory of the CARLA ScenarioRunner installation
5) (Optional) Patch Unreal Engine: Copy `patches/DefaultEngine.ini` to `CarlaUE4/Configs/` in the root directory of the CARLA installation
6) (Optional) Patch CARLA start script: Copy `patches/CARLAUE4.sh` to the root directory of the CARLA installation
   1) Should only be done when using SSAA feature of the CDR, instead of CARLA's default TAA

### Notes

- The updated Unreal Engine configuration modifies the engine's texture loading pipeline. Otherwise, blurry low-res textures can lead to unsatisfying data, which especially occur at the beginning of the data recording. Using this modified configuration should eliminate such frames.
  - Additionally, the temporal anti aliasing (TAA) is slightly tweaked, to reduce ghosting artifacts on moving objects.
- The start script of CARLA is modified, to allow for disabling the anti aliasing in CARLA before starting it. This can be enabled by setting the environment variable `DISABLE_AA=1` prior to starting CARLA.
  - This might be desirable, when opting to use the SSAA feature of the CDR, instead of CARLA's default TAA.

## Usage

### Integrate CARLA Data Recorder into custom simulation script

The interface of the CARLA Data Recorder is designed to be easy to integrate into your own simulation script. After the installation (see [Installation-Manual](#Manual)), you can import the `CARLADataRecorder` via `from cdr import CARLADataRecorder`. This class exposes five methods intended for public usage: `start_recording`, `stop_recording`, `__call__`, `set_num_simulation_runs` and `set_simulation_run_name`.
* `start_recording` should be called **after** you prepared your simulation (e.g. load map, spawn vehicles, ...) and **before** your actual simulation loop starts.
* `stop_recording` should be called **after** you simulation loop stopped and **before** you start cleaning up your simulation.
* `set_num_simulation_runs` is optional and should be called, when your simulation script runs multiple individual simulation runs sequentially. In this case, the recorder splits the individual recordings into different directories, stored inside the specified `results_dir`.
  * If this method is not called, the CARLA Data Recorder assumes that one simulation run will be executed and stored directly in `results_dir`.
  * If this method is called, each simulation run can be assigned a unique name via `set_simulation_run_name` to be easily identifiable on disk.

For a convenient integration into your simulation script, `CARLADataRecorder` can be used as a context manager, taking care of calling `start_recording` and `stop_recording`, while additionally providing some error handling. Thus, it is advised to use it this way, if possible. In any case, the CARLA Data Recorder needs access to the current `carla.World` instance used by the simulation script. When using above functions explicitly, the world can be provided as argument to `start_recording`. When using the CDR as context manager, it has to be provided previously to entering the context manager via `__call__`, i.e. calling the CDR object itself with the world (see example below).

The following code snippet illustrates the integration of the CARLA Data Recorder as a context manager:

```python
import carla
from cdr import CARLADataRecorder

client = carla.Client()
world = client.get_world()

recorder = CARLADataRecorder('<path/to/results/dir', client)  # you can pass additional arguments for configuration
# or
recorder = CARLADataRecorder.from_config('<path/to/results/dir', client, '<path/to/config>')  # if you use a custom config file

def run(scenario):
    ... # prepare your simulation (e.g. load map, spawn vehicles, ...)
    with recorder(world):
        ...  # here comes your simulation loop (you must use world.tick(), not world.wait_for_tick())
    ...  # clean up your simulation

def run_multiple(scenarios):
    num_scenarios = len(scenarios)
    recorder.set_num_simulation_runs(num_scenarios)

    for scenario in scenarios:
        recorder.set_simulation_run_name(scenario.name)
        run(scenario)
```

### Use CARLA Data Recorder to gather data from a CARLA recording (.log file)

CARLA provides an integrated [Recorder](https://carla.readthedocs.io/en/latest/adv_recorder/), which stores all required events during a simulation run to replay it again in CARLA. It is also possible to modify certain parts of the simulation when replaying a recording (e.g. the weather).
This repository provides a tool [recording_to_data](./tools/recording_to_data.py), which takes either a single recording or a directory of multiple recordings as input, and uses the CARLA Data Recorder to generate the data based on either the default or a specified recorder configuration.
It can be called from the command line via `python recording_to_data.py [options] <recording_file> <output_dir>`. Call `python recording_to_data.py -h` to see the help for the arguments.

Currently, this tool does not support gathering data for variations of a single recording (e.g. by modifying the weather). Such features might be added in the future but should also be easy to integrate if required.

### Configurations

If some configuration of the CARLA Data Recorder is required (e.g. ego's rolename, sensor configuration, data output types), the path to a configuration file can be provided via `--cdr_config`. Some exemplary configurations can be found at [here](./recorder_configs/). These configurations are also part of the CDR container (under `/home/carla/carla-data-recorder/recorder_configs/`), where the `host` `localhost` is substituted by `carla_server` during the build of the image. You can also copy/bind mount your own configuration files into the container.

The following tables specifies all available parameters of the CARLA Data Recorder. In the JSON config, the types have to be replaced with the corresponding JSON type (mainly: `' -> "` (for strings), `None -> null`, `(tuple_elements, ...) -> [tuple_elements, ...]`).
| name                              | type                      | default                   | description                                                                                                                                                                                                                                                                                                           |
| --------------------------------- | ------------------------- | ------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| host                              | str                       | 'localhost'               | IP or hostname running the CARLA server.                                                                                                                                                                                                                                                                              |
| port                              | int                       | 2000                      | TCP port of the CARLA server.                                                                                                                                                                                                                                                                                         |
| timeout                           | float                     | 10.                       | Timeout for CARLA client connecting to the server.                                                                                                                                                                                                                                                                    |
| ego_rolename                      | str                       | 'hero'                    | The rolename of the ego vehicle.                                                                                                                                                                                                                                                                                      |
| simulation_fps                    | int                       | 20                        | The frame rate of the simulation, i.e. number of ticks per second.                                                                                                                                                                                                                                                    |
| sensor_config                     | str                       | 'nuscenes'                | Either the name of a builtin sensor configuration or the path to the desired sensor configuration file (absolute or relative to recorder config file). The list of available builtin sensor configurations can be retrieved by via `list_available_sensor_configs`.                                                   |
| static_sensor_config              | Optional[str]             | None                      | Path to configuration file for static sensors (absolute or relative to recorder config file).                                                                                                                                                                                                                         |
| recording_outputs                 | Dict[str, Dict[str, Any]] | DEFAULT_RECORDING_OUTPUTS | Specifies the demanded outputs for every sensor type, as a mapping from the sensor output type identifier to the desired output format (key: "format") and optionally additional format parameters (key: "format_parameters"). See the data format specification for more details.                                    |
| delay_recording_seconds           | float                     | 0.                        | Delay the recording for the specified amount of seconds.                                                                                                                                                                                                                                                              |
| delay_recording_ticks             | int                       | 0                         | Delay the recording for the specified amount of world ticks.                                                                                                                                                                                                                                                          |
| enable_ssaa                       | bool                      | False                     | Enables Supersampling Anti Aliasing (SSAA).                                                                                                                                                                                                                                                                           |
| ssaa_factor                       | float                     | 2.                        | Sets the scaling factor to use for SSAA.                                                                                                                                                                                                                                                                              |
| annotations_traverse_translucency | Optional[bool]            | None                      | Enables/Disables annotations like depth or semantic and instance segmentation to traverse translucent materials. If `None`, this setting is kept as currently set. This feature requires CARLA >=0.9.16.                                                                                                              |
| prng_seed                         | int                       | 42                        | Sets a master PRNG seed, that controls randomness across the entire CDR.                                                                                                                                                                                                                                              |
| raise_exceptions                  | bool                      | True                      | If `True`, exceptions will be raised, otherwise they will only be logged.                                                                                                                                                                                                                                             |
| override_existing_data            | bool                      | False                     | If `True`, existing data in the directory of a new simulation run would be overridden, otherwise the recording will be rejected.                                                                                                                                                                                      |
| delete_on_error                   | bool                      | True                      | If `True`, a data recording will be deleted when an error occurs, otherwise the data will be kept, likely resulting in inconsistencies between sensors (e.g. missing data).                                                                                                                                           |
| allow_user_interrupt              | bool                      | False                     | If `True` and this object is used as context manager, a KeyboardInterrupt that leads to an exit of the context manager is allowed to stop the recording, otherwise the recording is killed. Depending on `delete_on_error`` this might either delete the data of the recording or likely result in inconsistent data. |
| verbose                           | bool                      | False                     | Enables verbose output to stdout. The logfile will always contain every logged message, independent of this setting.                                                                                                                                                                                                  |

### Notes

- The CARLA Data Recorder sets the simulator to the synchronous mode. Nevertheless, the ticking of the simulator must take place in your simulation script (i.e. use `world.tick()` not `world.wait_for_tick()`). This way, heavy computations in the simulation script can take their time.
  - When using the ScenarioRunner, you must use the `--sync` flag.
  - It also is the responsibility of the simulation script to call `traffic_manager.set_synchronous_mode(True)`, when a Traffic Manager is used (as stated in the docs: https://carla.readthedocs.io/en/latest/adv_traffic_manager/#synchronous-mode).
- The CARLA Data Recorder can also run entirely using threads instead of processes by setting the environment variable `CDR_THREADED=1`. This will slow down CARLA and the CDR but can be helpful in debugging.
- There are some know issues, which are documented [below](#known-issues)

#### Environment Vehicles

- Environment vehicles (parking vehicles in a map) introduce many problems:
  - The instance segmentation ID of environment vehicles does not correspond to the ID of the environment object itself, so that a direct link between the instance segmentation and the actual object cannot be established.
  - This problem is amplified by some environment vehicles, that consist of multiple individual `carla.EnvironmentObject`s with individual IDs, making it basically impossible to find the instance segmentation ID for each part.
  - The instance segmentation ID of environment vehicles can even collide with actual actor IDs, creating ambiguities!
- All of this motivates to disable environment vehicles. But since a user may want to have these objects available, so that the map is not fully empty, the CDR replaces ALL environment vehicles with a corresponding actor, solving the above problems.
  - The overhead of this operation is relatively small, except for Town15, which contains over 1100(!) environment vehicles.
    - To reduce this overhead here, you can disable all or maybe randomly some of these environment vehicles via CARLA's API (`world.enable_environment_objects(ids, False)`).
  - The color of the substitutes is randomized, but this randomness is controllable via the `prng_seed` parameter of the CDR, making it fully deterministic.
  - Since a user may want to control environment objects on its own, and the substitutes created by the CDR must reflect those calls, the CDR monkey-patches the following methods of the CARLA API: `world.enable_environment_objects`, `world.load_map_layer`, `world.unload_map_layer`.
    - It is essential, that
      - the CDR is imported, before you perform any call to one of these functions, otherwise the actual state in CARLA and the internal state of the CDR deviate,
      - non of these functions is called in any other Python process than the one, that imports the CDR.
    - This is due to the unavailability of CARLA to actually determine, whether an environment object is disabled or not. So this state has to be tracked on the client-side.

### Known Issues

- Spawning new actors directly in front of a camera leads to a couple of frames, where the instance segmentation is wrong
  - This is not related to our code but has to be a bug in CARLA (at least up to 0.9.16), where newly spawned actors become visible in the instance segmentation a couple of frames after they are already visible in the default RGB camera
- When using the `recording_to_data.py` tool with multiple recording files, we sometimes observed that the ego vehicle cannot be found in some recording files.
  - Calling the tool a second time, it suddenly works. Restarting CARLA, it does not work again. The underlying issue is still unknown.

## Dataset Format

The resulting structure for individual recordings as well as the used formats for different data types are described [here](./docs/dataset_format.md).

-----

# Authors

CARLA Data Recorder is developed by the German Aerospace Center (DLR) Institute Systems Engineering for Future Mobility (SE). See [AUTHORS](./AUTHORS) for more details.

# Acknowledgement

If you find this software useful for your own work, you are encouraged to acknowledge the authors of this work in your publications.

# Legal

Any non-bundled dependency of CARLA Data Recorder comes with its own license. By installing the dependencies, it is your responsibility to ensure your compliance with the respective licensing terms. See also [NOTICE](./NOTICE).

# Licensing

CARLA Data Recorder is licensed under the Apache License, Version 2.0. See [LICENSE](./LICENSE) for the full license text.
The file `patches/scenario_runner.py` is a modified version of the original [ScenarioRunner for CARLA](https://github.com/carla-simulator/scenario_runner), developed
by Intel Corporation, licensed under the MIT License. The modifications to this file are licensed under the Apache License, Version 2.0.
