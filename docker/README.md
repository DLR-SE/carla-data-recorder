# CARLA + CARLA Data Recorder | Docker Compose Setup

## Installation

We provide a [docker compose setup](./docker-compose.yml) that uses two Docker images.

The [first image](./docker/CARLA.Dockerfile) extends the default CARLA image by
* including the Vulkan configuration of the host to be able to run headless on a server,
* installing the additional maps package and
* performing slight reconfigurations of the Unreal Engine and its start script (see [notes in main readme](../README.md#notes)).

The [second image](./docker/CDR.Dockerfile) is based on a minimal Ubuntu22.04 and additionally installs,
* some required dependencies,
* the CARLA Data Recorder and
* the ScenarioRunner (patched to include the CARLA Data Recorder).

To build the Docker images, download this repository to your machine and execute
```bash
docker compose build
```
which will build the two images named `carla-data-recorder/carla` and `carla-data-recorder/cdr`. By default, this will build CARLA 0.9.16 and a corresponding CDR image for this version, also indicated by the Docker image tags.

## Usage

To start both Docker containers, simply run:
```bash
docker compose up -d
```
If you have multiple GPUs available, you can specify the GPU of the CARLA server by
```bash
GPU_ID=3 docker compose up -d
```
which would assign the third GPU to CARLA.

If you want to disable anti-aliasing in CARLA to use Supersampling AA in the CDR, this can be enabled by setting the environment variable `DISABLE_AA=1`:
```bash
DISABLE_AA=1 docker compose up -d
```
If you want this behavior being permanent, you may also modify the `.env` file in the root of the repository.

Finally, this will start CARLA in its container running in [off-screen mode](https://carla.readthedocs.io/en/latest/adv_rendering_options/#off-screen-mode) (container name: `cdr-carla-0.9.16`), while the CDR container will be started and simply idles (container name: `cdr-cdr-0.9.16`). 

Both containers share a Docker network, so that they can communicate with each other. The CARLA container uses the alias `carla_server`, which can be used in the CDR container as `host` address to connect a `carla.Client`. If you use a custom configuration file, make sure to use `carla_server` as host address.
The containers also share a Docker volume, which is used to expose the `PythonAPI` directory from CARLA to the CDR container. This is required to provide the `agents` module to the ScenarioRunner.

Now, with both containers running, we can use the ScenarioRunner to run a scenario. In the provided patched `scenario_runner.py`, the CARLA Data Recorder can be enabled using the CLI argument `--cdr_enable`. Then, the directory where the data shall be stored must be specified via `--cdr_results_dir`.
As an example, we issue to run our example scenario using:

```bash
docker exec -it cdr-cdr-0.9.16 /bin/bash -c 'python scenario_runner.py --host carla_server --openscenario ~/carla-data-recorder/examples/01_overtaking.xosc --reloadWorld --sync --cdr_enable --cdr_results_dir /home/carla/data_recordings/new_run'
```

The ScenarioRunner will now run the scenario while recording all originating data into the specified results directory `/home/carla/data_recordings/new_run`.
We can now copy the data from the container to the host machine using:

```bash
docker cp cdr-cdr-0.9.16:/home/carla/data_recordings/new_run new_run
```
