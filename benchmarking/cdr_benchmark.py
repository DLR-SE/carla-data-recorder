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
import logging
import os
import shutil
import sys
import time
from math import ceil
from typing import Any, Dict, List, Optional, Tuple

import carla
import numpy as np
try:  # works only with carla >= 0.9.16
    from carla.command import DestroyActor, FutureActor, SetAutopilot, SpawnActor  # type: ignore
except:
    DestroyActor = carla.command.DestroyActor
    FutureActor = carla.command.FutureActor
    SetAutopilot = carla.command.SetAutopilot
    SpawnActor = carla.command.SpawnActor
from agents.navigation.basic_agent import BasicAgent  # type: ignore

from cdr import CARLADataRecorder, NoRecorder
from cdr.utils.logging_utils import CDRColoredFormatter, CDRLogger
from cdr.workers.base import THREADED


WORLD_FPS = 20


class DummyRecorder(NoRecorder):

    def __init__(self):
        # Setup logger to mimic the logging of the default CDR
        logging.Logger.manager.setLoggerClass(CDRLogger)
        self._logger: CDRLogger = logging.getLogger(f'dummy-recorder-{id(self)}')  # type: ignore
        self._logger.setLevel(logging.DEBUG)

        log_stdout_handler = logging.StreamHandler(sys.stdout)
        log_stdout_handler.setLevel(logging.INFO)
        log_stdout_handler.setFormatter(CDRColoredFormatter())

        self._logger.addHandler(log_stdout_handler)

        self._logger.propagate = False

        # Some variables to track progress
        self.simulation_fps = WORLD_FPS
        self.sensor_config_path = ''

    def _parse_sensor_configuration(self, _) -> Tuple[Any, Dict[str, str]]:
        return None, {'carla_optimal_vehicle': 'vehicle.citroen.c3'}

    def __enter__(self):
        self._logger.info(f'Recording initialization finished (took 0.000s). Start scenario.')

    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type is None:
            self._logger.info('Simulation of the scenario finished at frame ? !')

def generate_traffic(client: carla.Client, ego_vehicle_bp_name: str, num_other_vehicles: int,
                     tm_port: Optional[int]) -> List[carla.Actor]:
    world = client.get_world()
    carla_map = world.get_map()
    waypoints = carla_map.generate_waypoints(5)
    waypoints = list(filter(lambda wp: not wp.is_junction, waypoints))  # filter junctions, since vehicles can collide here

    spawn_commands = []
    vehicles = []

    bp_lib = world.get_blueprint_library()
    # Spawn ego vehicle
    ego_bp = bp_lib.find(ego_vehicle_bp_name)
    ego_bp.set_attribute('role_name', 'hero')
    ego_transform = waypoints.pop(np.random.randint(len(waypoints))).transform
    ego_transform.location += carla.Location(z=1.)
    spawn_command = SpawnActor(ego_bp, ego_transform)
    if tm_port is not None:
        spawn_command = spawn_command.then(SetAutopilot(FutureActor, True, tm_port))
    spawn_commands.append(spawn_command)

    # Spawn other vehicles
    vehicle_bps = bp_lib.filter('vehicle.*')
    vehicle_bps = np.random.choice(vehicle_bps, num_other_vehicles)
    for vehicle_bp in vehicle_bps:
        vehicle_transform = waypoints.pop(np.random.randint(len(waypoints))).transform
        vehicle_transform.location += carla.Location(z=1.)
        spawn_command = SpawnActor(vehicle_bp, vehicle_transform)
        if tm_port is not None:
            spawn_command = spawn_command.then(SetAutopilot(FutureActor, True, tm_port))
        spawn_commands.append(spawn_command)

    num_failed_spawns = 0
    for response in client.apply_batch_sync(spawn_commands):
        if not response.error:
            actor = world.get_actor(response.actor_id)
            vehicles.append(actor)
        else:
            num_failed_spawns +=1
    if num_failed_spawns > 0:
        print(f'Failed to spawn {num_failed_spawns} vehicles.')
    return vehicles


def main():
    args_parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    # Required attributes
    args_parser.add_argument('output_dir', type=str,
                             help='Path to a directory, where the resulting data shall be stored.')

    # Scenario attributes
    args_parser.add_argument('-m', '--map', type=str, default='Town10HD_Opt',
                             help='The map to run the scenarios on.')
    args_parser.add_argument('-n', '--num_vehicles', type=int, nargs='+', default=[0, 20, 80, 150],
                             help='The number of other vehicles per scenario.')
    args_parser.add_argument('-r', '--repetitions', type=int, default=10,
                             help='The number of repetitions per scenario.')
    args_parser.add_argument('-d', '--duration', type=float, default=60.,
                             help='The duration of the random scenarios in seconds.')
    args_parser.add_argument('-cc', '--client_control', action='store_true',
                             help='Enables using client-controlled vehicles, instead of using the TrafficManager.')
    args_parser.add_argument('--seed', type=int, default=42,
                             help='Random seed to control randomness.')

    # Config attributes
    args_parser.add_argument('-c', '--config', type=str, default=None,
                             help='Path to a CARLA Data Recorder configuration.')

    # CARLA attributes
    args_parser.add_argument('--host', type=str, default='localhost',
                             help='Hostname of the CARLA server.')
    args_parser.add_argument('--port', type=int, default=2000,
                             help='Port of the CARLA server.')
    args_parser.add_argument('--tm_port', type=int, default=8000,
                             help='Port of the CARLA traffic manager.')
    args_parser.add_argument('--timeout', type=int, default=60,
                             help='Timeout of the connection to the CARLA server.')

    args = args_parser.parse_args()

    seed = args.seed
    np.random.seed(seed)
    print(f'PRNG seed is "{seed}".')

    client = carla.Client(args.host, args.port)
    client.set_timeout(args.timeout)
    world = client.get_world()
    pre_settings = world.get_settings()

    root_dir: str = args.output_dir
    map_name: str = args.map
    nums_vehicles: List[int] = args.num_vehicles
    num_repetitions: int = args.repetitions
    duration_scenarios: float = args.duration

    # Load the map and TM
    carla_map = world.get_map()
    current_map_name: str = carla_map.name.replace('Carla/Maps/', '')
    if current_map_name != map_name:
        client.load_world(map_name)
        world = client.get_world()
        carla_map = world.get_map()

    traffic_manager = client.get_trafficmanager(args.tm_port)
    traffic_manager.set_random_device_seed(np.random.randint(2**31-1))

    # Set to sync mode
    settings = world.get_settings()
    settings.synchronous_mode = True
    settings.fixed_delta_seconds = 1.0 / WORLD_FPS
    world.apply_settings(settings)
    traffic_manager.set_synchronous_mode(True)

    # Create CDR
    map_dir = os.path.join(root_dir, map_name)
    if os.path.isdir(map_dir):  # for safety, if old data survived due to some crash
        shutil.rmtree(map_dir, ignore_errors=True)
    if args.config is not None:
        data_recorder = CARLADataRecorder.from_config(map_dir, client, args.config)
    else:
        data_recorder = DummyRecorder()
    data_recorder.set_num_simulation_runs(len(nums_vehicles) * num_repetitions)

    # Get optimal ego vehicle
    ego_optimal_vehicle = data_recorder._parse_sensor_configuration(data_recorder.sensor_config_path)[1]['carla_optimal_vehicle']

    total_durations: Dict[int, List[float]] = {}
    durations_between_ticks: Dict[int, List[List[float]]] = {}
    for num_vehicles in nums_vehicles:
        total_durations[num_vehicles] = []
        durations_between_ticks[num_vehicles] = []
        for i in range(num_repetitions):
            # Generate ego vehicle and random traffic
            actors = generate_traffic(client, ego_optimal_vehicle, num_vehicles,
                                      tm_port=args.tm_port if not args.client_control else None)

            if not args.client_control:
                # TrafficManager is enabled for vehicles, so simply tick the world
                def run_scenario(num_ticks: int) -> List[float]:
                    post_tick_times = []
                    for _ in range(num_ticks):
                        world.tick()
                        post_tick_times.append(time.time())
                    return post_tick_times
            else:
                # Create agents for every spawned vehicle and call the when ticking the world
                actors_w_agents = {}
                world.tick()  # actors are not really spawned yet, without a tick, the route cannot be determined
                for actor in actors:
                    actors_w_agents[actor] = BasicAgent(actor, map_inst=carla_map)
                def run_scenario(num_ticks: int) -> List[float]:
                    post_tick_times = []
                    for _ in range(num_ticks):
                        for actor, agent in actors_w_agents.items():
                            actor.apply_control(agent.run_step())
                        world.tick()
                        post_tick_times.append(time.time())
                    return post_tick_times

            # Tick the scenario for some time, to get a flowing start
            run_scenario(120)

            # Tick the scenario as long as desired while recording data with the CDR
            recording_name = f'{num_vehicles}-{i:03d}'
            try:
                data_recorder.set_simulation_run_name(recording_name)
                t_pre_recording = time.time()
                with data_recorder(world):
                    num_frames = ceil(duration_scenarios * data_recorder.simulation_fps)
                    post_tick_times = run_scenario(num_frames)
                t_post_recording = time.time()
                total_duration = t_post_recording - t_pre_recording
                total_durations[num_vehicles].append(total_duration)
                print(f'Recording {recording_name} took {total_duration:.4f}s.')

                # Calculate durations between ticks
                post_tick_times = np.array(post_tick_times)
                durations_between_ticks[num_vehicles].append((post_tick_times[1:] - post_tick_times[:-1]).tolist())
            except:
                print(f'Running recording "{recording_name}" failed. Reload world, then continue with next one.')
                client.reload_world(reset_settings=False)
                world = client.get_world()
                continue

            # Cleanup
            destroy_commands = [DestroyActor(actor) for actor in actors]
            client.apply_batch_sync(destroy_commands)

    print('###################################')
    print(f'Benchmark Summary')
    print(f'|- Map               : {map_name}')
    print(f'|- Scenario duration : {duration_scenarios}')
    print(f'|- Num vehicles      : {nums_vehicles}')
    print(f'|- Num repetitions   : {num_repetitions}')
    print(f'|- Config            : {args.config}')
    print(f'|- Setting           : {"PythonAgents" if args.client_control else "TrafficManager"}')
    print(f'|- CDR_THREADED      : {THREADED}')
    print()
    print('Total durations:')
    print(f'{total_durations}')
    print('Durations between ticks:')
    print(f'{durations_between_ticks}')
    print('###################################')

    world.apply_settings(pre_settings)
    traffic_manager.set_synchronous_mode(pre_settings.synchronous_mode)

    # Delete the data, since we only wrote it for benchmarking purposes
    shutil.rmtree(map_dir, ignore_errors=True)


if __name__ == '__main__':
    main()
