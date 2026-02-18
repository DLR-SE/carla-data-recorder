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
import os
from math import ceil
from typing import List, Tuple

import carla

from cdr import CARLADataRecorder


def get_recording_info(client: carla.Client, recording_file: str, ego_rolename: str) -> Tuple[str, int, float]:
    map_name = None
    last_create_line = None
    ego_actor_id = None
    duration = None

    rec_info: List[str] = client.show_recorder_file_info(recording_file, False).split('\n')
    # print('\n'.join(rec_info))
    for line in rec_info:
        if line.startswith('Map: '):
            map_name = line[len('Map: '):]
        elif line.startswith(' Create '):
            last_create_line = line
        elif line.startswith(f'  role_name = {ego_rolename}'):
            assert last_create_line is not None
            ego_actor_id = int(last_create_line[len(' Create '):].split(':')[0])
        elif line.startswith('Duration: '):
            duration_str, unit = line[len('Duration: '):].split(' ')
            assert unit == 'seconds'
            duration = float(duration_str)

    if map_name is None:
        raise ValueError(f'Recording "{recording_file}" is missing the "Map" property.')
    if ego_actor_id is None:
        raise ValueError(f'Recording "{recording_file}" is missing actor with role_name "{ego_rolename}".')
    if duration is None:
        raise ValueError(f'Recording "{recording_file}" is missing the "Duration" property.')
    return map_name, ego_actor_id, duration


def main():
    args_parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    # Required attributes
    args_parser.add_argument('recording_file', type=str,
                             help='Path to a recording file (.log) or a directory containing multiple recording files, '
                             '(created with the "Recorder" feature of CARLA).')
    args_parser.add_argument('output_dir', type=str,
                             help='Path to a directory, where the resulting data shall be stored.')

    # Config attributes
    args_parser.add_argument('-c', '--config', type=str, default=None,
                             help='Path to a CARLA Data Recorder configuration.')
    # If bug of deleting actors from replay in `stop_replayer` is fixed, this can be enabled (see below)
    # args_parser.add_argument('-r', '--reload_world', action='store_true',
    #                          help='Enable to reload the world for every recording.')
    args_parser.add_argument('--continue_at', type=str, default=None,
                             help='Set the name of a recording (if a directory of recordings is given), '
                             'where recording shall be continued.')

    # CARLA attributes
    args_parser.add_argument('--host', type=str, default='localhost',
                             help='Hostname of the CARLA server.')
    args_parser.add_argument('--port', type=int, default=2000,
                             help='Port of the CARLA server.')
    args_parser.add_argument('--timeout', type=float, default=20,
                             help='Timeout of the connection to the CARLA server.')

    args = args_parser.parse_args()

    if os.path.isdir(args.recording_file):
        recordings = list(sorted(os.listdir(args.recording_file)))
        if args.continue_at is not None:
            assert args.continue_at in recordings
            recordings = recordings[recordings.index(args.continue_at):]
        recordings = [os.path.join(args.recording_file, recording) for recording in recordings]
    elif os.path.isfile(args.recording_file):
        recordings = [args.recording_file]
    else:
        raise FileNotFoundError('The specified recording file or directory could not be found.')

    client = carla.Client(args.host, args.port)
    client.set_timeout(args.timeout)
    world = client.get_world()
    pre_settings = world.get_settings()
    current_map_name: str = world.get_map().name.replace('Carla/Maps/', '')

    if args.config is not None:
        cdr = CARLADataRecorder.from_config(args.output_dir, client, args.config)
    else:
        # use default config for other parameters than the CARLA Client connection
        cdr = CARLADataRecorder(args.output_dir, client, host=args.host, port=args.port, timeout=args.timeout)
    cdr.set_num_simulation_runs(len(recordings))

    for recording in recordings:
        # Inspect recording and (re)load world, if required
        rec_map_name, ego_actor_id, duration = get_recording_info(client, recording, cdr.ego_rolename)
        print(f'Running recording "{os.path.basename(recording)}".')
        # if current_map_name != rec_map_name or args.reload_world:
        if True:  # Workaround since `stop_replayer(keep_actors=False)` does not delete the actors
            print(f'(Re)loading map "{rec_map_name}"')
            client.load_world(rec_map_name)
            world = client.get_world()
            current_map_name = rec_map_name

        # Already set to sync mode, so that the replay does not immediately start
        settings = world.get_settings()
        settings.synchronous_mode = True
        settings.fixed_delta_seconds = 1.0 / cdr.simulation_fps
        world.apply_settings(settings)

        # Load the replay
        client.replay_file(recording, 0, 0, ego_actor_id, False)

        # Tick the replay while using the CDR
        try:
            cdr.set_simulation_run_name(os.path.splitext(os.path.basename(recording))[0])
            with cdr(world):
                num_frames = ceil(duration * cdr.simulation_fps)
                for _ in range(num_frames):
                    world.tick()
        except:
            print(f'Running recording "{os.path.basename(recording)}" failed. Continue with next one.')
            continue

        # Stop the replay
        # Bug: This should delete the actors from the replay, but it does not. -> Always reload world.
        client.stop_replayer(False)

    world.apply_settings(pre_settings)


if __name__ == '__main__':
    main()
