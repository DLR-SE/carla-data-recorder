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
import sys

import carla


def main():
    args_parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    args_parser.add_argument('--host', type=str, default='localhost',
                             help='Hostname of the CARLA server.')
    args_parser.add_argument('--port', type=int, default=2000,
                             help='Port of the CARLA server.')
    args_parser.add_argument('--timeout', type=float, default=5,
                             help='Timeout of the connection to the CARLA server.')
    args_parser.add_argument('-d', '--duration', type=float, default=60,
                             help='The maximum duration to wait for the CARLA server.')
    args = args_parser.parse_args()

    num_repetitions = int(args.duration / args.timeout)
    # increase the timeout if duration and timeout are not evenly divisible
    timeout = args.timeout + (args.duration - num_repetitions * args.timeout) / num_repetitions
    for _ in range(num_repetitions):
        try:
            client = carla.Client(args.host, args.port)
            client.set_timeout(timeout)
            client.get_world()  # times out with RuntimeError, if CARLA server does not respond in time
            sys.exit(0)
        except RuntimeError:
            pass
    sys.exit(1)


if __name__ == '__main__':
    main()
