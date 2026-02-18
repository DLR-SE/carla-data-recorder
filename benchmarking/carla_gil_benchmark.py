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
import time
from functools import partial
from typing import List, Tuple

import carla
import numpy as np

THREADED = os.environ.get('THREADED', '1') == '1'
if THREADED:
    from queue import Queue
    from threading import Event, Thread as Runnable
    def spawn_sensor(world: carla.World, queue: Queue, num_ticks: int,  # type: ignore
                     host: str, port: int, timeout: float) -> carla.Sensor:
        cam_bp = world.get_blueprint_library().find('sensor.camera.rgb')
        sensor = world.spawn_actor(cam_bp, carla.Transform())
        sensor.listen(partial(read_image, queue=queue))
        return sensor
else:
    from multiprocessing import Event, Process as Runnable, Queue, Semaphore

    class SensorListener(Runnable):

        def __init__(self, queue: Queue, num_ticks: int,
                     host: str, port: int, timeout: float):
            super().__init__(daemon=True)

            self.queue = queue
            self.num_ticks = num_ticks
            self.host = host
            self.port = port
            self.timeout = timeout

            self.ticker = Semaphore(0)
            self.is_ready = Event()

        def run(self):
            client = carla.Client(self.host, self.port)
            client.set_timeout(self.timeout)
            world = client.get_world()
            cam_bp = world.get_blueprint_library().find('sensor.camera.rgb')

            sensor = world.spawn_actor(cam_bp, carla.Transform())
            def callback(image):
                read_image(image, self.queue)
                self.ticker.release()
            sensor.listen(callback)

            self.is_ready.set()  # signalize that we are ready to start
            tick = 0
            while tick < self.num_ticks:
                self.ticker.acquire()
                tick += 1

            sensor.destroy()

        def destroy(self):
            pass

    def spawn_sensor(world: carla.World, queue: Queue, num_ticks: int,
                     host: str, port: int, timeout: float) -> SensorListener:
        listener = SensorListener(queue, num_ticks, host, port, timeout)
        listener.start()
        return listener


def read_image(image, queue: Queue):
    image_arr = np.frombuffer(image.raw_data, dtype=np.uint8).reshape((image.height, image.width, 4))
    image_arr = image_arr.copy()
    image_arr = image_arr[:, :, :3]
    image_arr = image_arr[:, :, ::-1]
    queue.put(image_arr)


class Worker(Runnable):  # type: ignore

    def __init__(self, num_ops: int, num_ticks: int):
        super().__init__(daemon=True)

        self.num_ops = num_ops
        self.num_ticks = num_ticks
        self.queue = Queue()
        self.stop_running = Event()

    def run(self):
        tick = 0
        while tick < self.num_ticks:
            _ = self.queue.get()
            # Do some artificial work
            if not self.stop_running.is_set():
                dummy = 0
                for _ in range(self.num_ops):
                    dummy += 1
            tick += 1

    def stop(self):
        self.stop_running.set()


def measure_delays(world: carla.World, num_sensors: int, num_ops: int, num_ticks: int,
                   host: str, port: int, timeout: float) -> Tuple[float, float, float]:
    # Spawn sensors and workers
    sensors = []
    workers = []
    for _ in range(num_sensors):
        worker = Worker(num_ops, num_ticks)
        sensor = spawn_sensor(world, worker.queue, num_ticks, host, port, timeout)  # type: ignore
        worker.start()

        sensors.append(sensor)
        workers.append(worker)
    world.tick()

    # In the multiprocessing case, we need to wait for each process to have spawned its sensor
    for sensor in sensors:
        try:
            sensor.is_ready.wait()
        except:
            pass

    t0 = time.time()

    # Start measurement
    last_timestamp = world.get_snapshot().timestamp.platform_timestamp
    measured_delays = []
    for _ in range(num_ticks):
        # Simulate some work of the simulation script (e.g. controlling scenario/actors, etc.)
        dummy = 0
        for _ in range(100_000):  # ~3ms
            dummy += 1

        world.tick()
        timestamp = world.get_snapshot().timestamp.platform_timestamp
        delta_time = timestamp - last_timestamp
        measured_delays.append(delta_time)
        last_timestamp = timestamp
    measured_delays = np.array(measured_delays)
    elapsed_time_ticking = time.time() - t0

    # Stop workers and destroy the sensors
    for worker in workers:
        worker.stop()
    for worker in workers:
        worker.join()
    for sensor in sensors:
        sensor.destroy()

    num_drop_elems = int(0.1 * len(measured_delays))  # drop 10% of highest and lowest values (outliers)
    measured_delays = np.sort(measured_delays)[num_drop_elems:-num_drop_elems]
    return np.mean(measured_delays), np.max(measured_delays), elapsed_time_ticking  # type: ignore


def benchmark(num_sensors_min: int, num_sensors_max: int, num_sensors_step: int, num_ops_list: List[int],
              host: str, port: int, timeout: float):
    print(f'Running benchmark using {"threads" if THREADED else "processes"}')
    client = carla.Client(host, port)
    client.set_timeout(timeout)
    initial_settings = client.get_world().get_settings()
    world = client.load_world('Town10HD_Opt')

    world.apply_settings(carla.WorldSettings(synchronous_mode=True, fixed_delta_seconds=1./20.))

    # after starting CARLA, we need to perform lots of ticks until measurements are stable
    for _ in range(1000):
        world.tick()

    num_ticks = 500
    num_sensors_grid = list(range(num_sensors_min, num_sensors_max, num_sensors_step))
    num_ops_grid = num_ops_list

    mean_delays = np.zeros((len(num_sensors_grid), len(num_ops_grid)))
    max_delays = np.zeros((len(num_sensors_grid), len(num_ops_grid)))
    for num_sensors_idx, num_sensors in enumerate(num_sensors_grid):
        for num_ops_idx, num_ops in enumerate(num_ops_grid):
            print(f'[{num_sensors:>2d}, {num_ops:{max([len(str(w)) for w in num_ops_grid])}d}]', end=' -> ', flush=True)

            mean_delay, max_delay, elapsed_time_ticking = measure_delays(world, num_sensors, num_ops, num_ticks,
                                                                         host, port, timeout)
            print(f'{elapsed_time_ticking:.2f}s')
            mean_delays[num_sensors_idx, num_ops_idx] = mean_delay
            max_delays[num_sensors_idx, num_ops_idx] = max_delay

    print('num_sensors_grid:', num_sensors_grid)
    print('num_ops_grid:', num_ops_grid)
    print('mean_delays:', mean_delays, sep='\n')
    print('max_delays:', max_delays, sep='\n')

    client.get_world().apply_settings(initial_settings)


def main():
    args_parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    args_parser.add_argument('--host', type=str, default='localhost',
                             help='Hostname of the CARLA server.')
    args_parser.add_argument('--port', type=int, default=2000,
                             help='Port of the CARLA server.')
    args_parser.add_argument('--timeout', type=float, default=30,
                             help='Timeout of the connection to the CARLA server.')

    args_parser.add_argument('--num_sensors_min', type=int, default=0)
    args_parser.add_argument('--num_sensors_max', type=int, default=28)
    args_parser.add_argument('--num_sensors_step', type=int, default=2)
    args_parser.add_argument('--num_ops_list', nargs='+', type=int,
                             default=[0, 100_000, 200_000, 500_000, 1_000_000, 2_000_000, 5_000_000])

    args = args_parser.parse_args()

    benchmark(args.num_sensors_min, args.num_sensors_max, args.num_sensors_step, args.num_ops_list,
              args.host, args.port, args.timeout)


if __name__ == '__main__':
    main()
