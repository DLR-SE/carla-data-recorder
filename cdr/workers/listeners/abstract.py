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


import abc
import sys
from multiprocessing import Event, Value
from typing import Any, Callable, Dict

import carla

from cdr.utils.carla_utils import CARLAClient
from cdr.workers.base import BaseWorker, CARLAClientMixin, GracefulExit, PublisherMixin


class DataListener(CARLAClientMixin, PublisherMixin, BaseWorker, metaclass=abc.ABCMeta):

    def __init__(self, carla_client: CARLAClient, requires_tick: bool, ticks_per_frame: float):
        """
        A `DataListener` is a specialized `BaseWorker` responsible to receive data from the CARLA server
        and to publish it to all its subscribers.

        Args:
            carla_client (CARLAClient): The currently used CARLA client of the CDR
            requires_tick (bool): `True`, if this worker requires a tick before using the CARLA client/world,
                                  `False` otherwise.
            ticks_per_frame (float): the expected amount of world ticks that need to be executed, until this
                                     listeners receives a new frame
        """
        super().__init__(carla_client, requires_tick)
        self._frame_recording_start = Value('q', -1)
        self._frame_recording_end = Value('q', sys.maxsize)
        self._time_recording_start = Value('d', -1)

        self._is_ready = Event()

        self._last_frame_id = Value('q', -1)
        self._ticks_per_frame = ticks_per_frame
        self._check_finished = Event()

    def pre_work(self):
        self._connect_to_carla()  # may block execution until tick is received, depending on `self._requires_tick`
        self._setup()
        self._is_ready.set()

    @abc.abstractmethod
    def _setup(self):
        """
        Performs setup steps after a client is connected to the CARLA server and before this listener enters its loop
        to wait for data. The usual use case is to spawn a new sensor and register its callback or to subscribe a
        callback to the world.
        """
        raise NotImplementedError()

    def work(self):
        # The actual work of this process is carried out by a callback created in `setup`.
        # This thread will be notified either by the callback that new data was processed or by the main
        # data recorder process when setting the `frame_recording_end`. If this happens, we check if the last
        # frame was already processed and terminate, if this is the case. Otherwise, continue.
        while self._last_frame_id.value + self._ticks_per_frame <= self._frame_recording_end.value:
            while True:
                try:
                    if self._check_finished.wait(timeout=1):
                        break
                    else:
                        if self.should_stop():
                            raise GracefulExit()
                except InterruptedError:
                    pass  # wait does not catch and retry on InterruptedError, which occurs when signal arrives
            self._check_finished.clear()

        # This listener reached the last frame -> Put sentinel and stop this process
        self.publish(None)

    def _publish_data(self, data: carla.SensorData, convert_function: Callable[[carla.SensorData], Dict[str, Any]]):
        """
        Converts the given `carla.SensorData` object using the `convert_function` and publishes it to the subscribers
        of this listener. If the frame does not belong to the recording (cf. `set_recording_start` and
        `set_recording_end`), the data will not be published.

        Args:
            data (carla.SensorData): the data received from the CARLA server to publish
            convert_function (Callable[[carla.SensorData], Dict[str, Any]]): a function that transforms the
            `carla.SensorData` object into a dictionary of pickable data types
        """
        frame_recording_start = self._frame_recording_start.value
        frame_recording_end = self._frame_recording_end.value
        data_frame_id = data.frame

        # Send this frame to subscribers iff the recording has started and the frame belongs to the recording
        if frame_recording_start > 0 and frame_recording_start <= data_frame_id <= frame_recording_end:
            # CARLA objects are not pickable, thus we convert them previously
            data_dict = convert_function(data)
            data_dict['frame_id'] -= frame_recording_start
            data_dict['timestamp'] -= self._time_recording_start.value
            self.publish(data_dict)

        # Set this frame as last processed frame and notify main thread, so that it can decide if we have to stop
        self._last_frame_id.value = data_frame_id
        self._check_finished.set()

    def set_recording_start(self, frame_recording_start: int, timestamp_recording_start: float):
        """
        Sets the frame and time when the recording shall start.
        This listener will only publish data to its queue after it reached this frame.

        Args:
            frame_recording_start (int): the frame number of the start of the recording
            timestamp_recording_start (float): the timestamp of the start of the recording
        """
        self._frame_recording_start.value = frame_recording_start
        self._time_recording_start.value = timestamp_recording_start

    def set_recording_end(self, frame_recording_end: int):
        """
        Sets the frame when the recording shall stop. This listener will terminate after it reached this frame.

        Args:
            frame_recording_end (int): the frame number of the end of the recording
        """
        self._frame_recording_end.value = frame_recording_end
        self._check_finished.set()

    def wait_until_ready(self):
        """
        Waits for this listener to be "ready". A listener is called "ready", when it finished all of its required
        initialization steps and is ready to listen to data.
        """
        assert self.is_alive()
        self._is_ready.wait()
