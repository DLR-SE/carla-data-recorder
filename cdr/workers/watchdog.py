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


import time
import _thread
from multiprocessing import Queue, Event
from threading import Thread
from typing import List

from cdr.utils.logging_utils import CDRLogger
from cdr.workers.base import BaseWorker


class Watchdog(Thread):

    def __init__(self, logger: CDRLogger, check_interval: float = 0.2):
        """
        Creates a new watchdog thread, capable of monitoring a set of workers, which can be added to this watchdog via
        `register`. This monitoring currently includes the following tasks:
        1. Watching for new log messages of the workers, which then will be logged to the given logger.
        2. Observing the status of the works, whether they crashed due to an exception. If this happens, the watchdog
        will interrupt the main thread, signaling that the simulation should be stopped.

        Args:
            logger (CDRLogger): the main application logger
            check_interval (float, optional): the time interval (in seconds) specifying how often this watchdog performs
            its tasks. Defaults to 0.2.
        """
        super().__init__(name='CDR-Watchdog', daemon=True)

        self._logger = logger
        self._check_interval = check_interval

        self._workers: List[BaseWorker] = []
        self._log_message_queue = Queue()
        self._worker_crashed = Event()

        self._running = False

    def register(self, *workers: BaseWorker):
        """
        Registers the given workers to this watchdog.

        Note: This method must be called, before the worker was started. Otherwise, the watchdog cannot listen to
        log messages of the worker process.
        """
        for worker in workers:
            self._workers.append(worker)
            worker.set_log_message_queue(self._log_message_queue)

    def start(self):
        self._running = True
        super().start()

    def run(self):
        while self._running:
            time_wakeup = time.time()

            # Check if there are any log messages that we can log to the overall application logger
            num_log_messages = self._log_message_queue.qsize()
            for _ in range(num_log_messages):
                timestamp, log_level, message = self._log_message_queue.get()
                self._logger.log(log_level, message, extra={'timestamp': timestamp})

            # Check if any worker crashed (if we did not yet)
            if not self.has_observed_crash():
                for worker in self._workers:
                    if worker.has_crashed():
                        exc, tb = worker.wait_until_finished()  # type: ignore  <- Must be set, if has_crashed
                        self._logger.error(f'An error occurred in {worker.name}. -> Interrupt simulation.\n{exc}\n{tb}')
                        self._worker_crashed.set()
                        try:
                            _thread.interrupt_main()
                        except KeyboardInterrupt:
                            pass  # we ignore the interrupt in this thread...
                        finally:
                            break  # ... and break out of the loop to only interrupt once

            # Sleep (if necessary)
            time_to_sleep = self._check_interval - (time.time() - time_wakeup)
            if self._running and time_to_sleep > 0:
                time.sleep(time_to_sleep)

    def stop(self):
        """
        Signals the watchdog thread to stop.
        """
        self._running = False

    def has_observed_crash(self) -> bool:
        """
        Returns `True` if the watchdog observed an exception raised in a worker process, `False` otherwise.

        Returns:
            bool: `True` if the watchdog observed an exception raised in a worker process, `False` otherwise
        """
        return self._worker_crashed.is_set()
