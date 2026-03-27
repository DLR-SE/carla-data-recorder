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

import os
THREADED = os.getenv('CDR_THREADED')
THREADED = THREADED is not None and (THREADED == '1' or THREADED.lower() == 'true')

import abc
import multiprocessing
import signal
import time
import traceback
from queue import Empty
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Tuple, TYPE_CHECKING
from typing_extensions import final

from multiprocessing import Event, Queue, Value
if not THREADED or TYPE_CHECKING:
    from multiprocessing import Process
else:
    print('Explicitly running CARLA Data Recorder with threads by setting environment variable CDR_THREADED.')
    from threading import Thread as Process

from cdr.utils.carla_utils import CARLAClient


class GracefulExit(Exception):
    """
    Raised by a BaseWorker if it detects, that its execution shall gracefully stop.
    This exception shall not be raised further to the main thread.
    """
    pass


class BaseWorkerInterface(metaclass=abc.ABCMeta):

    @abc.abstractmethod
    def pre_work(self):
        """
        Implements logic that is executed previously to the call to `work`. This method is already executed
        inside the subprocess.
        """
        ...

    @abc.abstractmethod
    def work(self):
        """
        Implements the work carried out by this worker. If a `BaseWorker` is directly instantiated, it will run
        the given `target` function. In sub-classes, this method should be overridden with the actual work of the
        specialized class.
        """
        ...

    @abc.abstractmethod
    def post_work(self):
        """
        Implements logic that is executed posteriorly to the call to `work`. This method is still executed
        inside the subprocess.
        """
        ...

    @abc.abstractmethod
    def on_graceful_exit(self):
        """
        If `stop()` is called on a worker, this method will still be executed after running `work()` is interrupted.
        This may be used to release resources, hold by this worker. This method is still executed inside the subprocess.
        """
        ...

    @abc.abstractmethod
    def stop(self):
        """
        Notifies this worker to stop its execution. `post_work()` is guaranteed to still getting called.
        """
        ...

    @abc.abstractmethod
    def is_started(self) -> bool:
        """
        Returns `True` if the worker was already started (called `start`) and `False` otherwise.

        Returns:
            bool: `True` if the worker was already started (called `start`) and `False` otherwise
        """
        ...

    @abc.abstractmethod
    def should_stop(self) -> bool:
        """
        Returns `True` if the worker was signalized to stop (called `stop`) and `False` otherwise.

        Returns:
            bool: `True` if the worker was signalized to stop (called `stop`) and `False` otherwise.
        """
        ...

    @abc.abstractmethod
    def is_finished(self) -> bool:
        """
        Returns `True` if the worker already finished its work (`work` terminated) and `False` otherwise.

        Returns:
            bool: `True` if the worker already finished its work (`work` terminated) and `False` otherwise
        """
        ...

    @abc.abstractmethod
    def has_crashed(self) -> bool:
        """
        Returns `True` if the worker crashed, i.e. terminated with an exception and `False` otherwise.

        Returns:
            bool: `True` if the worker crashed, i.e. terminated with an exception and `False` otherwise.
        """
        ...

    @abc.abstractmethod
    def wait_until_finished(self) -> Optional[Tuple[Exception, str]]:
        """
        Waits for this worker to be finished, i.e. stopped executing.
        If this worker exited cleanly, `None` will be returned.
        Otherwise, the exception and a formatted stacktrace are returned.

        Returns:
            Optional[Tuple[Exception, str]]: the exception and a formatted stacktrace in case of an error, else `None`
        """
        ...

    @abc.abstractmethod
    def set_log_message_queue(self, log_message_queue: Queue):
        """
        Sets the given queue as communication channel for log messages of this worker. The queue will receive
        tuples of type (float, int, str), where the items encode the (1) original timestamp of the message,
        (2) the logging level and (3) the actual message.

        Note: This method must be called, before the worker was started. Otherwise, the queue cannot be attached to
        the worker process.

        Args:
            log_message_queue (Queue): the queue to use as communication channel for log messages of this worker
        """
        ...

    @abc.abstractmethod
    def log(self, level: int, msg: str):
        """
        Sends a log message with given level to the log queue of this worker. These messages can be read by the main
        thread and be logged.

        Args:
            level (int): level according to the `logging` module
            msg (str): the message
        """
        ...


class BaseWorker(BaseWorkerInterface, Process):

    def __init__(self, target: Optional[Callable[..., object]] = None, name: Optional[str] = None,
                 args: Iterable[Any] = (), kwargs: Mapping[str, Any] = {}):
        """
        Instantiates a `BaseWorker` object, which runs its work either as a single process or as a separate thread, if
        the environment variable `CDR_THREADED=1` is set.

        This class is designed to be used in two ways:
        1) By subclassing specialized worker implementations which can override the methods of this class, or
        2) by instantiating an object from the `BaseWorker` class and handing over `target` and required `args` and
           `kwargs`.

        Args:
            target (Optional[Callable[..., object]], optional): A callable which will be executed by the worker in case
            of the second option (see above). Defaults to None.
            name (Optional[str], optional): A name for this process/thread. If `None`, the default naming scheme for
            processes/threads will be used. Defaults to None.
            args (Iterable[Any], optional): Arguments to provide to `target`, if set. Defaults to ().
            kwargs (Mapping[str, Any], optional): Keyword arguments to provide to `target`, if set. Defaults to {}.
        """
        super().__init__(None, target, name, args, kwargs, daemon=True)

        # Some state variables
        self._is_started = False  # no need for synchronized value, since this is only set once by the parent
        self._stop_requested = Event()
        self._is_finished = Event()

        # A multiprocess queue can be set for this worker to send log messages to the main process
        self._log_message_queue: Optional[Queue] = None

        # Create multiprocess queue to send exception and formatted traceback to the main process
        self._exception_queue = Queue()
        self._has_crashed = Event()

    def start(self):
        super().start()
        self._is_started = True

    @final
    def run(self):
        """
        Entry-point of a BaseWorker. This method wraps calling `work` with some error handling.

        Do NOT override this method in sub-classes. Override `work` instead.
        """
        # Install a noop handler for SIGINT signals (KeyboardInterrupt), so that worker processes do not terminate by
        # KeyboardInterrupts issued by the user. It is expected that KeyboardInterrupts are caught (or not) by the main
        # process and the recording is gracefully terminated or killed explicitly.
        if not THREADED:
            def handler(sig, frame):
                pass
            signal.signal(signal.SIGINT, handler)

        try:
            self.pre_work()
            self.work()
            self.post_work()
            self._exception_queue.put(None)
        except GracefulExit:
            self.on_graceful_exit()
        except Exception as exc:
            tb = traceback.format_exc()
            self._exception_queue.put((exc, tb))
            self._has_crashed.set()
            self.on_graceful_exit()  # also a crashed worker should try to gracefully exit
        finally:
            self._is_finished.set()

    def pre_work(self):
        pass

    def work(self):
        super().run()

    def post_work(self):
        pass

    def on_graceful_exit(self):
        pass

    def stop(self):
        if self.is_started():
            self._stop_requested.set()

    def is_started(self) -> bool:
        return self._is_started

    def should_stop(self) -> bool:
        return self._stop_requested.is_set()

    def is_finished(self) -> bool:
        return self._is_finished.is_set()

    def has_crashed(self) -> bool:
        return self._has_crashed.is_set()

    def wait_until_finished(self) -> Optional[Tuple[Exception, str]]:
        self._is_finished.wait()
        return self._exception_queue.get()

    def set_log_message_queue(self, log_message_queue: Queue):
        self._log_message_queue = log_message_queue

    def log(self, level: int, msg: str):
        if self._log_message_queue is not None:
            timestamp = time.time()
            self._log_message_queue.put((timestamp, level, msg))


class PublisherMixin(BaseWorkerInterface):
    """
    A class designed to be used as Mixin with a `BaseWorker`. It provides functionality to `subscribe()` and to
    `publish()` data.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._publisher_queues: List[Queue] = []

    def publish(self, item: Any):
        """
        Publishes the given item to all subscribers. The item must be pickable.
        The item will not be published, if `stop()` was already called on this publisher.

        Args:
            item (Any): the item to publish to all subscribers
        """
        if not self.should_stop():
            for q in self._publisher_queues:
                q.put(item)

    def subscribe(self) -> Queue:
        """
        Subscribes a new `Queue` to this publisher, which will be returned. Data of this publisher will be published on
        this queue.

        Note: Subscription must be performed before this process is started. A violation results in a RuntimeError.

        Returns:
            Queue: a queue which will receive the data of this publisher
        """
        if self.is_started():
            raise RuntimeError('Cannot subscribe to subprocess after it already started!')

        q = Queue()
        self._publisher_queues.append(q)
        return q

    def on_graceful_exit(self):
        # Empty all publisher queues when gracefully exiting, so that they do not block the termination
        for q in self._publisher_queues:
            while q.qsize() > 0:  # qsize might not be fully reliable, but since we do not publish anymore, this is safe
                try:
                    q.get_nowait()
                except Empty:
                    pass
        super().on_graceful_exit()  # type: ignore


class ReceiverMixin(BaseWorkerInterface):
    """
    A class designed to be used as Mixin with a `BaseWorker`. It provides functionality to `receive()` data.
    """

    def __init__(self, main_queue: Queue, secondary_queues: Optional[Tuple[Queue, ...]] = None, *args, **kwargs):
        """
        Registers the given main input data queue and optionally also secondary input data queues.
        It is assumed that for each published item on the `main_queue`, there are corresponding items on every
        `secondary_queue`, i.e. having the same `frame_id`. The `secondary_queues` can run at a higher tickrate,
        then intermediate frames are simply discarded.

        Args:
            main_queue (Queue): the main input data queue
            secondary_queues (Optional[Tuple[Queue, ...]], optional): secondary input data queues. Defaults to None.
        """
        super().__init__(*args, **kwargs)
        self._main_queue = main_queue
        self._secondary_queues = secondary_queues or tuple()

    def receive(self) -> Optional[Tuple[Dict[str, Any], List[Dict[str, Any]]]]:
        """
        Waits for the next data to arrive and returns them as tuple with the first item corresponding to the
        `main_queue` and the second item being a list of the items in the `secondary_queues` in the same order as passed
        to the constructor.

        Raises:
            GracefulExit: While waiting for new data to arrive, this method regularly checks if this worker
                          `should_stop()` and raises a GracefulExit, if this is the case
            AssertionError: If no matching frames can be found in `main_queue` and `secondary_queues`

        Returns:
            main_queue_item,secondary_queue_items (Optional[Tuple[Dict[str, Any], List[Dict[str, Any]]]]):
            a tuple with the first item corresponding to the `main_queue` and the second item being a list of the items
            in the `secondary_queues` in the same order as passed to the constructor
        """
        def get_with_exit_checking(queue: Queue) -> Dict[str, Any]:
            while True:
                try:
                    return queue.get(timeout=1)
                except Empty:
                    if self.should_stop():
                        raise GracefulExit()
                except InterruptedError:
                    pass  # queue.get does not catch and retry on InterruptedError, which occurs when signal arrives

        # Get the data from the main queue
        main_queue_item = get_with_exit_checking(self._main_queue)

        if main_queue_item is None:
            # There might be still additional items in the other queues, which have to be consumed before exiting
            for secondary_queue in self._secondary_queues:
                secondary_queue_item = get_with_exit_checking(secondary_queue)
                while secondary_queue_item is not None:
                    secondary_queue_item = get_with_exit_checking(secondary_queue)
            return None

        frame_id = main_queue_item['frame_id']

        # Get the data from the secondary queues.
        # If the secondary queues are ticking faster than the main queue, skip previous frames.
        secondary_queue_items = []
        for secondary_queue in self._secondary_queues:
            secondary_queue_item = get_with_exit_checking(secondary_queue)
            while secondary_queue_item['frame_id'] < frame_id:
                secondary_queue_item = get_with_exit_checking(secondary_queue)
            secondary_queue_items.append(secondary_queue_item)

        # Last check if everything is in sync
        if len(secondary_queue_items) > 0:
            secondary_frame_ids = [secondary_queue_item['frame_id'] for secondary_queue_item in secondary_queue_items]
            if not all([frame_id == secondary_frame_id for secondary_frame_id in secondary_frame_ids]):
                raise AssertionError(f'{self.name}: {frame_id} vs {secondary_frame_ids}')  # type: ignore

        return main_queue_item, secondary_queue_items


class IterativeWorkerMixin(BaseWorkerInterface, metaclass=abc.ABCMeta):
    """
    A class designed to be used as Mixin with a `BaseWorker`. It provides functionality to iteratively call
    `self._work_iteration()` until `self._reached_end` is set to `True`.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._reached_end = False
        self._num_iterations = Value('q', 0)

    @final
    def work(self):
        """
        Iteratively calls `work_iteration` until `self._reached_end` was set to `True`.

        Do NOT override this method in sub-classes. Implement `work_iteration` instead.
        """
        while not self._reached_end:
            self._work_iteration()
            if not self._reached_end:  # do not increment for the final element, which just stops the loop
                self._num_iterations.value += 1

    @abc.abstractmethod
    def _work_iteration(self):
        """
        Implements the actual work of this worker process, which is called iteratively until `self._reached_end` was
        set to `True`. The actual end condition of this worker must be checked in the implementation of this method
        which then has to set `self._reached_end` to `True`.
        """
        raise NotImplementedError()

    def get_num_iterations(self) -> int:
        """
        Returns the amount of iterations performed by this worker.

        Returns:
            int: the amount of iterations performed by this worker
        """
        return self._num_iterations.value


class CARLAClientMixin(BaseWorkerInterface):
    """
    A class designed to be used as Mixin with a `BaseWorker`. It provides functionality to connect a CARLA client
    to a running CARLA server (`self._connect_to_carla`).
    """

    def __init__(self, carla_client: CARLAClient, requires_tick: bool, *args, **kwargs):
        """
        Registers the given `carla_client` and whether this worker requires a tick before using the CARLA
        client/world.

        Args:
            carla_client (CARLAClient): The currently used CARLA client of the CDR
            requires_tick (bool): `True`, if this worker requires a tick before using the CARLA client/world,
                                  `False` otherwise.
        """
        super().__init__(*args, **kwargs)
        self._client = carla_client
        self._requires_tick = requires_tick

        self._awaits_tick = Event()

    def _connect_to_carla(self):
        """
        Connects a new client to a running CARLA server and retrieves the current world instance.
        """
        # In threaded mode, the client is simply passed to this thread.
        # In multiprocessing mode, the client is automatically reestablished during unpickling.
        if not THREADED and multiprocessing.get_start_method() == 'fork':
            # When a process is forked, no pickling is actually happening. Instead, the client is copied in memory.
            # But the communication with the CARLA server will not work, so we reconnect explicitly in this case.
            self._client.connect()
        self._world = self._client.get_world()

        # A worker may already need access to world attributes during its setup phase.
        # But since we newly connected this client, the new world object does not have this information until
        # a tick of the server happens. Thus, we signalize the main process that we are waiting for the tick.
        if self._requires_tick:
            # world.wait_for_tick is unreliable and might miss a tick, while on_tick is called reliably
            received_tick = Event()
            callback_id = self._world.on_tick(lambda _: received_tick.set())
            self._awaits_tick.set()
            received_tick.wait()
            self._world.remove_on_tick(callback_id)
        else:
            self._awaits_tick.set()  # No need to wait, just signalize, that we completed this stage

    def wait_until_awaits_tick(self):
        """
        Waits for this worker to be waiting for a tick. This can be necessary, when the worker requires access
        to the current world state, which it only has access to, if a tick occurred after connecting its own client.
        """
        assert isinstance(self, Process) and self.is_alive()
        self._awaits_tick.wait()

    def on_graceful_exit(self):
        if hasattr(self, '_world'):
            del self._world
        if hasattr(self, '_client'):
            del self._client

        super().on_graceful_exit()  # type: ignore