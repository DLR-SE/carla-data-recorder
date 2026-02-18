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


from functools import partial
import logging

import carla

from .abstract import DataListener
from cdr.utils.bounding_boxes import get_bounding_box
from cdr.utils.carla_transformations import world_snapshot_to_dict
from cdr.utils.carla_utils import CARLAClient


class WorldSnapshotDataListener(DataListener):

    def __init__(self, carla_client: CARLAClient):
        """
        Listens to WorldSnapshots, originating from a simulation tick.
        The data will be published to all subscribers of this listener.

        Args:
            carla_client (CARLAClient): The currently used CARLA client of the CDR
        """
        super().__init__(carla_client=carla_client, requires_tick=True, ticks_per_frame=1)

    def _setup(self):
        # Workaround:
        # Retrieving the bounding boxes of actors issues an RPC to the server. If this RPC happens in
        # the callback of a sensor or the world, this can cause a deadlock (not in 0.9.13 but at >0.9.15).
        # We already cache the bounding boxes in our `get_bounding_box` function to prevent this. But it can still
        # happen when issued from a callback, thus we call the method already here to cache at least all actors
        # currently existing.
        for actor in self._world.get_actors():
            get_bounding_box(actor)

        convert_func = partial(world_snapshot_to_dict, world=self._world)
        def callback(world_snapshot: carla.WorldSnapshot):
            self.log(logging.DEBUG, f'Processing frame {world_snapshot.frame}.')
            self._publish_data(world_snapshot, convert_func)

        self._callback_id = self._world.on_tick(callback)

    def post_work(self):
        self._world.remove_on_tick(self._callback_id)

    def on_graceful_exit(self):
        self.post_work()
        super().on_graceful_exit()
