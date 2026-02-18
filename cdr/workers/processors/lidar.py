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


from multiprocessing import Queue

from cdr.utils.carla_classes import CARLA_TO_CITYSCAPES_SEGIDS_LUT
from .abstract import DataProcessor


class SemanticLidarDataProcessor(DataProcessor):

    def __init__(self, semantic_lidar_queue: Queue,
                 tag_ego: bool, ego_id: int):
        """
        Processes the original CARLA semantic lidar to yield better results, by
        1) adjusting the CARLA segmentation IDs to the original cityscapes IDs (explicitly tagging ego if set),

        Args:
            semantic_lidar_queue (Queue): a queue, where a `DataListener` publishes semantic lidar data
            tag_ego (bool): whether to explicitly tag the ego vehicle (True) or treat it as regular vehicle (False)
            ego_id (int): the actor ID of the ego vehicle
        """
        super().__init__(main_queue=semantic_lidar_queue)

        self.tag_ego = tag_ego
        self.ego_id = ego_id

    def _work_iteration(self):
        received_data = self.receive()
        if received_data is None:
            self.publish(None)
            self._reached_end = True
            return
        semantic_lidar_queue_item, _ = received_data

        semseg_carla = semantic_lidar_queue_item['vertex_attributes']['semseg_id']
        actor_ids = semantic_lidar_queue_item['vertex_attributes']['actor_id']

        # Adjust the CARLA segmentation IDs to the original cityscapes IDs
        semseg_cityscapes = CARLA_TO_CITYSCAPES_SEGIDS_LUT[semseg_carla]

        # The ego vehicle has a special ID in cityscapes
        if self.tag_ego:
            semseg_cityscapes[actor_ids == self.ego_id] = 1

        semantic_lidar_queue_item['vertex_attributes']['semseg_id'] = semseg_cityscapes
        semantic_lidar_queue_item['vertex_attributes']['actor_id'] = actor_ids
        self.publish(semantic_lidar_queue_item)
