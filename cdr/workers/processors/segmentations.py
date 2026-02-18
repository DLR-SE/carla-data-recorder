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
from typing import Tuple

import numpy as np

from .abstract import DataProcessor
from cdr.utils.carla_classes import CARLA_HUMANS_SEGIDS, CARLA_TO_CITYSCAPES_SEGIDS_LUT, CARLA_VEHICLES_SEGIDS
from cdr.utils.carla_transformations import decode_instseg_ids


class InstanceSegmentationDataProcessor(DataProcessor):

    def __init__(self,
                 instseg_queue: Queue,
                 tag_ego: bool, ego_id: int):
        """
        Processes the original CARLA instance segmentation to yield better results, by
        1) adjusting the CARLA segmentation IDs to the original cityscapes IDs (explicitly tagging ego if set),
        2) removing instance segmentation IDs for pixels that actually do not belong to actors,

        Args:
            inst_seg_queue (Queue): a queue that receives CARLA's instance segmentation
            tag_ego (bool): whether to explicitly tag the ego vehicle (True) or treat it as regular vehicle (False)
            ego_id (int): the actor ID of the ego vehicle
        """
        super().__init__(main_queue=instseg_queue)
        self.tag_ego = tag_ego
        self.ego_id = ego_id % 2**16  # since instance segmentation only supports 16-bit IDs

    def _work_iteration(self):
        received_data = self.receive()
        if received_data is None:
            self.publish(None)
            self._reached_end = True
            return
        instseg_queue_item, _ = received_data

        frame_id = instseg_queue_item['frame_id']
        instseg_array = instseg_queue_item['image_array']

        # Swap the instseg ids against the corresponding actor ids and refine the semantic segmentation
        semseg_carla = instseg_array[Ellipsis, 0]
        instseg_ids = decode_instseg_ids(instseg_array[Ellipsis, 1], instseg_array[Ellipsis, 2])

        try:
            semseg_cityscapes, actor_ids = self._semseg_instseg_refinement(semseg_carla, instseg_ids)
        except Exception as exc:
            raise RuntimeError(f'An error occurred while processing frame {frame_id}.') from exc

        # Replace the original carla instance segmentation with the refined one and put it on the output queue
        panopt_seg_queue_item = {
            'frame_id': frame_id,
            'timestamp': instseg_queue_item['timestamp'],
            'sem_seg': semseg_cityscapes,
            'inst_seg': actor_ids,
            'location': instseg_queue_item['location'],
            'rotation': instseg_queue_item['rotation']
        }
        self.publish(panopt_seg_queue_item)

    def _semseg_instseg_refinement(self, semseg_carla: np.ndarray, instseg_ids: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Refines the original semantic segmentation and instance segmentation by:
        - adjusting CARLA's custom semantic label IDs to CityScapes
        - dropping IDs in the original instance segmentation that do not belong to actual actors

        Args:
            semseg_carla (np.ndarray): the original semantic segmentation provided by CARLA (red channel)
            instseg_ids (np.ndarray): the original instance segmentation provided by CARLA (decoded green and blue channel)

        Returns:
            Tuple[np.ndarray, np.ndarray]: the refined semantic segmentation and instance segmentation
        """
        # Adjust the CARLA semantic segmentation IDs to the original cityscapes IDs
        semseg_cityscapes = CARLA_TO_CITYSCAPES_SEGIDS_LUT[semseg_carla]

        # The ego vehicle has a special ID in cityscapes
        if self.tag_ego:
            semseg_cityscapes[instseg_ids == self.ego_id] = 1

        # Copy only the actual actor IDs
        actor_ids = np.zeros_like(instseg_ids, dtype=np.uint16)
        actors_mask = np.zeros(semseg_cityscapes.shape[:2], dtype=bool)
        for semseg_carla_id in CARLA_HUMANS_SEGIDS + CARLA_VEHICLES_SEGIDS:
            actors_mask |= semseg_carla == semseg_carla_id
        actor_ids[actors_mask] = instseg_ids[actors_mask]

        if np.any(actor_ids[actors_mask] == 0):
            raise ValueError('Some pixels in frame were classified as vehicle/pedestrian by CARLA, '
                             'but their actual actor could not be determined.')
        return semseg_cityscapes, actor_ids
