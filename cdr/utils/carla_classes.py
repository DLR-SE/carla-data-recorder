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


from enum import Enum
from typing import List

import numpy as np


CITYSCAPES_ID_COLOR_MAP = {
     0: [  0,  0,  0],  # unlabeled
     1: [  0,  0,  0],  # ego vehicle
     2: [  0,  0,  0],  # rectification border
     3: [  0,  0,  0],  # out of roi
     4: [  0,  0,  0],  # static
     5: [111, 74,  0],  # dynamic
     6: [ 81,  0, 81],  # ground
     7: [128, 64,128],  # road
     8: [244, 35,232],  # sidewalk
     9: [250,170,160],  # parking
    10: [230,150,140],  # rail track
    11: [ 70, 70, 70],  # building
    12: [102,102,156],  # wall
    13: [190,153,153],  # fence
    14: [180,165,180],  # guard rail
    15: [150,100,100],  # bridge
    16: [150,120, 90],  # tunnel
    17: [153,153,153],  # pole
    18: [153,153,153],  # polegroup
    19: [250,170, 30],  # traffic light
    20: [220,220,  0],  # traffic sign
    21: [107,142, 35],  # vegetation
    22: [152,251,152],  # terrain
    23: [ 70,130,180],  # sky
    24: [220, 20, 60],  # person
    25: [255,  0,  0],  # rider
    26: [  0,  0,142],  # car
    27: [  0,  0, 70],  # truck
    28: [  0, 60,100],  # bus
    29: [  0,  0, 90],  # caravan
    30: [  0,  0,110],  # trailer
    31: [  0, 80,100],  # train
    32: [  0,  0,230],  # motorcycle
    33: [119, 11, 32],  # bicycle
    -1: [  0,  0,142],  # license plate
}

EXTRA_ID_COLOR_MAP = {
    34: [157, 234,  50],  # road line
    35: [ 45,  60, 150],  # water
    36: [ 55,  90,  80]   # other
}

OVERALL_ID_COLOR_MAP = CITYSCAPES_ID_COLOR_MAP
OVERALL_ID_COLOR_MAP.update(EXTRA_ID_COLOR_MAP)

CARLA_TO_CITYSCAPES_SEGIDS = {
     0:  0,                   # unlabeled
     1:  7,                   # road
     2:  8,                   # sidewalk
     3: 11,                   # building
     4: 12,                   # wall
     5: 13,                   # fence
     6: 17,                   # pole
     7: 19,                   # traffic light
     8: 20,                   # traffic sign
     9: 21,                   # vegetation
    10: 22,                   # terrain
    11: 23,                   # sky
    12: 24,                   # pedestrian
    13: 25,                   # rider
    14: 26,                   # car
    15: 27,                   # truck
    16: 28,                   # bus
    17: 31,                   # train
    18: 32,                   # motorcycle
    19: 33,                   # bicycle
    20:  4,                   # static
    21:  5,                   # dynamic
    22: 36,                   # other (custom)
    23: 35,                   # water (custom)
    24: 34,                   # road line (custom)
    25:  6,                   # ground
    26: 15,                   # bridge
    27: 10,                   # rail track
    28: 14                    # guard rail
}
CARLA_TO_CITYSCAPES_SEGIDS_LUT = np.zeros(max(CARLA_TO_CITYSCAPES_SEGIDS.keys()) + 1, dtype=np.uint8)
for carla_id, cityscapes_id in CARLA_TO_CITYSCAPES_SEGIDS.items():
    CARLA_TO_CITYSCAPES_SEGIDS_LUT[carla_id] = cityscapes_id

CARLA_HUMANS_SEGIDS = [12, 13]
CARLA_VEHICLES_SEGIDS = list(range(14, 20))

# Convert the color mapping to the Pillow palette layout
SEGMENTATION_PIL_PALETTE = []
for seg_id, seg_color in OVERALL_ID_COLOR_MAP.items():
    if seg_id == -1:  # all labels except "license plate" (id=-1)
        continue
    SEGMENTATION_PIL_PALETTE += seg_color


def get_car_blueprints() -> List[str]:
    """
    Returns a list of blueprint names, that represent cars

    Returns:
        List[str]: a list of blueprint names, that represent cars.
    """
    return [
        'vehicle.audi.a2',
        'vehicle.nissan.micra',
        'vehicle.audi.tt',
        'vehicle.mercedes.coupe_2020',
        'vehicle.bmw.grandtourer',
        'vehicle.micro.microlino',
        'vehicle.ford.mustang',
        'vehicle.chevrolet.impala',
        'vehicle.lincoln.mkz_2020',
        'vehicle.citroen.c3',
        'vehicle.dodge.charger_police',
        'vehicle.nissan.patrol',
        'vehicle.mini.cooper_s',
        'vehicle.mercedes.coupe',
        'vehicle.dodge.charger_2020',
        'vehicle.ford.crown',
        'vehicle.seat.leon',
        'vehicle.toyota.prius',
        'vehicle.tesla.model3',
        'vehicle.audi.etron',
        'vehicle.lincoln.mkz_2017',
        'vehicle.dodge.charger_police_2020',
        'vehicle.mini.cooper_s_2021',
        'vehicle.nissan.patrol_2021',
        'vehicle.mercedes.sprinter',
        'vehicle.tesla.cybertruck',
        'vehicle.volkswagen.t2',
        'vehicle.volkswagen.t2_2021',
        'vehicle.jeep.wrangler_rubicon'
    ]


def get_bus_blueprints() -> List[str]:
    """
    Returns a list of blueprint names, that represent busses.

    Returns:
        List[str]: a list of blueprint names, that represent busses
    """
    return [
        'vehicle.mitsubishi.fusorosa'
    ]


def get_truck_blueprints() -> List[str]:
    """
    Returns a list of blueprint names, that represent trucks.

    Returns:
        List[str]: a list of blueprint names, that represent trucks
    """
    return [
        'vehicle.ford.ambulance',
        'vehicle.carlamotors.firetruck',
        'vehicle.carlamotors.carlacola',
        'vehicle.carlamotors.european_hgv'
    ]


def get_motorcycles_blueprints() -> List[str]:
    """
    Returns a list of blueprint names, that represent motorcycles.

    Returns:
        List[str]: a list of blueprint names, that represent motorcycles
    """
    return [
        'vehicle.harley-davidson.low_rider',
        'vehicle.yamaha.yzf',
        'vehicle.kawasaki.ninja',
        'vehicle.vespa.zx125'
    ]


def get_bicycle_blueprints() -> List[str]:
    """
    Returns a list of blueprint names, that represent bicycles.

    Returns:
        List[str]: a list of blueprint names, that represent bicycles
    """
    return [
        'vehicle.bh.crossbike',
        'vehicle.gazelle.omafiets',
        'vehicle.diamondback.century'
    ]


class ActorType(Enum):
    """
    Represents a specific actor type.
    """

    PERSON = 24
    RIDER = 25
    CAR = 26
    TRUCK = 27
    BUS = 28
    MOTORCYCLE = 32
    BICYCLE = 33


def blueprint_name_to_type(blueprint_name: str) -> ActorType:
    """
    Infers the ActorType of a blueprint, based on its name.

    Args:
        name (str): the name of the blueprint

    Raises:
        ValueError: If the blueprint name is unknown.

    Returns:
        ActorType: The type of the actor
    """
    if blueprint_name in get_car_blueprints():
        return ActorType.CAR
    elif blueprint_name in get_truck_blueprints():
        return ActorType.TRUCK
    elif blueprint_name in get_bus_blueprints():
        return ActorType.BUS
    elif blueprint_name in get_motorcycles_blueprints():
        return ActorType.MOTORCYCLE
    elif blueprint_name in get_bicycle_blueprints():
        return ActorType.BICYCLE
    elif blueprint_name.startswith('walker.'):
        return ActorType.PERSON
    else:
        raise ValueError(f'Unknown blueprint name: {blueprint_name}')  # pragma: no cover
