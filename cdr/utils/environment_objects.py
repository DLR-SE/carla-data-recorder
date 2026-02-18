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


import atexit
import json
import random
from abc import ABCMeta, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple, Type

import carla
import numpy as np
from typing_extensions import Self
try:  # works only with carla >= 0.9.16
    from carla.command import DestroyActor, FutureActor, SetSimulatePhysics, ApplyTransform, SpawnActor  # type: ignore
except:
    DestroyActor = carla.command.DestroyActor
    FutureActor = carla.command.FutureActor
    SetSimulatePhysics = carla.command.SetSimulatePhysics
    ApplyTransform = carla.command.ApplyTransform
    SpawnActor = carla.command.SpawnActor

from cdr.utils.carla_transformations import carla_location_to_array, carla_vector3d_to_array
from cdr.utils.carla_utils import CARLAClient, get_environment_vehicles, get_map_name
from cdr.utils.json_utils import customs_to_json


###################################
####### HANDLER DEFINITIONS #######
###################################


class EnvironmentObjectHandler(metaclass=ABCMeta):

    # Keep track of all handlers to provide singletons.
    SINGLETONS: Dict[type, 'EnvironmentObjectHandler'] = {}

    @classmethod
    def get(cls, world: carla.World) -> Tuple[Self, bool]:
        """
        Returns a singleton of this handler associated with the given world.
        If the handler does not exist yet, it is created and properly initialized.

        Args:
            world (carla.World): the current world

        Returns:
            Self: a singleton of this handler associated with the given world
        """
        is_new = False
        if cls not in EnvironmentObjectHandler.SINGLETONS.keys() or EnvironmentObjectHandler.SINGLETONS[cls].world.id != world.id:
            # Either this is the first world, or a new world was loaded
            EnvironmentObjectHandler.SINGLETONS[cls] = cls(world)
            is_new = True
        return EnvironmentObjectHandler.SINGLETONS[cls], is_new  # type:ignore

    def __init__(self, world: carla.World):
        self.world = world
        self.map_name = get_map_name(world)

        self.initialize()
        self.client: Optional[CARLAClient] = None
        self.prng_seed: Optional[int] = None
        self.handle_buffer: List[Tuple[Optional[Set[int]], bool]] = []
        """If `handle` wants to use a client but it is not set yet via `set_client`, buffer the invocations here"""

    def set_additional_parameters(self, client: CARLAClient, prng_seed: int):
        """
        Sets additional parameters to this handler, that it may require for its task.

        Args:
            client (CARLAClient): the CARLAClient to use by this handler
            prng_seed (int): a PRNG seed tp use by this handler
        """
        self.client = client
        self.prng_seed = prng_seed
        for env_obj_ids, enable in self.handle_buffer:
            self.handle(env_obj_ids, enable)
        self.handle_buffer = []

    @classmethod
    @abstractmethod
    def get_map_layer(cls) -> Optional[carla.MapLayer]:
        raise NotImplementedError()

    @abstractmethod
    def initialize(self):
        """
        Performs required initialization of this handler.
        """
        raise NotImplementedError()

    @abstractmethod
    def cleanup(self):
        """
        Cleans up the effects of this handler.
        """
        raise NotImplementedError()

    @abstractmethod
    def filter(self, env_obj_ids: Set[int]) -> Set[int]:
        """
        Returns a filtered set of the given IDs, containing only those IDs that are actually handled by this handler.

        Args:
            env_obj_ids (Set[int]): a set of environment object IDs

        Returns:
            Set[int]: a filtered set of the given IDs, containing only those IDs that are actually handled by this handler
        """
        raise NotImplementedError()

    @abstractmethod
    def handle(self, env_obj_ids: Optional[Set[int]], enable: bool):
        """
        Actually handles the given set of environment object IDs according to the dedicated logic of the specific handler.

        Args:
            env_obj_ids (Optional[Set[int]], optional): a set of IDs to handle or `None`, to handle every applicable
                object in the world.
            enable (bool): if it was requested to enable/disable the corresponding environment objects
        """
        raise NotImplementedError()


class EnvironmentVegetationHandler(EnvironmentObjectHandler):
    """
    In many CARLA maps, there is vegetation that has black-flickery material on the leaves, which is visible
    in the RGB and instance segmentation cameras. This handler tries to identify these objects using a heuristic
    and disables them.
    """

    @classmethod
    def get_map_layer(cls) -> Optional[carla.MapLayer]:
        return carla.MapLayer.Foliage

    def initialize(self):
        vegetation_objs = self.world.get_environment_objects(carla.CityObjectLabel.Vegetation)
        self.vegetation_objs_to_disable = set()
        # Problematic vegetation objects can be identified using their name:
        # - members of an InstancedFoliageActor are not affected, identifiable via _Inst_ in the object name
        #   - this sometimes includes objects like walls or stones, which are allowed -> skip such specific objects
        for veg_obj in vegetation_objs:
            if '_Inst_' not in veg_obj.name:
                if any(allow_item in veg_obj.name.lower() for allow_item in ['plane', 'stone', 'terrain', 'wall']):
                    continue
                self.vegetation_objs_to_disable.add(veg_obj.id)

        # Disable identified vegetation objects
        if len(self.vegetation_objs_to_disable) > 0:
            self.world.enable_environment_objects_original(self.vegetation_objs_to_disable, False)

    def cleanup(self):
        if len(self.vegetation_objs_to_disable) > 0:
            self.world.enable_environment_objects_original(self.vegetation_objs_to_disable, True)

    def filter(self, env_obj_ids: Set[int]) -> Set[int]:
        return env_obj_ids.intersection(self.vegetation_objs_to_disable)

    def handle(self, env_obj_ids: Optional[Set[int]], enable: bool):
        pass  # do nothing, so problematic vegetation will not be enabled again, even if enable==True

# The following constants define the mapping from two-wheeler "vehicle.*"" blueprints to their vehicle-mesh + weight
TWO_WHEELERS_BP_MESH_MAP = {
    'vehicle.harley-davidson.low_rider': ('/Game/Carla/Static/Motorcycle/Harley/SM_Harley.SM_Harley', 320),
    'vehicle.yamaha.yzf': ('/Game/Carla/Static/Motorcycle/Yamaha/SM_Yamaha.SM_Yamaha', 180),
    'vehicle.kawasaki.ninja': ('/Game/Carla/Static/Motorcycle/kawasakiNinja/SM_Kawasaki.SM_Kawasaki', 200),
    'vehicle.vespa.zx125': ('/Game/Carla/Static/Motorcycle/Vespa/SM_Vespa.SM_Vespa', 120),
    # 'vehicle.bh.crossbike': ('', 0),  # does not have a static mesh equivalent -> not placed anywhere?
    # 'vehicle.gazelle.omafiets': ('', 0),  # does not have a static mesh equivalent -> not placed anywhere?
    'vehicle.diamondback.century': ('/Game/Carla/Static/Bicycle/Roadbike/SM_RoadBike.SM_RoadBike', 8),
}

# The following constants define IDs of environment objects in CARLA, which have to be specially treated.
# - ID of an env object in Town02 (not Town02_Opt), which intersects with another vehicle by 90°
# - ID of an env object in Town10HD_Opt (not Town02_Opt), which is inside of another vehicle
REMOVE_ENV_VEHICLES = {
    'Town02': {12701692852977341302},
    'Town10HD_Opt': {7633645211419130701}}


def _get_blueprint_without_rider(bp_lib: carla.BlueprintLibrary, bp: carla.ActorBlueprint) -> carla.ActorBlueprint:
    if bp.id in TWO_WHEELERS_BP_MESH_MAP.keys():
        # For two-wheelers, we need to find the mesh of the vehicle and use static.prop.mesh
        mesh_path, mesh_weight = TWO_WHEELERS_BP_MESH_MAP[bp.id]
        prop_bp = bp_lib.find('static.prop.mesh')
        prop_bp.set_attribute('mesh_path', mesh_path)
        prop_bp.set_attribute('mass', f'{mesh_weight}')
        return prop_bp
    else:
        # Otherwise, we can simply return the actual blueprint, since they do not contain a rider
        return bp


def _generate_bp_bb_lut(world: carla.World) -> Tuple[List[str], np.ndarray]:
    bp_lib = world.get_blueprint_library()
    bps = bp_lib.filter('vehicle.*')

    bp_names = []
    bb_extents = []
    for bp in bps:
        veh_bp_id = bp.id
        bp = _get_blueprint_without_rider(bp_lib, bp)
        actor = world.spawn_actor(bp, carla.Transform(carla.Location(z=100)))
        if bp.id == 'static.prop.mesh':
            bp_names.append('::'.join([bp.id,
                                       bp.get_attribute('mesh_path').as_str(),
                                       str(bp.get_attribute('mass').as_float()),
                                       veh_bp_id]))
        else:
            bp_names.append(bp.id)
        bb_extents.append(np.array((actor.bounding_box.extent.x, actor.bounding_box.extent.y)))
        actor.destroy()

    # Some manual fixups due to some small inconsistencies between environment objects and their corresponding actor
    bb_extents[bp_names.index('vehicle.mini.cooper_s_2021')] = np.array((2.276350, 1.092643))
    bb_extents[bp_names.index('vehicle.lincoln.mkz_2020')] = np.array((2.446191, 1.115301))
    bb_extents[bp_names.index('vehicle.dodge.charger_2020')] = np.array((2.503913, 1.048542))
    bb_extents[bp_names.index('vehicle.mercedes.coupe_2020')] = np.array((2.336819, 1.001146))
    bb_extents[bp_names.index('vehicle.ford.crown')] = np.array((2.682839, 0.973231))
    # Append secondary entry for carla truck, which shall replace trucks on Town10HD, which have no actual corresponding actor
    bp_names.append('vehicle.carlamotors.carlacola')
    bb_extents.append(np.array((3.665285, 1.163165)))
    bb_extents = np.stack(bb_extents)

    return bp_names, bb_extents


def get_bp_bb_lut(world: carla.World) -> Tuple[List[str], np.ndarray]:
    filepath = Path(__file__).parent / 'blueprints_bounding_boxes_lut.json'
    if filepath.exists():
        with open(filepath, 'rt') as file:
            lut_dict = json.load(file)
            bp_names, bb_extents = lut_dict['bp_names'], lut_dict['bb_extents']
            bb_extents = np.array(bb_extents)
    else:
        bp_names, bb_extents = _generate_bp_bb_lut(world)
        # Save results for future use, so it does not have to be recomputed again
        with open(filepath, 'wt') as file:
            json.dump({'bp_names': bp_names, 'bb_extents': bb_extents},
                      file, default=customs_to_json)

    return bp_names, bb_extents


@dataclass
class EnvironmentVehicleState:

    id: int
    """ID of the EnvironmentObject"""
    name: str
    """Name of the EnvironmentObject"""
    transform: carla.Transform
    """Transform of the EnvironmentObject"""
    bounding_box: carla.BoundingBox
    """Bounding box of the EnvironmentObject"""
    env_veh_enabled: bool = True
    """Represents the enabled/disabled state of the actual EnvironmentObject in CARLA"""
    actor_substitute: Optional[int] = None
    """Actor ID that substitutes the EnvironmentObject (if substituted)"""
    compound_parent: Optional[int] = None
    """ID of the primary object in a compound EnvironmentObject (`None` if object is primary itself)"""


class EnvironmentVehicleHandler(EnvironmentObjectHandler):
    """
    Parking vehicles in a map are not Actors but EnvironmentObjects. This has multiple drawbacks:
    - All code, that has to handle vehicle actors, also has to specially handle environment vehicles, creating
      additional complexity.
    - The instance segmentation ID of environment vehicles does not correspond to the ID of the environment object
      itself, so that a direct link between the instance segmentation and the actual object cannot be established.
      - A possible workaround is to spawn instance segmentation cameras above each environment vehicle and reading the
        corresponding instance segmentation ID to explicitly create this link.
      - But there are environment vehicles, that consist of multiple individual EnvironmentObjects with individual IDs,
        making it basically impossible to find the instance segmentation ID for each part.
        - A possible workaround for this issue is to replace these EnvironmentObjects with actual (parking) actors. But
          then we have to two workarounds and still the complexity, to handle the remaining EnvironmentObjects specially.
    - The instance segmentation ID of environment vehicles can even collide with actual Actor IDs, creating ambiguities!

    All of this motivates to disable environment vehicles. But since a user may want to have these objects available,
    so that the map is not fully empty, we replace ALL environment vehicles with a corresponding actor, solving above
    problems by this one-time overhead.
    """

    def __init__(self, world: carla.World):
        super().__init__(world)
        self.bp_lib = self.world.get_blueprint_library()

    @classmethod
    def get_map_layer(cls) -> Optional[carla.MapLayer]:
        return carla.MapLayer.ParkedVehicles

    def initialize(self):
        # Remove problematic environment vehicles from the world
        if self.map_name in REMOVE_ENV_VEHICLES:
            self.world.enable_environment_objects_original(REMOVE_ENV_VEHICLES[self.map_name], False)

        # Environment vehicles are enabled by default. Due to monkeypatching of CARLA module, we get notified about
        # enable/disable calls. But if a user of the CDR would
        # - load a world and disable environment vehicles before the import of the CDR, or
        # - perform API calls regarding environment vehicles in another process,
        # this would go unnoticed. -> Documented requirements of the CDR.
        env_vehs_carla = {env_veh.id: env_veh for env_veh in get_environment_vehicles(self.world)}
        env_veh_compounds = self._find_environment_vehicle_compounds(env_vehs_carla)
        self.env_veh_states: Dict[int, EnvironmentVehicleState] = {}
        for env_veh_compound in env_veh_compounds:
            for env_veh_id in env_veh_compound:
                env_veh = env_vehs_carla[env_veh_id]
                parent = env_veh_compound[0] if env_veh_compound.index(env_veh_id) != 0 else None
                self.env_veh_states[env_veh.id] = EnvironmentVehicleState(env_veh.id, env_veh.name,
                                                                          env_veh.transform, env_veh.bounding_box,
                                                                          compound_parent=parent)

        # Get a LUT for the bounding boxes of actor blueprints
        self.bp_bb_lut = get_bp_bb_lut(self.world)

    def cleanup(self):
        if self.client is not None:
            # If we have no client, we also did not spawn any substitutes
            self.remove_substitutes(None, True)

    def _find_environment_vehicle_compounds(self, env_vehs_carla: Dict[int, carla.EnvironmentObject],
                                            distance_threshold: float = 0.1) -> List[List[int]]:
        # If there is no environment vehicle at all, return empty list
        if len(env_vehs_carla) == 0:
            return []

        # Get IDs and positions of the environment vehicles in the current world
        env_vehicle_ids = []
        env_vehicle_positions = []
        for env_veh in env_vehs_carla.values():
            if env_veh.id not in REMOVE_ENV_VEHICLES.get(self.map_name, {}):  # ignore removed vehicles
                env_vehicle_ids.append(env_veh.id)
                env_vehicle_positions.append(carla_location_to_array(env_veh.transform.location))
        env_vehicle_positions = np.stack(env_vehicle_positions, axis=0)

        # Calculate the pairwise distances (only xy) of all environment vehicles, and form compound object sets,
        # if distance is below a threshold.
        pw_dists = np.linalg.norm(env_vehicle_positions[:, None, :2] - env_vehicle_positions[None, :, :2], axis=-1)
        compound_objects: List[List[int]] = [list() for _ in range(len(env_vehicle_ids))]
        for row, col in zip(*np.where(np.triu(pw_dists < distance_threshold, k=0))):
            compound_objects[row].append(env_vehicle_ids[col])

        # Sort each compound by the size of the bounding boxes in decreasing order, assuming that the
        # primary compound part has the biggest volume, followed by the secondary parts.
        for i in range(len(compound_objects)):
            compound_ids = np.array(compound_objects[i], dtype=np.uint64)
            compound_bbs = np.stack([carla_vector3d_to_array(env_vehs_carla[obj_id].bounding_box.extent) for obj_id in compound_ids], axis=0)
            compound_bbs_volume = np.prod(compound_bbs, axis=-1)
            sort_indices = np.argsort(compound_bbs_volume)[::-1]
            compound_objects[i] = compound_ids[sort_indices].tolist()
        return compound_objects

    def _find_similar_actor_blueprint(self, env_veh_bb: carla.BoundingBox) -> carla.ActorBlueprint:
        bp_names, bb_extents = self.bp_bb_lut
        # Use np.abs() since some environment objects' bounding boxes have negative extents (???)
        env_veh_extent = np.abs(np.array((env_veh_bb.extent.x, env_veh_bb.extent.y)))
        min_idx = np.argmin(np.linalg.norm(bb_extents - env_veh_extent[None, :], axis=-1))
        bp_name = bp_names[min_idx]
        if bp_name.startswith('static.prop.mesh'):
            bp_id, mesh_path, mesh_mass, vehicle_bp_id = bp_name.split('::')
            bp = self.bp_lib.find(bp_id)
            bp.set_attribute('mesh_path', mesh_path)
            bp.set_attribute('mass', f'{mesh_mass}')
            bp.set_attribute('role_name', vehicle_bp_id)
        else:
            bp = self.bp_lib.find(bp_name)
            bp.set_attribute('role_name', 'parking')
        return bp

    def filter(self, env_obj_ids: Set[int]) -> Set[int]:
        return env_obj_ids.intersection(set(self.env_veh_states.keys()))

    def handle(self, env_obj_ids: Optional[Set[int]], enable: bool):
        if self.client is None:
            self.handle_buffer.append((env_obj_ids, enable))
            return

        if enable:
            self.spawn_substitutes(env_obj_ids)
        else:
            self.remove_substitutes(env_obj_ids)

    def spawn_substitutes(self, env_veh_ids: Optional[Set[int]] = None):
        assert self.client is not None and self.prng_seed is not None

        if env_veh_ids is None:
            env_veh_ids = set(self.env_veh_states.keys())
        else:
            env_veh_ids = self.filter(env_veh_ids)
        if len(env_veh_ids) == 0:
            return

        # Disable the environment objects
        self.world.enable_environment_objects_original(env_veh_ids, False)

        prng_generator = random.Random(self.prng_seed)

        spawn_commands = []
        for env_veh_id in env_veh_ids:
            env_veh_state = self.env_veh_states[env_veh_id]
            env_veh_state.env_veh_enabled = False  # was disabled in batched call above

            # Substitutes are only spawned for the primary part of a compound object
            if env_veh_state.compound_parent is not None:
                continue
            # If substitute is already spawned, ignore
            if env_veh_state.actor_substitute is not None:
                continue

            # Spawn new actor with corresponding blueprint below the map (to avoid collisions), then disable physics
            # and teleport the actor to the original transform of the environment vehicle.
            bp = self._find_similar_actor_blueprint(env_veh_state.bounding_box)
            if bp.has_attribute('color'):
                color = prng_generator.choice(bp.get_attribute('color').recommended_values)
                bp.set_attribute('color', color)
            command = SpawnActor(bp, carla.Transform(carla.Location(z=-100.), carla.Rotation()))\
                      .then(SetSimulatePhysics(FutureActor, False))\
                      .then(ApplyTransform(FutureActor, env_veh_state.transform))
            spawn_commands.append(command)

        responses = self.client.apply_batch_sync(spawn_commands, False)
        for env_veh_id, response in zip(env_veh_ids, responses):
            if not response.has_error():
                env_veh_state = self.env_veh_states[env_veh_id]
                env_veh_state.actor_substitute = response.actor_id

    def remove_substitutes(self, env_veh_ids: Optional[Set[int]] = None, reenable_environment_objects: bool = False):
        assert self.client is not None

        if env_veh_ids is None:
            env_veh_ids = set(self.env_veh_states.keys())
        else:
            env_veh_ids = self.filter(env_veh_ids)
        if len(env_veh_ids) == 0:
            return

        destroy_commands = []
        for env_veh_id in env_veh_ids:
            env_veh_state = self.env_veh_states[env_veh_id]
            actor_id = env_veh_state.actor_substitute
            if actor_id is not None:
                destroy_commands.append(DestroyActor(actor_id))
                env_veh_state.actor_substitute = None

            if reenable_environment_objects:
                env_veh_state.env_veh_enabled = True  # will be enabled in batched call below

        self.client.apply_batch(destroy_commands)
        # If desired, reenable the original environment objects
        if reenable_environment_objects:
            self.world.enable_environment_objects_original(env_veh_ids, True)


REGISTERED_HANDLER_TYPES: List[Type[EnvironmentObjectHandler]] = [EnvironmentVegetationHandler, EnvironmentVehicleHandler]


###################################
######### MONKEYPATCHING ##########
###################################
# We monkeypatch some CARLA methods regarding environment objects to be able to handle certain types of
# EnvironmentObjects specially. See the corresponding doc-strings for the individual handlers for details.

##### carla.World.enable_environment_objects
carla.World.enable_environment_objects_original = carla.World.enable_environment_objects  # backup original method
def enable_environment_objects_patched(self, env_objects_ids: Set[int], enable: bool):
    remaining_ids = env_objects_ids

    # Call every registered environment object handler
    for handler_type in REGISTERED_HANDLER_TYPES:
        handler, _ = handler_type.get(self)
        handled_ids = handler.filter(env_objects_ids)
        handler.handle(handled_ids, enable)
        remaining_ids = remaining_ids.difference(handled_ids)

    # Remaining environment objects are still handled by CARLA
    self.enable_environment_objects_original(remaining_ids, enable)
carla.World.enable_environment_objects = enable_environment_objects_patched

##### carla.World.unload_map_layer
carla.World.unload_map_layer_original = carla.World.unload_map_layer  # backup original method
def unload_map_layer_patched(self, map_layers: carla.MapLayer):
    for handler_type in REGISTERED_HANDLER_TYPES:
        # If the handler is associated with a map layer and it is unloaded, invoke the handler
        handler_map_layer = handler_type.get_map_layer()
        if handler_map_layer is not None and (map_layers & handler_map_layer != 0):
            handler_type.get(self)[0].handle(None, False)

    self.unload_map_layer_original(map_layers)  # now invoke the actual layer unloading
carla.World.unload_map_layer = unload_map_layer_patched

##### carla.World.load_map_layer
carla.World.load_map_layer_original = carla.World.load_map_layer  # backup original method
def load_map_layer_patched(self, map_layers: carla.MapLayer):
    # Workaround:
    # Call it twice, since there seems to be a bug in CARLA, causing reenabled environment objects to not be properly
    # tagged before being registered, causing them to be missing when calling world.get_environment_objects(label)
    # Likely origin of this problem is wrong order of statements here:
    # https://github.com/carla-simulator/carla/blob/9ce853e9917c8dc497a59e06963707f3da89a75c/Unreal/CarlaUE4/Plugins/Carla/Source/Carla/Game/CarlaGameModeBase.cpp#L932-L933
    self.load_map_layer_original(map_layers)
    self.load_map_layer_original(map_layers)

    for handler_type in REGISTERED_HANDLER_TYPES:
        # If the handler is associated with a map layer and it is loaded, invoke the handler
        handler_map_layer = handler_type.get_map_layer()
        if handler_map_layer is not None and (map_layers & handler_map_layer != 0):
            handler, _ = handler_type.get(self)
            # When a layer is loaded, previously explicitly disabled objects are all enabled again
            # -> reinitialize the handler again
            handler.initialize()
            handler.handle(None, True)
carla.World.load_map_layer = load_map_layer_patched

@atexit.register
def cleanup_environment_object_handlers():
    # Cleanup the handlers
    for handler_type in REGISTERED_HANDLER_TYPES:
        handler = EnvironmentObjectHandler.SINGLETONS.get(handler_type, None)
        if handler is not None:
            handler.cleanup()
            del EnvironmentObjectHandler.SINGLETONS[handler_type]
