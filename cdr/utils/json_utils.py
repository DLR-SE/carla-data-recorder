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


import json
from pathlib import Path
from typing import Any, Callable, Union

import numpy as np
from pyquaternion import Quaternion


def load_json(file_path: Union[str, Path]):
    """
    Loads the given file as JSON and returns the content.

    Args:
        path (Union[str, Path]): path to the JSON file
    """
    with open(file_path) as file:
        return json.load(file)


def save_json(obj: Any, file_path: Union[str, Path], *, default: Union[Callable[[Any], Any], None] = None):
    """
    Writes the given object to a JSON file at the given path.
    If the object contains items, that are not natively serializable to JSON, a custom function can be set, that
    maps such objects to a JSON-serializable representation.

    Args:
        obj (Any): the object to save to a JSON file
        file_path (Union[str, Path]): path to the JSON file
        default (Union[Callable[[Any], Any], None], optional): a function that maps non-serializable objects to a
            JSON-serializable representation . Defaults to None.
    """
    with open(file_path, 'wt') as file:
        json.dump(obj, file, default=default)


def array_to_json(arr: np.ndarray) -> Any:
    """
    Converts a numpy array to a JSON-serializable type.
    """
    return arr.tolist()


def quaternion_to_json(quat: Quaternion) -> Any:
    """
    Converts a Quaternion to a JSON-serializable type.
    """
    return quat.elements.tolist()


def customs_to_json(obj) -> Any:
    """
    Converts the given object to a JSON-serializable type. Supported types are:
    - numpy array
    - Quaternion

    Raises:
        TypeError: if the object's type is not supported

    Returns:
        Any: a JSON-serializable type of the given object
    """
    if isinstance(obj, np.ndarray):
        return array_to_json(obj)
    elif isinstance(obj, Quaternion):
        return quaternion_to_json(obj)
    else:
        raise TypeError(f"Object of type '{obj.__class__.__name__}' is not JSON serializable")
