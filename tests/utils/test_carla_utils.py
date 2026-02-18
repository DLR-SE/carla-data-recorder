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


import pickle
from unittest import mock

from cdr.utils.carla_utils import CARLAClient


def test_carla_client_pickling():
    with mock.patch('cdr.utils.carla_utils.CARLAClient.connect') as connect_mock:
        cc = CARLAClient()
        connect_mock.assert_called_once()
        connect_mock.reset_mock()
        cc = pickle.loads(pickle.dumps(cc))
        connect_mock.assert_called_once()
