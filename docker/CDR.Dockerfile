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

FROM ubuntu:22.04

# Install core dependencies
RUN packages='software-properties-common wget git python3-pip libjpeg8 libtiff5 libgeos-dev' \
    && apt-get update \
    && DEBIAN_FRONTEND=noninteractive apt-get install -y $packages \
    && update-alternatives --install /usr/bin/python python /usr/bin/python3 100

# Create "carla" user
RUN useradd -m carla
USER carla
WORKDIR /home/carla/

# Install the scenario runner
ARG CARLA_VERSION=0.9.16
RUN git clone --depth 1 --branch v${CARLA_VERSION} https://github.com/carla-simulator/scenario_runner.git && \
   cat scenario_runner/requirements.txt | sed -e "s/numpy.*//g" -e "s/opencv-python==.*/opencv-python/g" -e "s/networkx==.*/networkx/g" | xargs pip install
ENV SCENARIO_RUNNER_ROOT=/home/carla/scenario_runner/

# Install CDR's Python dependencies explicitly for better docker caching
COPY --chown=carla:carla requirements.txt /tmp/
RUN python -m pip install --upgrade pip setuptools && pip install carla==${CARLA_VERSION} && pip install -r /tmp/requirements.txt

# We expect a volume to provide the PythonAPI from CARLA to this container, as it contains the "agents" module (not available from PyPI) as well as useful scripts
ENV PYTHONPATH=/home/carla/PythonAPI/carla

# Apply our patched scenario_runner.py
COPY --chown=carla:carla patches/scenario_runner.py ${SCENARIO_RUNNER_ROOT}

# Install the CARLA Data Recorder
ENV CDR_ROOT=/home/carla/carla-data-recorder/
COPY --chown=carla:carla . ${CDR_ROOT}
# We replace "localhost" with "carla_server" in all configs, which will be the alias of the container running CARLA, when started with docker compose
RUN find . -type f -name "*.json" -exec sed -i 's/"host": "localhost"/"host": "carla_server"/g' {} +
RUN pip install ${CDR_ROOT}

WORKDIR ${SCENARIO_RUNNER_ROOT}
