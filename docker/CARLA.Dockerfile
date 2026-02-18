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

ARG CARLA_VERSION=0.9.16

FROM carlasim/carla:${CARLA_VERSION}
ARG CARLA_VERSION

USER root
# Fix /workspace dir to belong to carla user, so that it can write files there
RUN chown carla:carla .
# Install additional packages
RUN apt-get update && apt-get install -y wget
USER carla

# Vulkan is installed, but configuration files are missing for NVIDIA to run the container headless
# We simply copy the Vulkan configuration from the host to the image
COPY --from=vulkan --chown=root:root icd.d/nvidia_icd.json /usr/share/vulkan/icd.d/nvidia_icd.json
COPY --from=vulkan --chown=root:root implicit_layer.d/nvidia_layers.json /usr/share/vulkan/implicit_layer.d/nvidia_layers.json
COPY --from=glvnd --chown=root:root egl_vendor.d/10_nvidia.json /usr/share/glvnd/egl_vendor.d/10_nvidia.json

# Install the additional maps package
RUN wget -c https://carla-releases.b-cdn.net/Linux/AdditionalMaps_${CARLA_VERSION}.tar.gz -O AdditionalMaps_${CARLA_VERSION}.tar.gz \
  && tar -xvzf AdditionalMaps_${CARLA_VERSION}.tar.gz \
  && rm AdditionalMaps_${CARLA_VERSION}.tar.gz

# Apply a patched Unreal config, which solves some problems with texture loading in CARLA
COPY --chown=carla:carla patches/DefaultEngine.ini CarlaUE4/Config/
# Apply a patched CARLA start script, that enables to modify some configs using environment variables when starting the container
COPY --chown=carla:carla patches/CarlaUE4.sh .

# Launch CARLA in headless mode
CMD ["./CarlaUE4.sh", "-nosound", "-RenderOffScreen"]
