#!/bin/bash

# Check if environment variable to disable anti aliasing is set
DISABLE_AA_VAR_NAME="DISABLE_AA"
if [[ "${!DISABLE_AA_VAR_NAME}" == "1" ]]; then
    sed -e 's/r.DefaultFeature.AntiAliasing=./r.DefaultFeature.AntiAliasing=0/g' \
      -e 's/r.Tonemapper.Sharpen=.*//g' \
      -i "$CARLA_ROOT/CarlaUE4/Config/DefaultEngine.ini"
    echo Disabled AA
fi

UE4_TRUE_SCRIPT_NAME=$(echo \"$0\" | xargs readlink -f)
UE4_PROJECT_ROOT=$(dirname "$UE4_TRUE_SCRIPT_NAME")
chmod +x "$UE4_PROJECT_ROOT/CarlaUE4/Binaries/Linux/CarlaUE4-Linux-Shipping"
"$UE4_PROJECT_ROOT/CarlaUE4/Binaries/Linux/CarlaUE4-Linux-Shipping" CarlaUE4 "$@"
