#!/bin/bash

for i in {1..10}; do
    echo Run iteration $i

    # Threads
    THREADED=1 python "$CDR_ROOT"/benchmarking/carla_gil_benchmark.py

    # Processes
    THREADED=0 python "$CDR_ROOT"/benchmarking/carla_gil_benchmark.py
done
