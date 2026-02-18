#!/bin/bash

gpu=$1

echo '####################################'
echo '##### TrafficManager Benchmark #####'
echo '####################################'
scenarios_num_vehicles=(0 20 80 150)

for threaded in {0..1}; do
    if [ $threaded -eq 0 ]; then echo Processes:; else echo Threads:; fi
    for i in {0..7}; do
        for num_vehicles in ${scenarios_num_vehicles[@]}; do
            echo $i.json - $num_vehicles:
            config="-c "$CDR_ROOT"/benchmarking/cdr_benchmark_configs/$i.json"
            if [ $i -eq 0 ]; then config=''; fi
            CDR_THREADED=$threaded python "$CDR_ROOT"/benchmarking/cdr_benchmark.py /tmp/benchmark_data/$i $config -n $num_vehicles
        done
    done
done

echo '##################################'
echo '##### PythonAgents Benchmark #####'
echo '##################################'
scenarios_num_vehicles=(0 10 20 40)

for threaded in {0..1}; do
    if [ $threaded -eq 0 ]; then echo Processes:; else echo Threads:; fi
    for i in {0..7}; do
        for num_vehicles in ${scenarios_num_vehicles[@]}; do
            echo $i.json - $num_vehicles:
            config="-c "$CDR_ROOT"/benchmarking/cdr_benchmark_configs/$i.json"
            if [ $i -eq 0 ]; then config=''; fi
            CDR_THREADED=$threaded python "$CDR_ROOT"/benchmarking/cdr_benchmark.py /tmp/benchmark_data/$i $config -n $num_vehicles -cc
        done
    done
done
