#!/bin/bash

gpu=$1
scenarios_num_vehicles=(0 10 20 40)

echo Threads:
for i in {0..7}; do
    docker run --rm -itd --gpus '"device='$gpu'"' -w /home/carla/carla_data_recorder --name cdr-benchmark-$gpu-t-$i degr_th/carla_data_recorder:ue4-dev > /dev/null
    sleep 3
    for num_vehicles in ${scenarios_num_vehicles[@]}; do
        echo $i.json - $num_vehicles:
        if [ $i -eq 0 ]; then
            docker exec -it cdr-benchmark-$gpu-t-$i /bin/bash -c 'export CDR_THREADED=1; python "$CDR_ROOT"/tools/wait_for_carla.py; python -u -X faulthandler "$CDR_ROOT"/benchmarking/cdr_benchmark.py /tmp/benchmark_data/'$i' -n '$num_vehicles' -cc'
        else
            docker exec -it cdr-benchmark-$gpu-t-$i /bin/bash -c 'export CDR_THREADED=1; python "$CDR_ROOT"/tools/wait_for_carla.py; python -u -X faulthandler "$CDR_ROOT"/benchmarking/cdr_benchmark.py /tmp/benchmark_data/'$i' -c "$CDR_ROOT"/benchmarking/cdr_benchmark_configs/'$i'.json -n '$num_vehicles' -cc'
        fi
    done
    docker stop cdr-benchmark-$gpu-t-$i > /dev/null
    sleep 1
done

echo Processes:
for i in {1..7}; do
    docker run --rm -itd --gpus '"device='$gpu'"' -w /home/carla/carla_data_recorder --name cdr-benchmark-$gpu-p-$i degr_th/carla_data_recorder:ue4-dev > /dev/null
    sleep 3
    for num_vehicles in ${scenarios_num_vehicles[@]}; do
        echo $i.json - $num_vehicles:
        if [ $i -eq 0 ]; then
            docker exec -it cdr-benchmark-$gpu-p-$i /bin/bash -c 'export CDR_THREADED=0; python "$CDR_ROOT"/tools/wait_for_carla.py; python -u -X faulthandler "$CDR_ROOT"/benchmarking/cdr_benchmark.py /tmp/benchmark_data/'$i' -n '$num_vehicles' -cc'
        else
            docker exec -it cdr-benchmark-$gpu-p-$i /bin/bash -c 'export CDR_THREADED=0; python "$CDR_ROOT"/tools/wait_for_carla.py; python -u -X faulthandler "$CDR_ROOT"/benchmarking/cdr_benchmark.py /tmp/benchmark_data/'$i' -c "$CDR_ROOT"/benchmarking/cdr_benchmark_configs/'$i'.json -n '$num_vehicles' -cc'
        fi
    done
    docker stop cdr-benchmark-$gpu-p-$i > /dev/null
    sleep 1
done
