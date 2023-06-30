#!/bin/bash
#SBATCH --job-name=knn
#SBATCH --nodes=3
#SBATCH --exclusive
#SBATCH --qos=debug
#SBATCH --time=01:30:00

set -a

module load python/3.10.2
module load DATACLAY/Devel3Alex
module load COMPSs/3.1
module load dislib

echo "Starting dataClay"

# Pass the environment to Python
/usr/bin/env python -u launch_dataclay.py

echo "Proceeding to \`launchcompss\`"

host_list=$(scontrol show hostname $SLURM_JOB_NODELIST | awk {' print $1 '} | sed -e 's/\.[^\ ]*//g')
export COMPSS_MASTER_NODE=$(hostname)
export COMPSS_WORKER_NODES=$(echo ${host_list} | sed -e "s/${COMPSS_MASTER_NODE}//g")

launch_compss \
        --tracing \
        --master_node="${COMPSS_MASTER_NODE}" \
        --worker_nodes="${COMPSS_WORKER_NODES}" \
        --node_memory=disabled \
        --node_storage_bandwidth=450 \
        --wall_clock_limit=9999 \
        --max_tasks_per_node=16 \
        --worker_in_master_cpus=0 \
        --summary \
        --graph \
        --lang=python \
        --pythonpath=$PWD \
                knn_app.py
