"""This file should be part of MareNostrum deployment.

But there is some version mismatch, and some fragile parameters here and there.
So, for the time being, it lives here. You can disregard it entirely, as it is
not relevant for the demo.
"""

import os
import os.path
import shutil
import subprocess
from time import sleep

from fabric import Connection, SerialGroup

DATACLAY_LOG_ROOT = os.path.expandvars("$HOME/.dataclay/$SLURM_JOB_ID/logs")
DATACLAY_STORAGE_ROOT = os.path.expandvars("$HOME/.dataclay/$SLURM_JOB_ID/storage")
PYTHON_BINARY = shutil.which("python")
REDIS_SERVER = os.path.expandvars("$DATACLAY_HOME/bin/redis-server")
REDIS_LOG = os.path.join(DATACLAY_LOG_ROOT, "redis.log")
METADATA_LOG = os.path.join(DATACLAY_LOG_ROOT, "metadata.log")
BACKEND_LOG_TEMPLATE = os.path.join(DATACLAY_LOG_ROOT, "backend-core%d_%02d.log")

# Get the value of the SLURM_JOB_NODELIST environment variable
node_list = os.environ.get('SLURM_JOB_NODELIST')

if node_list is None:
    raise RuntimeError("No SLURM_JOB_NODELIST envvar found. Aborting")

print("SLURM_JOB_NODELIST contains: %s" % node_list)

# Use scontrol to expand the node list into individual nodes
command = f"scontrol show hostname {node_list}"
process = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
stdout, stderr = process.communicate()

# Parse the stdout to extract the node names
nodes = stdout.decode().split()
print("Parsed into: %s" % nodes)

master_c = Connection(nodes[0])
backends_g = SerialGroup(*nodes[1:])

# Prepare folder (assuming shared filesystem)
master_c.local(f"mkdir -p {DATACLAY_LOG_ROOT}")
master_c.local(f"mkdir -p {DATACLAY_STORAGE_ROOT}")

# Remove the previous dump... a smarted thing may be done, but for now, this works
master_c.local(f"rm -f {os.path.expanduser('~/dump.rdb')}")

extra_env = {
    "DATACLAY_METADATA_HOST": nodes[0],
    "DATACLAY_STORAGE_PATH": DATACLAY_STORAGE_ROOT,
    "DATACLAY_KV_HOST": nodes[0],
    "DATACLAY_USERNAME": "bscuser",
    "DATACLAY_PASSWORD": "bscpassword",
    "DATACLAY_DATASET" : "hpcdataset",
    "PYTHONPATH": f"{os.environ.get('PYTHONPATH')}:{os.path.expandvars('$PWD')}",
    "LD_LIBRARY_PATH": os.environ.get("LD_LIBRARY_PATH"),
    "DATACLAY_LOGLEVEL": "info",
}

print("Using the following extra environment:")
from pprint import pprint
pprint(extra_env)

print("* Starting Redis ...")
master_c.run(f"nohup {REDIS_SERVER} --protected-mode no &> {REDIS_LOG} &")

print("* Starting Metadata Service ...")
master_c.run(f"(nohup {PYTHON_BINARY} -u -m dataclay.metadata &> {METADATA_LOG} &)", env=extra_env)
print("* Metadata Service should be starting")
print("")

sleep(5)

print("* Starting Backends ...")
for i, backend_c in enumerate(backends_g):
    extra_env0 = extra_env.copy()
    extra_env1 = extra_env.copy()

    extra_env0["DATACLAY_BACKEND_PORT"] = 6890
    extra_env1["DATACLAY_BACKEND_PORT"] = 6891
    
    print("*** Starting backend %d" % i)
    backend_c.run(f"(nohup numactl -m 0 -N 0 -- {PYTHON_BINARY} -u -m dataclay.backend &> {BACKEND_LOG_TEMPLATE % (0, i)} &)",
                  env=extra_env0)
    backend_c.run(f"(nohup numactl -m 1 -N 1 -- {PYTHON_BINARY} -u -m dataclay.backend &> {BACKEND_LOG_TEMPLATE % (1, i)} &)",
                  env=extra_env1)

print("* Everything has been started / is starting")
