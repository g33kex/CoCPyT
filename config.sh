# This is the configuration file for CoCPyT

## Settings
# Name of the repository (change this to the name of your project)
REPO_NAME="CoCPyT"
# Username on compute canada (change this if it's different than your local username)
USERNAME="${USER}"
# Which modules to load (change this if you need other modules or versions)
MODULES="StdEnv/2023 python/3.11.5 arrow/15.0.1 cuda/12.2 httpproxy/1.0"

## Default paths (change only if you know what you're doing)
# Where to store the git repo on CC
REPO_PATH="/home/${USERNAME}/scratch/${REPO_NAME}"
# Where to extract the venv. This path must be accessible on both the login node and the compute node.
# It is recommended to use fast storage like /tmp or /localscratch
VENV_PATH="/tmp/${USERNAME}/${REPO_NAME}"
# Where to store the venv tar (should be persistent storage)
VENV_TAR_PATH="${REPO_PATH}/venv.tar.gz"
# Location of the cluster requirements file
REQS_PATH="${REPO_PATH}/requirements_cluster.txt"
# Where to store the run logs
LOG_PATH="${REPO_PATH}/run.log" 
# Name of the git remote to push the repo to the cluster
REMOTE_NAME="cluster"

