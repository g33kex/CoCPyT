# This is the configuration file for CCpipeline

## Settings
# Name of the repository (change this to the name of your project)
REPO_NAME="CCpipeline"
# Which modules to load (change this if you need other modules)
MODULES="python/3.10 StdEnv/2023 cuda httpproxy"

## Default paths (change only if you know what you're doing)
# Where to store the git repo on CC
REPO_PATH="${HOME}/scratch/${REPO_NAME}"
# Where to extract the venv. This path must be accessible on both the login node and the compute node.
# It is recommended to use fast storage like /tmp or /localscratch
VENV_PATH="/tmp/${USER}/${REPO_NAME}"
# Where to store the venv tar (should be persistent storage)
VENV_TAR_PATH="${REPO_PATH}/venv.tar.gz"
# Location of the cluster requirements file
REQS_PATH="${REPO_PATH}/requirements_cluster.txt"
# Where to store the run logs
LOG_PATH="${REPO_PATH}/run.log" 
# Name of the git remote to push the repo to the cluster
REMOTE_NAME="cluster"

