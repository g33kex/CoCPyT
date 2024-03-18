#!/bin/bash

# Variables
source config.sh # User config

# Handle arguments
if [ "$#" -lt 1 ]; then
    echo -e "Usage: run_on_cluster.sh [--sync-only] <host> [parameters for main.py]\nSpecify <host> as the target host.\n--sync-only: Only synchronize the git repository, don't run main."
    exit 1
fi

HOST=$1
COMMAND="python3 main.py ${@:2}"

echo "===== Initializing git repository in ${REPO_PATH} ====="
# SSH into the server and initialize the repository
ssh -T "${HOST}" "git init ${REPO_PATH}"
ssh -T "${HOST}" "git -C ${REPO_PATH} config receive.denyCurrentBranch updateInstead"

echo "===== Updating remote ${REMOTE_NAME} ====="
# Add the remote server to the local repository configuration
git remote add "${REMOTE_NAME}" "ssh://${HOST}:/${REPO_PATH}" 2>/dev/null || git remote set-url "${REMOTE_NAME}" "ssh://${HOST}:/${REPO_PATH}"
# 
git config receive.denyCurrentBranch updateInstead

# Get the current branch name
CURRENT_BRANCH=$(git rev-parse --abbrev-ref HEAD)

echo "===== Pushing branch ${CURRENT_BRANCH} to ${REMOTE_NAME} ====="
# Push the current branch to the remote server repository
git push "${REMOTE_NAME}" "${CURRENT_BRANCH}" --force

ssh -T "${HOST}" << EOF
cd ${REPO_PATH}
echo "===== Checking out ${CURRENT_BRANCH} ====="
git checkout "${CURRENT_BRANCH}"
echo "===== Setting up environment ====="
if [ ! -f "${VENV_TAR_PATH}" ]; then
    echo "Couldn't find ${VENV_TAR_PATH}. Please initialize the environment by running create_venv.sh on the cluster."
    exit 1
fi
source setup_environment.sh
echo '===== Starting "${COMMAND}" at date  $(date '+%Y-%m-%d %H:%M:%S') =====' >> "${LOG_FILE}"
nohup unbuffer ${COMMAND} >> "${LOG_FILE}" 2>&1 &
EOF

echo "===== Now tailing ${LOG_PATH} ====="

ssh -T "$HOST" "tail -F ${LOG_PATH}" 
