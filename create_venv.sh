# Setup/update virtualenv (run this on the cluster from the repo directory)
# venv is setup in tmp, then archived and copied back to scratch
source config.sh # User config
echo "===== Loading modules ====="
module load "${MODULES}"
echo "===== Creating python virtual environment ====="
mkdir -p "${VENV_PATH}"
virtualenv --no-download "${VENV_PATH}/venv"
source "${VENV_PATH}/venv/bin/activate"
echo "===== Installing requirements ====="
pip install --no-index --upgrade pip
pip install --no-deps -r "${REQS_PATH}"
echo "===== Packaging venv ====="
deactivate
tar -czf "${VENV_TAR_PATH}" "${VENV_PATH}/venv"
echo "===== Finished! ====="
