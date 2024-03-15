# Makes sure venv is in TMP_PATH and up to date.
# Runs automatically on login nodes and compute nodes when starting a job.
source  config.sh # User config

echo "===== Loading modules ====="
module load ${MODULES}
echo "===== Checking venv for updates ====="
current_checksum=$(md5sum "${VENV_TAR_PATH}" | cut -d ' ' -f 1)
if [ -f "${CHECKSUM_FILE}" ]; then
    stored_checksum=$(cat "${VENV_PATH}/venv_checksum")
else
    stored_checksum=""
fi
if [ "${current_checksum}" = "${stored_checksum}" ]; then
    echo "===== Skipping venv extraction ====="
else
    echo "===== Extracting venv ====="
    mkdir -p "${VENV_PATH}"
    tar -xzf "${VENV_TAR_PATH}" -C "${VENV_PATH}"
    echo "${current_checksum}" > "${VENV_PATH}/venv_checksum"
fi
echo "===== Entering venv ====="
source "${VENV_PATH}/venv/bin/activate"
