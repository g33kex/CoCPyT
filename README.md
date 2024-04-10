# CoCPyT

This is a Python project template to easily run experiments on [Compute Canada](https://alliancecan.ca/en) clusters.

Experiment parameters can be set with strongly-typed structured configuration using [hydra](https://hydra.cc/) and overwritten from the command-line.

Jobs on the SLURM cluster are started from Python with [submitit](https://github.com/facebookincubator/submitit) which allows automated hyper parameter tuning with [nevergrad](https://facebookresearch.github.io/nevergrad/).

The experiment metrics can be tracked on [comet](https://comet.ml).

This template shows as an example how to fine-tune [Mistral 7B](https://huggingface.co/mistralai/Mistral-7B-v0.1) on the [OpenHermes-2.5 dataset](https://huggingface.co/datasets/teknium/OpenHermes-2.5) using [QLoRA](https://arxiv.org/abs/2305.14314). See [Example Usage](#example-usage).

## Template structure

This section describes the structure of CoCPyT.

```
├── README.md # This file
├── config.sh # Configuration related to Compute Canada
├── create_venv.sh # Create a python virtual environment archive
├── data # Contains the training data
├── experiments # Contains the different experiments
│   ├── abrasive_reef_4045 # Each experiment has a random name
│   │   └── checkpoints # Checkpoints stored during training
│   ├── angular_bison_3657
│   ├── buoyant_pancake_3541
├── main.py # Main python file to start experiments
├── model # Model weights
├── run_on_cluster.sh # Run main.py on the computing cluster
├── setup_environment.sh # Setup the virtual environment on the compute node
├── src # Project sources
│   ├── __init__.py # Define a python module
│   ├── callbacks.py # Training callbacks
│   ├── config.py # Experiment structured configuration
│   ├── data.py # Dataloader
│   └── train.py # Trainer
└── words.csv # List of words to use for experiment names
```

This structure can of course be adapted to suit your needs. Here is a rundown of the important files:
- `main.py` contains the code to start new jobs on the SLURM cluster. It can start a single training jobs, or multiple jobs for hyper parameter searches. You can add more actions (for instance test) to this file. For instance, using `./main.py +action=train` will call the `train` function of `train.py`. You can add easily add more actions.
- `train.py` You should modify this to train the model you need.
- `config.py` This file contains the default configuration of your experiments. Every configuration defined here can be accessed through Hydra's config object `cfg` in your code. They can also be easily overwritten in the command-line, and you can also define different presets. Please read the [Example Usage](#example-usage) section and [Hydra's documentation](https://hydra.cc/docs/intro/) for more info.
- `config.sh` This file contains cluster related configuration, for instance where to store your project files on the node or which modules to load.

## Setup

This section explains how to setup your cluster for experiments.

### SSH Configuration

From April 2024, Compute Canada uses [Multifactor authentication](https://docs.alliancecan.ca/wiki/Multifactor_authentication). Since the `run_on_cluster` script automatically opens ssh connections, you can use a persistent SSH connection to avoid being asked for the second factor code multiple times.

For instance to only be asked for your code every day, edit `~/.ssh/config` and add the following lines. Replace `your_username` with your username on the cluster. This example is for beluga but you can use any other cluster.
```
Host beluga
    Hostname beluga.alliancecan.ca
    User your_username
    ControlPath ~/.ssh/cm-%r@%h:%p
    ControlMaster auto
    ControlPersist 1d
```

### Comet Configuration

To use [comet](comet.ml) and see your training metrics graphed in real time, you must add your comet API key to `~/.comet.config` on the cluster. To get an API key, create a comet account and go your your account settings. They offer a free academic accounts here: <https://www.comet.com/signup?plan=academic>
```
[comet]
api_key=YOUR_COMET_API_KEY
```

You can add this file to your home directory on your own machine as well if you want to also log your local runs on comet.

If you don't want to use comet, you can add `~comet` in your experiment parameters. If you'd like to add support for an open source alternative to comet, please make a PR! 

### Python Environment Setup

In order to run your Python project on Compute Canada, you need to configure the right modules and identify the correct requirements. See [here](https://docs.alliancecan.ca/wiki/Python) for more information about Python on Compute Canada.

#### Configure the modules

Python versions and certain dependencies like cuda are managed by modules on Compute Canada. The modules can be configured in `config.sh`. For instance, the default modules below are suitable for running a machine learning project with cuda and Python 3.11.5. The `httpproxy` module is required for comet. You can find a list of available modules [here](https://docs.alliancecan.ca/wiki/Available_software#List_of_globally-installed_modules).

```bash
MODULES="StdEnv/2023 python/3.11.5 arrow/15.0.1 cuda/12.2 httpproxy/1.0"
```

#### Generate the requirements

The `requirements_cluster.txt` file contains the required Python dependencies for your project and must be crafted to work with the particular cluster you're using, because all of the usual Python packages might not be available there. 

It can be a bit difficult to find the right versions of the Python modules that are compatible with the cluster, an those versions will most likely differ from your local Python installation because Compute Canada clusters use a [custom wheelhouse](https://docs.alliancecan.ca/wiki/Available_Python_wheels). Here's a general method to build `requirements_cluster.txt` file.

1. Copy your project to the cluster you'd like to use, for instance on beluga. This can be done automatically with the `run_on_cluster` script:
    ```bash
    ./run_on_cluster.sh --sync-only beluga
    ```
2. Connect via ssh to the cluster, load the modules and create a new Python virtual environment in `/tmp`:
    ```bash
    ssh beluga
    cd ~/scratch/CoCPyT
    source config.sh
    module load $MODULES
    virtualenv --no-download /tmp/$USER/venv
    source /tmp/$USER/venv/bin/activate
    ```
3. Try to run run your project, for instance try to start the training:
    ```bash
    python3 main.py "~slurm" +action=train
    ```
    It will likely fail and complain about a missing wheel. For instance:
    ```
    ModuleNotFoundError: No module named 'submitit'
    ```
4. Install the missing wheel. The list of available wheels can be found [here](https://docs.alliancecan.ca/wiki/Available_Python_wheels). Sometimes the name of the wheel differs from the name of the name reported above, so check the documentation of the library you're using for installation instruction and the list of available wheels. Sometimes, the wheel or the version you need isn't available. In that case, you can either install it from [pypi](https://pypi.org) by removing `--no-index`, or contact Compute Canada technical support and ask them to add the wheel. Note that some wheels like `pyarrow` are only available through [modules](#configure-the-modules), so you need to make sure they don't end up in your `requirements_cluster.txt`.
    ```bash
    pip install --no-index submitit
    ```
5. Repeat steps 3 and 4 until your application runs correctly. Kill it immediately once it does because you're not supposed to run compute intensive tasks on login nodes! Make sure to test all of the functionalities of your application if it's loading modules dynamically.
6. Build the `requirements_cluster.txt` file:
    ```bash
    pip freeze > requirements_cluster.txt
    ```


#### Build the virtual environment

The `requirements_cluster.txt` file is used to generate a Python virtual environment containing all of the required packages. This environment is gzipped so it can be quickly transferred to the compute nodes. This method ensures that the tasks spawn very quickly by avoiding to reinstalling the environment every time. Since virtual environment are not relocatable, we build and extract them in `/tmp`, a path that is accessible on both the login and compute nodes.

To generate the virtual environment, run `./create_venv.sh` on the login node. The resulting `venv.tar.gz` will be automatically copied to the compute node and extracted when starting a task. You need to run that script again each time you modify the requirements.

## General Usage

The parameters for your experiments are defined in `src/config.py` and are structured and strongly-typed. Read the [hydra documentation](https://hydra.cc/docs/tutorials/structured_config/schema/) to learn how to modify this file to add your own parameters. You can also define presets to quickly change between different models for instance.

The experiment results and checkpoints are stored in the `experiments` folder.

### Training locally

To use `main.py` locally, please specify an action (i.e. `train`) and preset (i.e. `base`) using:
```
python3 main.py +action=[action] +preset=[preset]
```

Hydra is used to overwrite parameters directly from the commandline.

For instance, you can overwrite the batch_size and n_epochs by doing:
```
python3 main.py +action=train +preset=base data.batch_size=16 train.n_epochs=10
```

To disable submitit and/or comet you can use the `~` operator:
```
python3 main.py +action=train +preset=base "~comet" "~slurm"
```

### Training on cluster

First, set the `REPO_NAME` in `config.sh` to the name of your project.

Then start the training with:
```bash
./run_on_cluster.sh [host] <params>
```

Where `<params>` are the parameters to pass to `main.py` (see [Training locally](#training-locally))

The code for the current branch will be synchronized to the `REPO_PATH` directory on the cluster and the `params` will be passed to `main.py` on the login node of the cluster. `submitit` is used in `main.py` to schedule and run the jobs with SLURM and gather the results. The logs of the runs are stored in `LOG_PATH`. 

A snapshot of the git repository is taken when the job starts, so it is possible to start multiple concurrent jobs with different versions of the code.

The `REPO_PATH` variable can be used to specify an alternate location to store and run the code.

If running locally (i.e. not on a SLURM cluster node), `submitit` won't try to use SLURM, so you can use `python3 main.py [param]` as usual on your computer.

If you want to run your job in an interactive `salloc` session, add `~slurm` to the parameters of `main.py` to avoid submitting the job.

## Example Usage

Follow these instructions to fine-tune Mistral on beluga.

1. Clone this project locally on your machine.
    ```bash
    git clone https://github.com/g33kex/CoCPyT
    ```
2. Make sure you have [setup your SSH connection](#configure-ssh) to beluga.
3. Copy the project over to beluga.
    ```bash
    ./run_on_cluster.sh --sync-only beluga
    ```

Then, connect to beluga via ssh and [build the virtual environment](#build-the-virtual-environment). You can use the default `requirements_cluster.txt` file that has been generated for this example:
```bash
./create_venv.sh
```

If you don't have it already, install the Hugging Face Hub module.
```bash
pip install --no-index huggingface_hub
```

Download the model:
```bash
huggingface-cli download mistralai/Mistral-7B-v0.1 --local-dir model
```

Download the dataset:
```bash
huggingface-cli download --repo-type dataset teknium/OpenHermes-2.5 --local-dir data
```

Submit a fine-tuning job to SLURM directly from your local machine:
```bash
./run_on_cluster.sh beluga +action=train +preset=base/run_on_cluster.sh beluga +action=train +preset=base
```
## License

Copyright (C) 2024 g33kex, Kkameleon, ran-ya, yberreby

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

We are not affiliated with the Digital Research Alliance of Canada.
