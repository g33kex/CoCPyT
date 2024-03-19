#!/usr/bin/env python3
"""Run experiments."""

import logging
import os.path
import sys
from concurrent.futures import Executor
from typing import Any, Callable
from submitit import AutoExecutor, Job
from submitit.helpers import RsyncSnapshot
from omegaconf import OmegaConf
import hydra
import nevergrad as ng

from src.config import (
    Actions,
    ExperimentConfig,
    SlurmConfig,
    Config,
    nevergrad_instrumentations,
)
from src.train import train

logging.basicConfig(level=logging.INFO)


def create_experiment(cfg):
    """Create new experiment, directory, and add it to the config"""
    experiment_config = OmegaConf.structured(ExperimentConfig())
    cfg = OmegaConf.merge(cfg, {"experiment": experiment_config})
    cfg.experiment.path.mkdir(parents=True)
    return cfg


def build_instrumentation(cfg):
    """Build nevergrad instrumentation based on config"""
    if cfg.nevergrad.instrumentation not in nevergrad_instrumentations:
        print(
            f"Invalid nevergrad instrumentation '{cfg.nevergrad.instrumentation}'.",
            f"Possible values are {list(nevergrad_instrumentations.keys())}.",
        )
        sys.exit(1)
    instrumentation_dict = nevergrad_instrumentations[cfg.nevergrad.instrumentation]
    return ng.p.Instrumentation(cfg, **instrumentation_dict)


class ExperimentExecutor(Executor):
    """Executor for an experiment."""

    def __init__(self, cfg: SlurmConfig):
        super().__init__()
        self.slurm_executor = AutoExecutor(folder="./")
        self.slurm_executor.update_parameters(
            timeout_min=cfg.timeout_min,
            mem_gb=cfg.mem_gb,
            cpus_per_task=cfg.cpus_per_task,
            gpus_per_node=cfg.gpus_per_node,
            slurm_gpus_per_task=cfg.gpus_per_task,
            slurm_ntasks_per_node=cfg.ntasks_per_node,
            slurm_account=cfg.account,
            slurm_setup=["source setup_environment.sh"],
        )

    def submit(self, fn: Callable[..., Any], /, *args: Any, **kwargs: Any) -> Job[Any]:
        """Submit the experiment to slurm."""
        if args:
            cfg = args[0]
        else:
            raise ValueError("Missing first argument 'cfg'")
        # Create an experiment and add it to the config
        cfg = create_experiment(cfg)
        # Create logger with experiment name
        log = logging.getLogger(cfg.experiment.name)
        log.info("Submitting experiment with overrides %s", kwargs)

        # Apply overrides to the config
        for key, value in kwargs.items():
            OmegaConf.update(cfg, key, value, merge=True)

        # Set job name and path
        self.slurm_executor.update_parameters(name=cfg.experiment.name)
        self.slurm_executor.folder = cfg.experiment.path

        # Create relative symlinks for large files not in git
        code_path = cfg.experiment.path.joinpath("code")
        code_path.mkdir(parents=True)
        for path in cfg.experiment.symlink_paths:
            link_path = code_path / path
            link_path.parent.mkdir(parents=True)
            link_path.symlink_to(os.path.relpath(link_path, code_path))

        # Take snapshot of commited files and submit job
        with RsyncSnapshot(snapshot_dir=code_path):
            job = self.slurm_executor.submit(fn, cfg)
        log.info(
            "Submitted experiment %s with job id %s", cfg.experiment.name, job.job_id
        )

        # Create log files symlinks
        # log_files = [("error.log", job.paths.stderr), ("output.log", job.paths.stdout)]
        # for log_file, target_path in log_files:
        #     symlink_path = cfg.experiment.path.joinpath(log_file)
        #     tgt = os.path.relpath(target_path, cfg.experiment.path)
        #     symlink_path.symlink_to(tgt)

        return job


def run_single_experiment(cfg, fn) -> float:
    """Runs a single experiment, either traditionally or with submitit."""
    if "slurm" not in cfg:
        cfg = create_experiment(cfg)
        # Create logger with experiment name
        log = logging.getLogger(cfg.experiment.name)
        log.info("Running locally without submitit")
        return fn(cfg)

    executor = ExperimentExecutor(cfg.slurm)
    job = executor.submit(fn, cfg)
    output = job.result()
    return output


def print_cfg(cfg: Config) -> None:
    """Print configuration."""
    print(OmegaConf.to_yaml(cfg))
    print("Path: ", os.getcwd())


def hyperparameter_tuning(cfg: Config) -> None:
    """Tune hyperparameters with nevergrad."""

    executor = ExperimentExecutor(cfg.slurm)
    instrumentation = build_instrumentation(cfg)
    print(
        f"Starting hyperparameter tuning with instrumentation {cfg.nevergrad.instrumentation}"
    )
    optimizer = ng.optimizers.NGOpt(
        parametrization=instrumentation,
        budget=cfg.nevergrad.budget,
        num_workers=cfg.nevergrad.num_parallel_jobs,
    )
    recommendation = optimizer.minimize(train, executor=executor, verbosity=2)
    print("Finished hyperparameter tuning")
    print("Recommandation:", recommendation)


@hydra.main(version_base=None, config_name="config")
def main(cfg: Config) -> None:
    """Entrypoint."""
    if OmegaConf.is_missing(cfg, "action"):
        ok_actions = "|".join([action.name for action in Actions])
        print(f"Usage: main.py action=[{ok_actions}]")
        sys.exit(1)
    if cfg.action == Actions.train:
        run_single_experiment(cfg, train)
    elif cfg.action == Actions.tune:
        hyperparameter_tuning(cfg)
    elif cfg.action == Actions.none:
        run_single_experiment(cfg, print_cfg)


if __name__ == "__main__":
    main()
