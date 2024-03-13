"""Configuration file.""" 
from pathlib import Path
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, Any, List
import random
import pandas as pd
from hydra.core.config_store import ConfigStore
from hydra.conf import HydraConf, RunDir
from omegaconf import MISSING
from nevergrad.parametrization import parameter as p

## Default Configs
@dataclass
class ExperimentConfig:
    """Configuration of an experiment."""
    name: str = MISSING # Name of the experiment
    path: Path = MISSING # Path of the experiment
    checkpoints_path: Path = MISSING # Path of saved checkpoints
    experiments_folder: Path = Path("experiments/") # Experiments folder
    seed: int = 42 # Seed

    def _generate_name(self, wordlist_path: Path = Path("words.csv")):
        """Generated a name for the experiments using a wordlist."""
        dataframe = pd.read_csv(wordlist_path)
        adjective, noun = random.choice(dataframe["Adjective"]), random.choice(dataframe["Noun"])
        number = random.randint(1, 9999)
        return f"{adjective}_{noun}_{number}"

    def __post_init__(self):
        """Generate name and path, make sure path doesn't exists."""
        while self.name == MISSING or (
            self.path == MISSING and self.experiments_folder.joinpath(self.name).exists()
        ):
            self.name = self._generate_name()
        if self.path == MISSING:
            self.path = self.experiments_folder / self.name
        if self.checkpoints_path == MISSING:
            self.checkpoints_path = self.path / "checkpoints"


@dataclass
class CometConfig:
    """Configuration for comet.ml integration."""
    project: str = "CCpipeline"
    workspace: str = "g33kex"


@dataclass
class SlurmConfig:
    """SLURM job configuration."""
    artifacts_folder: Path = Path("artifacts") # Where to store the run artifacts. (logs, pickles)
    timeout_min: int = 6 * 60 # Max duration of job in minutes.
    mem_gb: int = 80 # Memory to allocate to each job in GB.
    cpus_per_task: int = 16 # Number of cpu per task.
    gpus_per_task: int = 1 # Number of gpus per task.
    gpus_per_node: Optional[int] = None # Number of gpus per node.
    ntasks_per_node: int = 1 # Number of tasks on each node.
    account: Optional[str] = None # Account to use for allocation

@dataclass
class DataConfig:
    """Data configuration."""
    dataset_path: Path = Path("data/openhermes2_5.json")
    batch_size: int = 4
    test_size: float = 0.2

@dataclass
class TrainConfig:
    """Trainer configuration."""
    n_epochs: int = 50
    base_lr: float = 2e-4
    # checkpoint: Optional[Path] = None

@dataclass
class ModelConfig:
    """Configuration of the model."""
    model_path: Path = Path("model/")

@dataclass
class NevergradConfig:
    """Nevergrad configuration."""
    budget: int = 10
    num_parallel_jobs: int = 2
    instrumentation: str = "base"

## Main Config
class Actions(Enum):
    """Actions for main."""
    train = 0 # Train the model
    tune = 1 # Tune the hyperparameters using nevergrad
    none = 2 # Do nothing (for testing purposes)

@dataclass
class CustomHydraConf(HydraConf):
    """Hydra configuration."""
    output_subdir: Optional[str] = field(default_factory=lambda: None)
    run: RunDir = field(default_factory=lambda: RunDir(dir="."))

@dataclass
class Config:
    """Default Configuration."""
    action: Actions = MISSING
    experiment: ExperimentConfig = MISSING
    model: ModelConfig = MISSING
    train: TrainConfig = field(default_factory=TrainConfig)
    nevergrad: Optional[NevergradConfig] = None
    data: DataConfig = field(default_factory=DataConfig)
    slurm: SlurmConfig = field(default_factory=SlurmConfig)
    comet: CometConfig = field(default_factory=CometConfig)

    # Hydra config
    defaults: List[Any] = field(
        default_factory=lambda: [
            "_self_",
            {"override hydra/job_logging": "stdout"},
        ]
    )
    hydra: CustomHydraConf = field(default_factory=CustomHydraConf)


## Presets
@dataclass
class Preset(Config):
    """Abstract Preset class."""

@dataclass
class Base(Preset):
    """Base preset."""
    model: ModelConfig = field(default_factory=ModelConfig)
    train: TrainConfig = field(default_factory=lambda: TrainConfig(n_epochs=3))

## Nevergrad Instrumentations
nevergrad_instrumentations = {
    "base": {
        "train.base_lr": p.Log(lower=1e-7, upper=1e-1),
    },
}

## Setup hydra
# Dirty hack, probably fixing a bug in hydra
# Basically you can't set hydra config in config that's not the primary config
# But Preset must inherit from Config to override value
# So Preset would inherit hydra parameters and try to set hydra settings outside of primary config
# We need to make sure hydra or defaults is not present in any subclass of Preset
for subclass in Preset.__subclasses__():
    if "hydra" in subclass.__dataclass_fields__:
        del subclass.__dataclass_fields__["hydra"]
    if "defaults" in subclass.__dataclass_fields__:
        del subclass.__dataclass_fields__["defaults"]

## Create main config and presets
cs = ConfigStore.instance()
cs.store(name="config", node=Config)
for subclass in Preset.__subclasses__():
    cs.store(group="preset", name=subclass.__name__.lower(), package="_global_", node=subclass)
