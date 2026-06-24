import argparse
import yaml
from dataclasses import asdict
from pathlib import Path

from config import Config, ModelConfig, OptimConfig, TrainConfig, IOConfig


def load_config(yaml_path: str) -> Config:
    with open(yaml_path) as f:
        raw = yaml.safe_load(f)
    return Config(
        model=ModelConfig(**raw["model"]),
        optim=OptimConfig(**raw["optim"]),
        train=TrainConfig(**raw["train"]),
        io=IOConfig(**raw["io"]),
    )


def dump_config(cfg: Config, out_path: str | Path) -> None:
    with open(out_path, "w") as f:
        yaml.safe_dump(asdict(cfg), f, sort_keys=False)


def parse_args() -> Config:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    return load_config(args.config)