"""Configuration management using TOML."""

import toml
from pathlib import Path
from typing import Dict, Any, List
from dataclasses import dataclass, field


@dataclass
class HyperParamsConfig:
    gamma_f: float = 0.9
    gamma_l: float = 0.8
    p_target: float = 0.1
    momentum: float = 0.95
    lr: float = 0.1


@dataclass
class LayerConfig:
    kernel_size: List[int] = field(default_factory=lambda: [3, 3])
    in_channels: int = 2
    out_channels: int = 32
    init_scale_f: float = 1.0
    stride: List[int] = field(default_factory=lambda: [1, 1])
    padding: str = "SAME"


@dataclass
class ModelConfig:
    name: str = "unnamed_model"
    architecture: str = "conv2d_stack"
    layers: List[LayerConfig] = field(default_factory=list)
    seed: int = 42


@dataclass
class TrainingConfig:
    n_epochs: int = 50
    batch_size: int = 32
    shuffle: bool = True
    seed: int = 42
    skip_first: int = 0
    checkpoint_interval: int = 5
    eval_interval: int = 50


@dataclass
class ValidationConfig:
    split: float = 0.2
    batch_size: int = 32


@dataclass
class LoggingConfig:
    log_dir: str = "./logs"
    log_interval: int = 10
    save_activations: bool = True


@dataclass
class DatasetConfig:
    name: str = "dvs_gesture"
    data_dir: str = "./data"
    n_timesteps: int = 100
    train_split: float = 0.8
    dt: float = 1.0
    max_rate: float = 100.0
    height: int = 128
    width: int = 128
    subsample_factor: int = 1


@dataclass
class Config:
    model: ModelConfig = field(default_factory=ModelConfig)
    hyperparams: HyperParamsConfig = field(default_factory=HyperParamsConfig)

def _parse_toml_manually(config_path: str) -> Dict[str, Any]:
    """Simple manual TOML parser for basic configs."""
    toml_data = {"model": {}, "hyperparams": {}, "training": {}, "dataset": {}, "layers": []}
    current_section = None
    current_layer = None
    
    with open(config_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            
            if line == "[[layers]]":
                current_layer = {}
                toml_data["layers"].append(current_layer)
                continue
            elif line.startswith("[") and line.endswith("]"):
                current_section = line[1:-1]
                current_layer = None
                continue
            
            if "=" in line:
                key, value = line.split("=", 1)
                key = key.strip()
                value = value.strip()
                
                # Parse value
                if value.lower() in ("true", "false"):
                    value = value.lower() == "true"
                elif value.startswith("[") and value.endswith("]"):
                    value = eval(value)
                elif value.replace(".", "").replace("-", "").isdigit():
                    value = float(value) if "." in value else int(value)
                elif value.startswith('"') and value.endswith('"'):
                    value = value[1:-1]
                
                if current_layer is not None:
                    current_layer[key] = value
                else:
                    toml_data[current_section][key] = value
    
    return toml_data


# return FrozenDict({
#         'hyperparams': hyperparams,
#         'params': params,
#         'config': config,
#         'n_modules': len(params),
#     })



def load_model_config(config_path: str) -> Config:
    """Load configuration from TOML file."""
    config_path = Path(config_path)
    
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    with open(config_path, "rb") as f:
        toml_data = toml.load(config_path)
    
    # Parse sections
    hp_data = toml_data.get("hyperparams", {})
    hyperparams = HyperParamsConfig(**hp_data)
    
    model_data = toml_data.get("model", {})
    layers_data = toml_data.get("layers", [])
    layers = [LayerConfig(**layer_dict) for layer_dict in layers_data]
    model = ModelConfig(
        name=model_data.get("name", "unnamed_model"),
        architecture=model_data.get("architecture", None),
        layers=layers,
    )
    
    
    return Config(
        model=model,
        hyperparams=hyperparams,
    )


def save_model_config(config: Config, config_path: str):
    """Save configuration to TOML file."""
    config_path = Path(config_path)
    config_path.parent.mkdir(parents=True, exist_ok=True)
    
    lines = []
    
    lines.append("[model]")
    lines.append(f'name = "{config.model.name}"')
    lines.append(f'architecture = "{config.model.architecture}"')
    lines.append("")
    
    lines.append("[hyperparams]")
    lines.append(f"gamma_f = {config.hyperparams.gamma_f}")
    lines.append(f"gamma_l = {config.hyperparams.gamma_l}")
    lines.append(f"p_target = {config.hyperparams.p_target}")
    lines.append(f"momentum = {config.hyperparams.momentum}")
    lines.append(f"lr = {config.hyperparams.lr}")
    lines.append("")
    
    for layer in config.model.layers:
        lines.append("[[layers]]")
        lines.append(f"kernel_size = {list(layer.kernel_size)}")
        lines.append(f"in_channels = {layer.in_channels}")
        lines.append(f"out_channels = {layer.out_channels}")
        lines.append(f"init_scale_f = {layer.init_scale_f}")
        lines.append(f"stride = {list(layer.stride)}")
        lines.append(f'padding = "{layer.padding}"')
        lines.append("")
    
    with open(config_path, "w") as f:
        f.write("\n".join(lines))
