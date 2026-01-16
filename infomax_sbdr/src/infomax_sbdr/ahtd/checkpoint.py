"""Checkpoint utilities for saving and loading models."""

import jax
import pickle
import json
from pathlib import Path
from typing import Dict, Any, Optional
import jax.numpy as jnp
import numpy as np

# from .models.stacking import StackedNetwork
from .core.types import HyperParams
from flax.core.frozen_dict import FrozenDict

def save_checkpoint(
    module: FrozenDict,
    checkpoint_path: str,
    metadata: Optional[Dict[str, Any]] = None,
):
    """Save model checkpoint."""
    checkpoint_path = Path(checkpoint_path)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    
    def jax_to_numpy(tree):
        return jax.tree_util.tree_map(
            lambda x: np.array(x) if isinstance(x, jnp.ndarray) else x, tree
        )
    
    checkpoint_data = {
        "params_list": [jax_to_numpy(params) for params in network.params_list],
        "module_types": network.module_types,
        "hyperparams": {
            "gamma_f": network.hyperparams.gamma_f,
            "gamma_l": network.hyperparams.gamma_l,
            "p_target": network.hyperparams.p_target,
            "momentum": network.hyperparams.momentum,
            "lr": network.hyperparams.lr,
        },
        "metadata": metadata or {},
    }
    
    with open(checkpoint_path, "wb") as f:
        pickle.dump(checkpoint_data, f)
    
    print(f"Checkpoint saved to {checkpoint_path}")


def load_checkpoint(checkpoint_path: str) -> FrozenDict:
    """Load model checkpoint."""
    checkpoint_path = Path(checkpoint_path)
    
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    
    with open(checkpoint_path, "rb") as f:
        checkpoint_data = pickle.load(f)
    
    def numpy_to_jax(tree):
        return jax.tree_util.tree_map(
            lambda x: jnp.array(x) if isinstance(x, np.ndarray) else x, tree
        )
    
    params_list = [numpy_to_jax(params) for params in checkpoint_data["params_list"]]
    module_types = checkpoint_data["module_types"]
    
    hp_data = checkpoint_data["hyperparams"]
    hyperparams = HyperParams(
        gamma_f=hp_data["gamma_f"],
        gamma_l=hp_data["gamma_l"],
        p_target=hp_data["p_target"],
        momentum=hp_data["momentum"],
        lr=hp_data["lr"],
    )
    
    network = StackedNetwork(params_list, module_types, hyperparams)
    
    print(f"Checkpoint loaded from {checkpoint_path}")
    
    if "metadata" in checkpoint_data and checkpoint_data["metadata"]:
        print("Metadata:")
        for key, value in checkpoint_data["metadata"].items():
            print(f"  {key}: {value}")
    
    return network


def save_metadata(metadata: Dict[str, Any], metadata_path: str):
    """Save metadata as JSON."""
    metadata_path = Path(metadata_path)
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)


def load_metadata(metadata_path: str) -> Dict[str, Any]:
    """Load metadata from JSON."""
    with open(metadata_path, "r") as f:
        return json.load(f)


class CheckpointManager:
    """Manages checkpoints during training."""
    
    def __init__(
        self,
        checkpoint_dir: str,
        keep_best: bool = True,
        keep_last_n: int = 3,
        metric_name: str = "accuracy",
        metric_mode: str = "max",
    ):
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        self.keep_best = keep_best
        self.keep_last_n = keep_last_n
        self.metric_name = metric_name
        self.metric_mode = metric_mode
        
        self.best_metric = float("-inf") if metric_mode == "max" else float("inf")
        self.checkpoint_history = []
    
    def save(self, network: StackedNetwork, epoch: int, metrics: Dict[str, float]):
        """Save checkpoint with automatic management."""
        latest_path = self.checkpoint_dir / "latest.pkl"
        metadata = {"epoch": epoch, **metrics}
        save_checkpoint(network, str(latest_path), metadata)
        
        periodic_path = self.checkpoint_dir / f"checkpoint_epoch_{epoch}.pkl"
        save_checkpoint(network, str(periodic_path), metadata)
        self.checkpoint_history.append(periodic_path)
        
        if len(self.checkpoint_history) > self.keep_last_n:
            old_checkpoint = self.checkpoint_history.pop(0)
            if old_checkpoint.exists():
                old_checkpoint.unlink()
        
        if self.keep_best and self.metric_name in metrics:
            metric_value = metrics[self.metric_name]
            is_better = (
                (self.metric_mode == "max" and metric_value > self.best_metric)
                or (self.metric_mode == "min" and metric_value < self.best_metric)
            )
            
            if is_better:
                self.best_metric = metric_value
                best_path = self.checkpoint_dir / "best.pkl"
                save_checkpoint(network, str(best_path), metadata)
                print(f"New best {self.metric_name}: {metric_value:.4f}")
    
    def load_best(self) -> StackedNetwork:
        """Load best checkpoint."""
        return load_checkpoint(str(self.checkpoint_dir / "best.pkl"))
    
    def load_latest(self) -> StackedNetwork:
        """Load latest checkpoint."""
        return load_checkpoint(str(self.checkpoint_dir / "latest.pkl"))
