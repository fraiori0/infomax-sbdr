"""Anti-Hebbian TD Learning Framework."""

from .core import dense, conv1d, conv2d
from .core.types import (
    HyperParams,
    DenseParams, DenseState, DenseOutputs,
    Conv1DParams, Conv1DState, Conv1DOutputs,
    Conv2DParams, Conv2DState, Conv2DOutputs,
    AHTDModule,
)

from .learning import updates

from .models.stacking import (
    # StackedNetwork,
    forward_stack,
    update_stack,
    extract_features,
    init_conv2d_stack,
    init_state_from_input
)

from .config import (
    Config,
    HyperParamsConfig,
    LayerConfig,
    ModelConfig,
    TrainingConfig,
    DatasetConfig,
    load_model_config,
    save_model_config,
)

from .model_builder import build_model #, get_model_info

# from .checkpoint import (
#     save_checkpoint,
#     load_checkpoint,
#     CheckpointManager,
# )

__all__ = [
    "dense", "conv1d", "conv2d",
    "HyperParams",
    "DenseParams", "DenseState", "DenseOutputs",
    "Conv1DParams", "Conv1DState", "Conv1DOutputs",
    "Conv2DParams", "Conv2DState", "Conv2DOutputs",
    "updates",
    "StackedNetwork",
    "forward_stack",
    "update_stack",
    "extract_features",
    "init_conv2d_stack",
    "Config",
    "load_config",
    "save_config",
    "build_model",
    # "get_model_info",
    # "save_checkpoint",
    # "load_checkpoint",
    # "CheckpointManager",
    "init_state_from_input"
]
