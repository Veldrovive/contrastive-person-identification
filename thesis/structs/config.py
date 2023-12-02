"""
The full config for a training run
"""

from pydantic import BaseModel, Field, validator
import subprocess
import os

from .dataset_structs import *
from .training_structs import *
from .preprocessor_structs import *
from .augmentation_structs import *
from .model_structs import *
from .contrastive_head_structs import *

from thesis.utils import get_git_hash


class Config(BaseModel):
    git_hash: str = Field(get_git_hash(), description="The git hash of the current commit")
    random_seed: int = Field(0, description="The seed to use for the random number generator")
    
    training_config: TrainingConfig = Field(...)

    load_time_preprocess_config: LoadTimePreprocessorConfig = Field(..., description="The preprocessing that is applied when datasets are loaded")
    inference_time_preprocess_config: PreprocessorConfig = Field(..., description="The preprocessing that is applied when the sample is returned from the dataloader")
    augmentation_config: AugmentationConfig | None = Field(None, description="Inference time preprocessing that is only applied to the training set")

    datasets: list[DatasetConfig] = Field(..., description="The datasets to use for training")
    train_dataloader_config: DataloaderConfig = Field(..., description="The dataloader config to use for training")
    eval_dataloader_config: DataloaderConfig = Field(..., description="The dataloader config to use for evaluation")

    target_channels: list[str] | None = Field(None, description="The target channels to use. None to automatically select channels")
    channel_blacklist: list[str] = Field([], description="The channels to exclude from the dataset")
    channel_equality_map: list[tuple[str, str]] = Field(default_equality_map, description="The channels to treat as equal. The first channel will be used as the representative channel.")
    window_size_s: float = Field(..., description="The size of the window to use for the sliding window")
    window_stride_s: float = Field(..., description="The stride of the window to use for the sliding window")

    embedding_model_config: ModelConfig = Field(..., description="The model config to use for training", discriminator="type")
    head_config: HeadConfig = Field(..., description="The head config to use for training", discriminator="type")