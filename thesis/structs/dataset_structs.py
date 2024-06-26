from pathlib import Path

from .preprocessor_structs import PreprocessorConfig
from .augmentation_structs import AugmentationConfig

from pydantic import BaseModel, Field, validator, model_validator, ValidationError

# When mapping between the 10/20 system and 10/10 system we have the following equalities
high_resolution_channel_equality_map = [  # We map from the 10/20 system to the 10/10 system
    ('T7', 'T3'),
    ('T8', 'T4'),
    ('P7', 'T5'),
    ('P8', 'T6'),
]  # Sourced from wikipedia https://en.wikipedia.org/wiki/10%E2%80%9320_system_%28EEG%29
# Better citation needed, but just from visual inspection of the two systems we can see this is correct

default_equality_map = [*high_resolution_channel_equality_map]

class DataloaderConfig(BaseModel):
    batch_size: int = Field(32, description="The batch size to use for training")
    shuffle: bool = Field(True, description="Whether to shuffle the training data")
    num_workers: int = Field(0, description="The number of workers to use for the dataloader")

class ContrastiveSubjectDatasetConfig(BaseModel):
    target_channels: list[str] | None = Field(None, description="The target channels to use. None to automatically select channels")
    toggle_direct_sum_on: bool = Field(False, description="If false then the shared subset of channels will be used. If true then the direct sum of the channels will be used")
    channel_blacklist: list[str] = Field([], description="The channels to exclude from the dataset")
    channel_equality_map: list[tuple[str, str]] = Field(default_equality_map, description="The channels to treat as equal. The first channel will be used as the representative channel.")

    window_size_s: float = Field(..., description="The size of the window to use for the sliding window")
    window_stride_s: float = Field(..., description="The stride of the window to use for the sliding window")

    sample_over_subjects_toggle: bool = Field(False, description="If true, then each subject will have even representation over an epoch. If false, each subject will have representation proportional the number of samples they have")
    max_samples_per_subject: int | None = Field(None, description="The maximum number of samples to use per subject. If None, then all samples will be used")

    positive_separate_session: bool = Field(True, description="Whether to only consider positive samples from the same session")
    error_on_no_separate_session: bool = Field(False, description="Error if there are no separate sessions for positive samples. If false, will default to using the same session")

    n_pos: int = Field(1, description="The number of positive samples to return per anchor")
    n_neg: int = Field(1, description="The number of negative samples to return per anchor")

    preprocessor_config: PreprocessorConfig | None = Field(None, description="The preprocessor config to use. If None, then no preprocessing will be done")
    augmentation_config: AugmentationConfig | None = Field(None, description="The augmentation config to use. If None, then no augmentation will be done")

    random_seed: int = Field(0, description="The seed to use for the random number generator")

class DatasetSplitConfig(BaseModel):
    train_prop: float = Field(0.8, description="The proportion of subjects to use for training")
    extrap_val_prop: float = Field(0.05, description="The proportion of subjects to use for validation in the extrapolation set")
    extrap_test_prop: float = Field(0.05, description="The proportion of subjects to use for testing in the extrapolation set")
    intra_val_prop: float = Field(0.05, description="The proportion of subjects to use for validation in the intra set")
    intra_test_prop: float = Field(0.05, description="The proportion of subjects to use for testing in the intra set")

    downstream_num_subjects: int = Field(-1, description="The number of subjects to use for the downstream task. If -1, then all subjects will be used")

    # Validate that the proportions add up to 1
    @model_validator(mode="after")
    def validate_proportions(self):
        s = self.train_prop + self.extrap_val_prop + self.extrap_test_prop + self.intra_val_prop + self.intra_test_prop
        if abs(s - 1) > 1e-6:
            raise ValidationError("The proportions must add up to 1")
        return self
    
class DatasetConfig(BaseModel):
    name: str = Field(..., description="The name of the dataset")
    path: Path = Field(..., description="The path to the dataset")
    split_config: DatasetSplitConfig = Field(..., description="The split config to use for the dataset")
    max_subjects: int | None = Field(None, description="The maximum number of subjects to use. If None, then all subjects will be used")