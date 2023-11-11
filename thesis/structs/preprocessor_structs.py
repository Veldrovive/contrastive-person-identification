"""
Defines configuration for the preprocessing options.
"""

from pydantic import BaseModel, Field, validator

class PreprocessorConfig(BaseModel):
    pass

class MetaPreprocessorConfig(PreprocessorConfig):
    sample_rate: int = Field(..., description="The sample rate of the data")
    target_sample_rate: int | None = Field(None, description="The target sample rate to downsample to")
    stats_size_s: float = Field(0.5, description="The size of the window to use for getting the sample stats")
    clamping_sd: float = Field(20, description="The number of standard deviations to use for clamping")

    use_baseline_correction: bool = Field(..., description="Whether to use baseline correction")
    use_robust_scaler: bool = Field(..., description="Whether to use robust scaler from sklearn")
    use_clamping: bool = Field(..., description="Whether to use clamping")