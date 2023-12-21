"""
Defines configuration for the preprocessing options.
"""

from pydantic import BaseModel, Field, validator
from typing import Literal

class MetaPreprocessorConfig(BaseModel):
    type: Literal["meta_preprocessor"] = "meta_preprocessor"

    sample_rate: int = Field(..., description="The sample rate of the data")
    target_sample_rate: int | None = Field(None, description="The target sample rate to downsample to")
    stats_size_s: float = Field(0.5, description="The size of the window to use for getting the sample stats")
    clamping_sd: float = Field(20, description="The number of standard deviations to use for clamping")

    use_baseline_correction: bool = Field(..., description="Whether to use baseline correction")
    use_robust_scaler: bool = Field(..., description="Whether to use robust scaler from sklearn")
    use_clamping: bool = Field(..., description="Whether to use clamping")

class InHousePreprocessorConfig(BaseModel):
    type: Literal["in_house_preprocessor"] = "in_house_preprocessor"

    stats_size_s: float = Field(0.5, description="The size of the window to use for getting the sample stats")
    target_sample_rate: int | None = Field(None, description="The target sample rate to downsample to. Errors if base sample rate < target sample rate")
    clamping_sd: float = Field(5, description="The number of standard deviations to use for clamping")
    freq_band: tuple[float | None, float | None] | None = Field(None, description="The frequency band to filter to. None means no filtering.")

    use_baseline_correction: bool = Field(..., description="Whether to use baseline correction")
    use_robust_scaler: bool = Field(..., description="Whether to use robust scaler from sklearn")
    use_clamping: bool = Field(..., description="Whether to use clamping")

PreprocessorConfig = MetaPreprocessorConfig | InHousePreprocessorConfig  # Union of all preprocessor configs

class LoadTimePreprocessorConfig(BaseModel):
    type: Literal["load_time_preprocessor"] = "load_time_preprocessor"

    target_sample_rate: int | None = Field(None, description="The target sample rate to downsample to. Errors if base sample rate < target sample rate")
    band_pass_lower_cutoff: float | None = Field(None, description="The lower cutoff for the band pass filter")
    band_pass_upper_cutoff: float | None = Field(None, description="The upper cutoff for the band pass filter")