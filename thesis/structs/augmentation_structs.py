"""
Defines configuration for the augmentation system
"""

from pydantic import BaseModel, Field, validator

class AugmentationConfig(BaseModel):
    pass

class SessionVarianceAugmentationConfig(AugmentationConfig):
    apply_gaussian_noise: bool = Field(True, description="Whether to add gaussian noise to the data")
    noise_amplitude: float | None = Field(None, description="The amplitude of the noise to add to the session variance")
    relative_noise_amplitude: float | None = Field(0.01, description="The amplitude of the noise to add to the session variance, as a proportion of the SD of the data")

    apply_per_channel_amplitude_scaling: bool = Field(True, description="Whether to scale the amplitude of each channel by a random amount")
    min_amplitude_scaling: float = Field(0.5, description="The minimum amount to scale the amplitude of each channel by")
    max_amplitude_scaling: float = Field(1.5, description="The maximum amount to scale the amplitude of each channel by")

    apply_per_channel_time_offset: bool = Field(False, description="Whether to shift the time of each channel by a random amount")
    min_time_offset: int = Field(-5, description="The minimum amount to shift the time of each channel by")
    max_time_offset: int = Field(5, description="The maximum amount to shift the time of each channel by")