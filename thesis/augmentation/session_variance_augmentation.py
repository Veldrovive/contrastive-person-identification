"""
Attempts to simulate the domain shift between sessions by augmenting the data
"""

from thesis.structs.augmentation_structs import SessionVarianceAugmentationConfig
import torch as torch

def noise(data, amplitude=None, relative_amplitude=None):
    """

    """
    if relative_amplitude is not None:
        # Then the amplitude is some proportion of the SD of the data
        amplitude = relative_amplitude * torch.std(data)
    if amplitude is None:
        raise ValueError("Must provide either amplitude or relative_amplitude")
    noise = torch.randn(data.shape) * amplitude
    return data + noise

def scale_channel_amplitude(data, min_ratio, max_ratio):
    """
    Scales each channel a consistent amount
    The scale for each channel is chosen between min_ratio and max_ratio
    """
    # scales = np.random.uniform(min_ratio, max_ratio, size=(data.shape[0], 1))
    # return data * scales
    # Re implement this in pytorch
    scales = torch.rand(data.shape[0], 1) * (max_ratio - min_ratio) + min_ratio
    return data * scales

def offset_chanel(data, min_offset: int, max_offset: int):
    """
    Offsets each channel a consistent amount. Edge values are rolled over
    """
    # offsets = np.random.randint(min_offset, max_offset, size=(data.shape[0], 1))
    # return np.roll(data, offsets, axis=1)
    # Re implement this in pytorch
    offsets = torch.randint(min_offset, max_offset, size=(data.shape[0], 1))
    return torch.roll(data, offsets, dims=1)


def preprocessor_factory(config: SessionVarianceAugmentationConfig):
    """
    Returns a function that runs the session variance augmentation
    """

    def augment(data):
        """
        Assumes that data is (num_channels, num_timesteps)
        """
        if config.apply_gaussian_noise:
            data = noise(data, amplitude=config.noise_amplitude, relative_amplitude=config.relative_noise_amplitude)
        if config.apply_per_channel_amplitude_scaling:
            data = scale_channel_amplitude(data, config.min_amplitude_scaling, config.max_amplitude_scaling)
        if config.apply_per_channel_time_offset:
            data = offset_chanel(data, config.min_time_offset, config.max_time_offset)
        return data
    
    return augment
