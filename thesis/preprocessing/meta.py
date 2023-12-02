"""
Preprocessing meta's papers where they applied CLIP to EEG and MEG.
https://www.nature.com/articles/s42256-023-00714-5#Sec15

They argue that this method is just as effective as more advanced systems such as autoreject
for deep learning downstream tasks and does not require the full dataset to be in memory.
"""

from thesis.structs.preprocessor_structs import MetaPreprocessorConfig
import torch
import numpy as np
import time

from sklearn.preprocessing import RobustScaler
from torchaudio.transforms import Resample

def downsample(sample, sample_rate, target_rate):
    """
    Downsamples the sample from sample_rate to target_rate
    Assumes (channels, timesteps) shape
    """
    resample = Resample(sample_rate, target_rate)
    return resample(sample)

def baseline_correction(sample, num_points):
    """
    De-mean the signal by subtracting the mean of the first num_points
    Assumes (channels, timesteps) shape
    """
    baseline = sample[:, :num_points].mean(axis=1, keepdims=True)
    return sample - baseline

def robust_scaler(sample):
    """
    Scales the sample using a robust scaler
    Assumes (channels, timesteps) shape
    """
    scaler = RobustScaler()
    return scaler.fit_transform(sample)


def custom_robust_scaler(data, n=200):
    # Step 1: Determine the number of data points to use
    k = min(data.shape[1], n)
    
    # Step 2: Select the first k data points from each channel
    data_subset = data[:, :k]
    
    # Step 3: Compute the median and IQR for each channel
    median = np.median(data_subset, axis=1, keepdims=True)
    q75, q25 = np.percentile(data_subset, [75, 25], axis=1)
    iqr = q75 - q25
    
    # Avoid division by zero by setting IQRs of 0 to 1 (or you could use a small epsilon value)
    iqr[iqr == 0] = 1
    
    # Step 4: Scale the data using broadcasting
    scaled_data = (data - median) / iqr[:, np.newaxis]
    
    return scaled_data

def clamp(sample, max_sd):
    """
    Clamps any timesteps that exceed max_sd standard deviations
    """
    mean = sample.mean(axis=1, keepdims=True)
    sd = sample.std(axis=1, keepdims=True)
    return np.clip(sample, mean - max_sd * sd, mean + max_sd * sd)

# TODO: Check that I made the same assumptions as the paper


def preprocessor_factory(config: MetaPreprocessorConfig, track_times: bool = False):
    """
    Produces a preprocessing function based on the config
    """
    downsample_time = 0
    baseline_correction_time = 0
    robust_scaler_time = 0
    clamp_time = 0
    def preprocess(data):
        """
        Preprocesses the data
        """
        sample_rate = config.sample_rate
        data = torch.from_numpy(data)
        if config.target_sample_rate is not None:
            nonlocal downsample_time
            start = time.time()
            data = downsample(data, sample_rate, config.target_sample_rate)
            downsample_time += time.time() - start
            sample_rate = config.target_sample_rate
        if config.use_baseline_correction:
            nonlocal baseline_correction_time
            start = time.time()
            num_steps = int(sample_rate * config.stats_size_s)
            data = baseline_correction(data, num_steps)
            baseline_correction_time += time.time() - start
        if config.use_robust_scaler:
            nonlocal robust_scaler_time
            start = time.time()
            # data = robust_scaler(data)
            data = custom_robust_scaler(data)
            robust_scaler_time += time.time() - start
        if config.use_clamping:
            nonlocal clamp_time
            start = time.time()
            data = clamp(data, config.clamping_sd)
            clamp_time += time.time() - start
        return data

    def print_times():
        print(f"Downsample time: {downsample_time}")
        print(f"Baseline correction time: {baseline_correction_time}")
        print(f"Robust scaler time: {robust_scaler_time}")
        print(f"Clamp time: {clamp_time}")
    
    if track_times:
        return preprocess, print_times
    else:
        return preprocess