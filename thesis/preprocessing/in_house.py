import torch
from torchaudio.transforms import Resample

from thesis.structs import InHousePreprocessorConfig

class InHousePreprocessor:
    def __init__(self, config: InHousePreprocessorConfig):
        self.resamplers = {}  # Maps from sample rate to resampler
        self.target_sample_rate = config.target_sample_rate
        self.stats_size_s = config.stats_size_s
        self.clamping_sd = config.clamping_sd
        self.freq_band = config.freq_band

        self.use_baseline_correction = config.use_baseline_correction
        self.use_robust_scaler = config.use_robust_scaler
        self.use_clamping = config.use_clamping

    def downsample(self, sample, sample_rate):
        """
        Downsamples the sample from sample_rate to target_rate
        Assumes (channels, timesteps) shape
        """
        if sample_rate not in self.resamplers:
            self.resamplers[sample_rate] = Resample(sample_rate, self.target_sample_rate)
        return self.resamplers[sample_rate](sample)

    def baseline_correction(self, sample, num_points):
        """
        De-mean the signal by subtracting the mean of the first num_points
        Assumes (channels, timesteps) shape
        """
        baseline = sample[:, :num_points].mean(axis=1, keepdims=True)
        return sample - baseline

    # def robust_scaler(self, data, num_points):
    #     """
    #     Scales the data using a robust scaler
    #     """
    #     k = min(data.shape[1], num_points)
        
    #     data_subset = data[:, :k]
        
    #     median = np.median(data_subset, axis=1, keepdims=True)
    #     q75, q25 = np.percentile(data_subset, [75, 25], axis=1)
    #     iqr = q75 - q25
        
    #     iqr[iqr == 0] = 1
        
    #     # Use broadcasting to scale the data
    #     scaled_data = (data - median) / iqr[:, np.newaxis]
        
    #     return scaled_data

    def robust_scaler(self, data, num_points):
        """
        Scales the data using a robust scaler in PyTorch
        """
        k = min(data.shape[1], num_points)
        
        data_subset = data[:, :k]
        
        # Calculate the median
        median = torch.median(data_subset, dim=1, keepdim=True).values

        # Calculate the 25th and 75th percentiles
        q75, q25 = torch.quantile(data_subset, torch.tensor([0.75, 0.25]), dim=1)

        # Calculate the interquartile range (IQR)
        iqr = q75 - q25
        
        # Replace zeros in IQR with ones to avoid division by zero
        iqr[iqr == 0] = 1
        
        # Perform the scaling
        scaled_data = (data - median) / iqr.unsqueeze(1)
        
        return scaled_data

    def clamp(self, sample, max_sd):
        """
        Clamps any timesteps in a PyTorch tensor that exceed max_sd standard deviations
        """
        # Calculate mean and standard deviation along the specified axis
        mean = torch.mean(sample, dim=1, keepdim=True)
        sd = torch.std(sample, dim=1, keepdim=True)

        # Clip values to be within the specified standard deviation range
        return torch.clamp(sample, mean - max_sd * sd, mean + max_sd * sd)

    def band_pass(self, sample, lower_cutoff, upper_cutoff, sample_rate):
        """
        Applies a band pass filter to the sample

        Lower or upper may be none to indicate no cutoff
        """
        raise NotImplementedError("Band pass filtering for single windows is not implemented. Please use a load time preprocessor instead.")

    def process_sample(self, sample, metadata):
        """

        """
        sample_freq = metadata["freq"]
        states_size_num_points = int(self.stats_size_s * sample_freq)

        if self.target_sample_rate is not None:
            sample = self.downsample(sample, sample_freq)
            # Set the metadata freq to the new sample rate and update the sample_freq
            metadata["freq"] = self.target_sample_rate
            sample_freq = self.target_sample_rate
            states_size_num_points = int(self.stats_size_s * sample_freq)

        if self.use_baseline_correction:
            sample = self.baseline_correction(sample, states_size_num_points)

        if self.use_robust_scaler:
            sample = self.robust_scaler(sample, states_size_num_points)

        if self.use_clamping:
            sample = self.clamp(sample, self.clamping_sd)

        if self.freq_band is not None and (self.freq_band[0] is not None or self.freq_band[1] is not None):
            sample = self.band_pass(sample, self.freq_band[0], self.freq_band[1], sample_freq)

        return sample