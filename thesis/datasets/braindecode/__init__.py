import numpy as np
import torch
from torchaudio.transforms import Resample
from torch.utils.data import Dataset
from braindecode.datasets import MOABBDataset
from braindecode.preprocessing import create_fixed_length_windows
from typing import TypedDict, Dict, Tuple
from thesis.structs.dataset_structs import WindowedContrastiveDatasetConfig

SubjectID = str
SessionID = str
DatasetName = str

class ElementDict(TypedDict):
    subject: SubjectID
    session: SessionID
    data: np.ndarray  # Shape: (num_windows, num_channels, window_size_samples)
    metadata: dict  # At least contains the keys 'channels', 'freq', 'window_size_s', 'window_stride_s'

WindowedDataset = Dict[Tuple[SubjectID, SessionID], ElementDict]

class SplitDataset(TypedDict):
    train: WindowedDataset
    extrap_val: WindowedDataset
    extrap_test: WindowedDataset
    intra_val: WindowedDataset
    intra_test: WindowedDataset

class WindowedContrastiveDataset(Dataset):
    def __init__(self, windowed_dataset: WindowedDataset, config: WindowedContrastiveDatasetConfig):
        self.windowed_dataset = windowed_dataset
        self.config = config

        self.target_channels = None  # Set in select_channels
        self.target_freq = config.target_freq

        self.index_map = None  # Set in construct_index_map

        self.select_channels()
        self.construct_index_map()

    def select_channels(self):
        """
        Selects the target channels to use based on the config and the channels in the dataset
        """
        # If the target channels are explicitly set we can just use those
        if self.config.target_channels is not None:
            self.target_channels = self.config.target_channels
            return

        # Some channels have multiple names. We build a from:to map to convert them to a single name.
        channel_equality_map = {mapping[1]: mapping[0] for mapping in self.config.channel_equality_map}

        # Otherwise, we use the channels in the dataset
        # To start, we get a map from the dataset to the list of channels
        dataset_channels_map = {}
        for (subject_id, session_id), element in self.windowed_dataset.items():
            dataset = element['metadata']['dataset']
            if dataset not in dataset_channels_map:
                # Then we add the channels to the map
                mapped_channels = []
                for channel in element['metadata']['channels']:
                    if channel in channel_equality_map:
                        mapped_channels.append(channel_equality_map[channel])
                    else:
                        mapped_channels.append(channel)

        channels = []
        if self.config.toggle_direct_sum_on:
            # Then the channels are all channels from all datasets. We can use the union of all channels
            for dataset, dataset_channels in dataset_channels_map.items():
                channels += dataset_channels
            channels = list(set(channels))
        else:
            # Then the channels are the intersection of all channels from all datasets
            for dataset, dataset_channels in dataset_channels_map.items():
                if len(channels) == 0:
                    channels = dataset_channels
                else:
                    channels = list(set(channels) & set(dataset_channels))
        
        # Finally, we remove the channels in the blacklist
        for channel in self.config.channel_blacklist:
            if channel in channels:
                channels.remove(channel)
            
        assert len(channels) > 0, "There must be at least one channel"
        self.target_channels = channels

    def construct_index_map(self):
        """
        Constructs a map from the index in the dataset to a tuple (subject_id, session_id, window_index)
        The length of the dataset is exactly equal to the length of the index map

        If sample_over_subjects_toggle, then we need to first get the max number of samples any subject has so that we can re-sample
        from subjects with fewer samples to even out the probability that any given index maps to a subject

        If sample_over_subjects_toggle is False, then we can simply generate a map by iterating over the windowed dataset once
        """
        self.index_map = []

        # For bookkeeping purposes, we keep track of the number of samples we have for each subject in reality
        self.subject_sample_num_map = {}
        for (subject_id, session_id), element in self.windowed_dataset.items():
            if subject_id not in self.subject_sample_num_map:
                self.subject_sample_num_map[subject_id] = 0
            self.subject_sample_num_map[subject_id] += len(element['data'])

        if not self.config.sample_over_subjects_toggle:
            # Then we can simply iterate over every sample and add it to the map
            for (subject_id, session_id), element in self.windowed_dataset.items():
                for window_index in range(len(element['data'])):
                    self.index_map.append((subject_id, session_id, window_index))
            return
        # Otherwise we need to complicate this a bit more
        # Step 1: Find the max number of samples any subject has
        max_num_samples = max(self.subject_sample_num_map.values())
        subject_upsample_ratio_map = {subject_id: max_num_samples / num_samples for (subject_id, num_samples) in self.subject_sample_num_map.items()}

        # Step 2: Resample the subjects to have the same number of samples
        for (subject_id, session_id), element in self.windowed_dataset.items():
            # We use a ratio so that it can operate over each session individually. This does mean we may be off by a bit due to rounding
            # but the idea is to get approximately the same number of samples. Exactness is not important
            subject_upsample_ratio = subject_upsample_ratio_map[subject_id]
            num_samples = len(element['data'])
            num_samples_to_add = int((subject_upsample_ratio - 1) * num_samples)  # The number of extra samples to add
            # Add the normal samples
            for window_index in range(num_samples):
                self.index_map.append((subject_id, session_id, window_index))
            # Add the extra samples by sampling the index randomly without replacement
            # If we run out of samples, then just restart the process
            left_to_sample = num_samples_to_add
            while left_to_sample > 0:
                if left_to_sample > num_samples:
                    # Then we can just add the entire list again
                    for window_index in range(num_samples):
                        self.index_map.append((subject_id, session_id, window_index))
                    left_to_sample -= num_samples
                else:
                    # Then we need to randomly select a subset of the samples
                    to_add = list(range(num_samples))
                    np.random.shuffle(to_add)
                    for window_index in to_add[:left_to_sample]:
                        self.index_map.append((subject_id, session_id, window_index))
                    left_to_sample = 0

                    






def get_windowed_datasets(dataset_name: DatasetName, window_size_s: int, window_stride_s: int, subject_ids: list[SubjectID], target_freq: int | None = None, **kwargs) -> WindowedDataset:
    """
    Returns a map from subject id to a list of samples of the given time window size
    """
    full_dataset = MOABBDataset(dataset_name=dataset_name, subject_ids=subject_ids, **kwargs)
    windowed_datasets: WindowedDataset = {}
    subject_datasets = full_dataset.split('subject')

    dataset_freq = full_dataset.datasets[0].raw.info['sfreq']
    dataset_channels = full_dataset.datasets[0].raw.info['ch_names']
    window_size = int(window_size_s * dataset_freq)
    window_stride = int(window_stride_s * dataset_freq)

    resampler = Resample(orig_freq=dataset_freq, new_freq=target_freq) if target_freq is not None else None
    def process_sample(sample):
        # If the target frequency is not None and the dataset frequency is not the target frequency, then we need to downsample
        # We use torchaudio for this
        sample = torch.from_numpy(sample)
        if resampler is not None:
            sample = resampler(sample)
        return sample

    for subject_id, subject_dataset in subject_datasets.items():
        session_datasets = subject_dataset.split('session')
        for session_id, session_dataset in session_datasets.items():
            sliding_windows_dataset = create_fixed_length_windows(
                session_dataset, start_offset_samples=0, stop_offset_samples=None,
                window_size_samples=window_size, window_stride_samples=window_stride,
                drop_last_window=False
            )
            sliding_windows_dataset = [process_sample(d[0]) for d in sliding_windows_dataset]
            element = {
                'subject': subject_id,
                'session': session_id,
                'data': sliding_windows_dataset,  # Extract the raw data from the tuple
                'metadata': {
                    'channels': dataset_channels,
                    'freq': dataset_freq if target_freq is None else target_freq,
                    'window_size_s': window_size_s,
                    'window_stride_s': window_stride_s,
                    'dataset': dataset_name,
                }
            }
            windowed_datasets[(subject_id, session_id)] = element
    
    return windowed_datasets

def combine_datasets(windowed_dataset_set: Dict[DatasetName, WindowedDataset]) -> WindowedDataset:
    """
    Combines the datasets into a single dataset

    Converts the subject id to be the dataset name followed by the original subject id
    """
    # Check that all the dataset names are unique
    dataset_names = list(windowed_dataset_set.keys())
    assert len(dataset_names) == len(set(dataset_names)), "The dataset names must be unique"

    combined_dataset: WindowedDataset = {}
    for dataset_name, windowed_dataset in windowed_dataset_set.items():
        for (subject_id, session_id), element in windowed_dataset.items():
            combined_dataset[(dataset_name + '_' + subject_id, session_id)] = element
    return combined_dataset

def window_dataset_len(windowed_dataset: WindowedDataset) -> tuple[int, Dict[SubjectID, int]]:
    """
    Returns the number of windows in the dataset
    """
    total_size = 0
    size_per_subject = {}
    for (subject_id, session_id), element in windowed_dataset.items():
        if subject_id not in size_per_subject:
            size_per_subject[subject_id] = 0
        size_per_subject[subject_id] += len(element['data'])
        total_size += len(element['data'])
    return total_size, size_per_subject

def get_dataset_split(windowed_dataset: WindowedDataset, train_p, extrap_val_p, extrap_test_p, intra_val_p, intra_test_p, seed=0) -> SplitDataset:
    """
    Returns the dataset split into train and evaluation sets
    The evaluation set is further split into extrapolation and interpolation sets

    The extrapolation set only includes subjects never in the training set
    The interpolation set is samples from subjects seen in training, but the samples themselves were never seen
    """
    rng = np.random.default_rng(seed)

    subject_session_sample_map = {}
    for (subject_id, session_id), element in windowed_dataset.items():
        if subject_id not in subject_session_sample_map:
            subject_session_sample_map[subject_id] = {}
        subject_session_sample_map[subject_id][session_id] = element

    # Extract the subjects so that we can split them
    all_subject_ids = list(set(subject_session_sample_map.keys()))
    num_subjects = len(all_subject_ids)
    rng.shuffle(all_subject_ids)

    num_extrap_val_subjects = int(num_subjects * extrap_val_p)
    num_extrap_test_subjects = int(num_subjects * extrap_test_p)
    num_train_and_intra_subjects = num_subjects - num_extrap_val_subjects - num_extrap_test_subjects

    train_and_intra_subjects = all_subject_ids[:num_train_and_intra_subjects]
    extrap_val_subjects = all_subject_ids[num_train_and_intra_subjects:num_train_and_intra_subjects + num_extrap_val_subjects]
    extrap_test_subjects = all_subject_ids[num_train_and_intra_subjects + num_extrap_val_subjects:]

    # Construct the extrap sets
    extrap_val_set = {}
    extrap_test_set = {}
    for subject_id in extrap_val_subjects:
        for session_id, element in subject_session_sample_map[subject_id].items():
            extrap_val_set[(subject_id, session_id)] = element
    for subject_id in extrap_test_subjects:
        for session_id, element in subject_session_sample_map[subject_id].items():
            extrap_test_set[(subject_id, session_id)] = element

    # For the intra sets, we are going to take a fixed slice of the data for each subject
    # This is to ensure that the train and intra sets are disjoint due to the overlapping of adjacent windows
    train_set = {}
    intra_val_set = {}
    intra_test_set = {}
    for subject_id in train_and_intra_subjects:
        for session_id, element in subject_session_sample_map[subject_id].items():
            num_samples = len(element['data'])
            num_train_samples = int(num_samples * train_p)
            num_intra_val_samples = int(num_samples * intra_val_p)
            num_intra_test_samples = num_samples - num_train_samples - num_intra_val_samples

            train_set[(subject_id, session_id)] = {
                'subject': subject_id,
                'session': session_id,
                'data': element['data'][:num_train_samples],
                'metadata': element['metadata']
            }
            intra_val_set[(subject_id, session_id)] = {
                'subject': subject_id,
                'session': session_id,
                'data': element['data'][num_train_samples:num_train_samples + num_intra_val_samples],
                'metadata': element['metadata']
            }
            intra_test_set[(subject_id, session_id)] = {
                'subject': subject_id,
                'session': session_id,
                'data': element['data'][num_train_samples + num_intra_val_samples:],
                'metadata': element['metadata']
            }

    return {
        'train': train_set,
        'extrap_val': extrap_val_set,
        'extrap_test': extrap_test_set,
        'intra_val': intra_val_set,
        'intra_test': intra_test_set
    }



if __name__ == "__main__":
    # windowed_datasets_004 = get_windowed_datasets('BNCI2014_004', 5, 2, list(range(1, 9+1)))
    # windowed_datasets_001 = get_windowed_datasets('BNCI2014_002', 5, 2, list(range(1, 14+1)))
    # windowed_dataset = combine_datasets({'BNCI2014_004': windowed_datasets_004, 'BNCI2014_001': windowed_datasets_001})
    # split_dataset = get_dataset_split(windowed_dataset, 0.8, 0.05, 0.05, 0.05, 0.05)
    # print("Done")

    # physionet = get_windowed_datasets('PhysionetMI', 5, 2, list(range(1, 2+1)))
    # lee2019 = get_windowed_datasets('Lee2019_MI', 5, 2, list(range(1, 2+1)))
    # shin2017A = get_windowed_datasets('Shin2017A', 5, 2,  list(range(1, 2+1)), dataset_kwargs={"accept": True})
    # shin2017B = get_windowed_datasets('Shin2017B', 5, 2,  list(range(1, 2+1)), dataset_kwargs={"accept": True})
    # windowed_dataset = combine_datasets({
    #     'PhysionetMI': physionet,
    #     'Lee2019': lee2019
    # })
    # dataset_config = WindowedContrastiveDatasetConfig(
    #     target_freq=120,
    #     target_channels=None,
    #     toggle_direct_sum_on=True,
    #     channel_blacklist=[],
    #     sample_over_subjects_toggle=True
    # )
    # dataset = WindowedContrastiveDataset(windowed_dataset, dataset_config)
    # print("Done")

    lee2019 = get_windowed_datasets('Lee2019_MI', 5, 2, list(range(1, 3+1)), target_freq=120)
    size = window_dataset_len(lee2019)
    print("Done")