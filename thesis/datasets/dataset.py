"""
The main dataset class for the contrastive learning experiments
"""

from typing import Any, Callable, Optional, Tuple, Union, Dict, List
from thesis.datasets.subject_dataset import SubjectDataset, SubjectID, SessionID, RunIndex
from thesis.structs.dataset_structs import ContrastiveSubjectDatasetConfig, DataloaderConfig
from thesis.preprocessing import PreprocessorConfig, construct_preprocess_fn
from thesis.augmentation import AugmentationConfig, construct_augmentation_fn
import numpy as np
import torch
from torch.utils.data import Dataset
import time
import mne
import matplotlib.pyplot as plt

class ContrastiveSubjectDataset(Dataset):
    def __init__(
        self,
        datasets: Dict[str, SubjectDataset],
        config: ContrastiveSubjectDatasetConfig,
        *args, **kwargs
    ):
        """
        Parameters:
            datasets: A dictionary mapping a unique subject id to subject dataset. Each subject dataset is assumed to only contain one subject with multiple sessions.
        """
        self.config = config
        self.preprocessor_config = config.preprocessor_config
        self.augmentation_config = config.augmentation_config
        self.preprocess_fn = None
        self.augmentation_fn = None

        self.n_pos = config.n_pos
        self.n_neg = config.n_neg

        self.datasets = datasets
        self.verify_dataset_compatibility()
        self.channels = self.select_channels()
        self.channel_to_index_map = {channel: index for index, channel in enumerate(self.channels)}

        self.unique_subjects = set(datasets.keys())
        self.window_size_s = self.config.window_size_s
        self.window_stride_s = self.config.window_stride_s

        self.max_samples_per_subject = config.max_samples_per_subject
        self.anchors = self.compute_anchors()
        self.subject_anchor_map = {}
        for anchor in self.anchors:
            unique_subject_id, _, _ = anchor
            if unique_subject_id not in self.subject_anchor_map:
                self.subject_anchor_map[unique_subject_id] = []
            self.subject_anchor_map[unique_subject_id].append(anchor)

        self.get_sample_time = 0
        self.channel_process_sample_time = 0
        self.preprocess_sample_time = 0

    def to_equated_channels(self, channels: List[str]) -> List[str]:
        """
        Converts the given channels to the equated channels using the config.channel_equality_map

        This allows us to use 10/20 channels with 10/10 channels
        """
        channel_equality_map = {mapping[1]: mapping[0] for mapping in self.config.channel_equality_map}
        return [channel_equality_map.get(channel, channel) for channel in channels]

    def select_channels(self) -> List[str]:
        """
        Uses the set of datasets to select the channels to use.

        If config.target_channels is given, we use those
        If config.toggle_direct_sum_on is false, then we use the intersection of all channels
        If config.toggle_direct_sum_on is true, then we use the direct sum of all channels
        """
        # If the target channels are explicitly set we can just use those
        if self.config.target_channels is not None:
            return self.config.target_channels

        # To start, we build a list of all channels sets that we see in the dataset
        all_channel_sets = list()
        for unique_subject_id, dataset in self.datasets.items():
            dataset = dataset.model_dump()
            for element_key, element in dataset.items():
                all_channel_sets.append(set(self.to_equated_channels(element['metadata']["channels"])))

        # We then build the channel set we will use
        if self.config.toggle_direct_sum_on:
            # If we are using the direct sum, we just use the union of all channel sets
            channel_set = set.union(*all_channel_sets)
        else:
            # If we are using the intersection, we use the intersection of all channel sets
            channel_set = set.intersection(*all_channel_sets)

        # We then remove any channels that are blacklisted
        channel_set = channel_set.difference(self.config.channel_blacklist)
        
        channels = list(channel_set)
        channels.sort()

        return channels

    def process_sample_channels(self, sample_channels: list[str], sample: np.ndarray | torch.Tensor) -> np.ndarray | torch.Tensor:
        """
        Takes the old sample of shape (old_channels, timesteps) and converts the channels to the self.channels
        by rearranging the channels and introducing zeros where there is a new channel but no old channel
        """
        new_indices = [self.channel_to_index_map.get(channel, -1) for channel in sample_channels]
        if isinstance(sample, np.ndarray):
            new_sample = np.zeros((len(self.channels), sample.shape[1]))

            # for new_index, old_index in enumerate(new_indices):
            #     if old_index != -1:
            #         new_sample[new_index] = sample[old_index]

            # List comprehensions, zip, and list unpacking are faster than the above for loop
            valid_indices = [(new_index, old_index) for old_index, new_index in enumerate(new_indices) if old_index != -1]
            new_indices, old_indices = zip(*valid_indices)
            new_sample[list(new_indices)] = sample[list(old_indices)]

            # We could use np.where to accelerate the numpy further, but I feel like I'll just introduce bugs
            # TODO: Maybe accelerate this
        elif isinstance(sample, torch.Tensor):
            # Torch should be faster still since it has better indexing for this task
            new_indices = torch.tensor(new_indices)
            new_sample = torch.zeros((len(self.channels), sample.shape[1]), dtype=sample.dtype, device=sample.device)
            valid_mask = new_indices != -1
            new_sample[valid_mask] = sample[new_indices[valid_mask]]

        return new_sample

    def compute_window_parameters(self, freq: int) -> tuple[int, int]:
        """
        Computes the number of samples in a single window given the frequency as well as the stride in samples

        Frequency is given in Hz
        """
        return int(self.window_size_s * freq), int(self.window_stride_s * freq)

    def verify_dataset_compatibility(self):
        """
        Verifies that all datasets are compatible with each other

        Ensures that all elements of all datasets have the same frequency
        We handle channels separately
        """
        # Verify that all datasets have the same frequency
        self.freq = None
        for unique_subject_id, dataset in self.datasets.items():
            dataset = dataset.model_dump()
            for dataset_key, element in dataset.items():
                metadata = element['metadata']
                if self.freq is None:
                    self.freq = metadata['freq']
                else:
                    assert self.freq == metadata['freq'], "All datasets must have the same frequency"
        # Verify that all datasets have one subject
        for unique_subject_id, dataset in self.datasets.items():
            dataset = dataset.model_dump()
            dataset_subjects = [subject_id for subject_id, _, _ in dataset.keys()]
            assert len(set(dataset_subjects)) == 1, "All datasets must have one subject"

    def compute_anchors(self) -> List[Tuple[str, Tuple[SubjectID, SessionID, RunIndex], int]]:
        """
        Computes the full set of all anchors in the dataset

        Each element of the list contains the unique subject id, the dataset key, and the start timestep

        TODO: Implement even sampling over subjects
        """
        # First, we make a mapping from the unique subject id to a list of all anchors for that subject
        unique_subject_anchors: Dict[str, list[tuple[str, (SubjectID, SessionID, RunIndex), int]]] = {}
        for unique_subject_id, dataset in self.datasets.items():
            dataset = dataset.model_dump()
            for dataset_key, element in dataset.items():
                num_samples = element['data'].shape[1]
                metadata = element['metadata']
                num_samples_per_window, num_samples_per_stride = self.compute_window_parameters(metadata['freq'])
                for start_timestep in range(0, num_samples - num_samples_per_window, num_samples_per_stride):
                    # anchors.append((unique_subject_id, dataset_key, start_timestep))
                    if unique_subject_id not in unique_subject_anchors:
                        unique_subject_anchors[unique_subject_id] = []
                    if self.max_samples_per_subject is None or len(unique_subject_anchors[unique_subject_id]) < self.max_samples_per_subject:
                        unique_subject_anchors[unique_subject_id].append((unique_subject_id, dataset_key, start_timestep))
        
        # Then, if we are not re-sampling over subjects, we can just return the list of anchors concatenated
        if not self.config.sample_over_subjects_toggle:
            anchors = []
            for _, anchor_list in unique_subject_anchors.items():
                anchors.extend(anchor_list)
            return anchors
        else:
            # But if we are, we need to extend the list of anchors for each subject such that they have the same number as the maximum
            # We do this by randomly sampling from the list of anchors for each subject
            max_num_anchors = max([len(anchor_list) for _, anchor_list in unique_subject_anchors.items()])
            anchors = []
            rng = np.random.default_rng(self.config.random_seed)
            for _, anchor_list in unique_subject_anchors.items():
                num_anchors_to_add = max_num_anchors - len(anchor_list)
                anchors.extend(anchor_list)

                while num_anchors_to_add > 0:
                    if num_anchors_to_add > len(anchor_list):
                        # Add the entire list again
                        anchors.extend(anchor_list)
                        num_anchors_to_add -= len(anchor_list)
                    else:
                        # Add a random sample of the list
                        to_add = rng.choice(len(anchor_list), size=num_anchors_to_add, replace=False)
                        anchors.extend([anchor_list[index] for index in to_add])
                        num_anchors_to_add = 0
            return anchors
        
    def __len__(self):
        return len(self.anchors)

    def get_element(self, unique_subject_id: str, dataset_key: Tuple[SubjectID, SessionID, RunIndex], start_timestep: int) -> np.ndarray:
        """
        Gets the element from the dataset given the unique subject id, dataset key, and start timestep and
        extracts the window from the element
        """
        dataset = self.datasets[unique_subject_id]
        dataset = dataset.model_dump()
        element = dataset[dataset_key]
        data = element['data']
        metadata = element['metadata']
        num_samples_per_window, num_samples_per_stride = self.compute_window_parameters(metadata['freq'])
        window = data[:, start_timestep:start_timestep + num_samples_per_window]
        channels = self.to_equated_channels(metadata['channels'])
        return channels, window

    def description(self):
        """
        Prints a description of the dataset
        """
        length = len(self)
        # Get the number of unique subjects
        num_unique_subjects = len(self.unique_subjects)
        # Get the median number of anchors per subject and the IQR
        num_anchors_per_subject = [len(anchor_list) for _, anchor_list in self.subject_anchor_map.items()]
        median_num_anchors_per_subject = np.median(num_anchors_per_subject)
        iqr_num_anchors_per_subject = np.subtract(*np.percentile(num_anchors_per_subject, [75, 25]))
        max_num_anchors_per_subject = max(num_anchors_per_subject)
        min_num_anchors_per_subject = min(num_anchors_per_subject)

        num_channels = len(self.channels)

        window_size_samples, window_stride_samples = self.compute_window_parameters(self.freq)

        print(f"Dataset description:")
        print(f"\t----------")
        print(f"\tSampling over subjects: {self.config.sample_over_subjects_toggle}")
        print(f"\tLength: {length}")
        print(f"\tNumber of unique subjects: {num_unique_subjects}")
        print(f"\tMedian number of anchors per subject: {median_num_anchors_per_subject}")
        print(f"\tIQR number of anchors per subject: {iqr_num_anchors_per_subject}")
        print(f"\tRange number of anchors per subject: {max_num_anchors_per_subject - min_num_anchors_per_subject} ({max_num_anchors_per_subject} - {min_num_anchors_per_subject})")
        print(f"\t----------")
        print(f"\tUsing channel direct sum: {self.config.toggle_direct_sum_on}")
        print(f"\tNumber of channels: {num_channels}")
        print(f"\tChannels: {self.channels}")
        print(f"\tFrequency: {self.freq}")
        print(f"\tWindow size: {self.window_size_s} ({window_size_samples} samples)")
        print(f"\tWindow stride: {self.window_stride_s} ({window_stride_samples} samples)")
        print(f"\t----------")
        print(f"\tTotal time spent getting samples: {self.get_sample_time}")
        print(f"\tTotal time spent processing samples for channels: {self.channel_process_sample_time}")
        print(f"\tTotal time spent preprocessing samples: {self.preprocess_sample_time}")

        return {
            "length": length,
            "num_unique_subjects": num_unique_subjects,
            "median_num_anchors_per_subject": median_num_anchors_per_subject,
            "iqr_num_anchors_per_subject": iqr_num_anchors_per_subject,
            "max_num_anchors_per_subject": max_num_anchors_per_subject,
            "min_num_anchors_per_subject": min_num_anchors_per_subject,
            "num_channels": num_channels,
            "channels": self.channels,
            "freq": self.freq,
        }

    def visualize_item(self, index):
        """
        Uses MNE-python to visualize the item
        """
        item = self.__getitem__(index, testing=True)
        raw_anchor = item['raw_anchor']
        preprocessed_anchor = item['preprocessed_anchor']
        augmented_anchor = item['augmented_anchor']
        augmentation_residuals = augmented_anchor - preprocessed_anchor

        ch_types = ['eeg' for _ in self.channels]
        info = mne.create_info(ch_names=self.channels, sfreq=self.freq, ch_types=ch_types)
        raw_anchor = mne.io.RawArray(raw_anchor, info)
        preprocessed_anchor = mne.io.RawArray(preprocessed_anchor, info)
        augmented_anchor = mne.io.RawArray(augmented_anchor, info)
        augmentation_residuals = mne.io.RawArray(augmentation_residuals, info)

        raw_anchor.plot(n_channels=10, scalings='auto', duration=self.window_size_s, show=False, title="Raw Anchor")
        preprocessed_anchor.plot(n_channels=10, scalings='auto', duration=self.window_size_s, show=False, title="Preprocessed Anchor")
        augmented_anchor.plot(n_channels=10, scalings='auto', duration=self.window_size_s, show=False, title="Augmented Anchor")
        augmentation_residuals.plot(n_channels=10, scalings='auto', duration=self.window_size_s, show=False, title="Augmentation Residuals")

        plt.show()

    def __getitem__(self, index: int, testing: bool = False):
        """
        Returns a single sample from the dataset

        Anchor is selected based on index.
        """
        # Create the preprocess function is required
        if self.preprocess_fn is None and self.preprocessor_config is not None:
            self.preprocess_fn = construct_preprocess_fn(self.preprocessor_config)
        if self.augmentation_fn is None and self.augmentation_config is not None:
            self.augmentation_fn = construct_augmentation_fn(self.augmentation_config)
        attempt_preprocess = lambda data: self.preprocess_fn(data) if self.preprocess_fn is not None else data
        attempt_transform = lambda data: self.augmentation_fn(data) if self.augmentation_fn is not None else data
        process_data = lambda data: attempt_transform(attempt_preprocess(data))

        anchor = self.anchors[index]
        anchor_unique_subject_id, anchor_dataset_key, anchor_start_timestep = anchor
        _, anchor_session, _ = anchor_dataset_key

        # We select the anchor sample
        anchor_sample = self.get_element(anchor_unique_subject_id, anchor_dataset_key, anchor_start_timestep)
        anchor_channels, raw_anchor_window = anchor_sample
        channel_corrected_anchor_window = self.process_sample_channels(anchor_channels, raw_anchor_window)
        if testing:
            # Then we process the data in steps and return each of them
            preprocessed_anchor_window = attempt_preprocess(channel_corrected_anchor_window)
            augmented_anchor_window = attempt_transform(preprocessed_anchor_window)
            return {
                'raw_anchor': channel_corrected_anchor_window,
                'preprocessed_anchor': preprocessed_anchor_window,
                'augmented_anchor': augmented_anchor_window,
            }
        
        # If we aren't testing then just process the data as usual
        anchor_window = process_data(channel_corrected_anchor_window)
        anchor_metadata = {
            'unique_subject_id': anchor_unique_subject_id,
            'dataset_key': anchor_dataset_key,
            'start_timestep': anchor_start_timestep,
        }

        # We first select the positive samples
        positive_set = self.subject_anchor_map[anchor_unique_subject_id]
        if self.config.positive_separate_session:
            # We restrict the positive set to samples where the session is different from the anchor session
            restricted_positive_set = [positive for positive in positive_set if positive[1][1] != anchor_session]
            if len(restricted_positive_set) == 0:
                if self.config.error_on_no_separate_session:
                    raise ValueError("No separate session for positive samples")
                else:
                    # If there are no separate sessions, we just use the same session
                    restricted_positive_set = positive_set
                    # TODO: Remove the anchor itself in this case.
            positive_set = restricted_positive_set
        
        positive_samples = []
        positive_metadata = []
        for _ in range(self.n_pos):
            positive = positive_set[np.random.choice(len(positive_set))]
            positive_unique_subject_id, positive_dataset_key, positive_start_timestep = positive
            positive_sample = self.get_element(positive_unique_subject_id, positive_dataset_key, positive_start_timestep)
            positive_channels, raw_positive_window = positive_sample
            positive_window = self.process_sample_channels(positive_channels, raw_positive_window)
            positive_window = process_data(positive_window)
            positive_samples.append(positive_window)
            positive_metadata.append({
                'unique_subject_id': positive_unique_subject_id,
                'dataset_key': positive_dataset_key,
                'start_timestep': positive_start_timestep,
            })
        positive_window = positive_samples[0] if self.n_pos == 1 else positive_samples
        positive_metadata = positive_metadata[0] if self.n_pos == 1 else positive_metadata

        # Then we select the negative samples
        negative_samples = []
        negative_metadata = []
        negative_subject_ids = list(self.unique_subjects.difference({anchor_unique_subject_id}))
        for _ in range(self.n_neg):
            # The negative sample can be from any subject other than the anchor subject
            negative_subject_id = np.random.choice(negative_subject_ids)
            # We then select a random sample from the negative subject
            negative_set = self.subject_anchor_map[negative_subject_id]
            negative = negative_set[np.random.choice(len(negative_set))]
            negative_unique_subject_id, negative_dataset_key, negative_start_timestep = negative
            negative_sample = self.get_element(negative_unique_subject_id, negative_dataset_key, negative_start_timestep)
            negative_channels, raw_negative_window = negative_sample
            negative_window = self.process_sample_channels(negative_channels, raw_negative_window)
            negative_window = process_data(negative_window)
            negative_samples.append(negative_window)
            negative_metadata.append({
                'unique_subject_id': negative_unique_subject_id,
                'dataset_key': negative_dataset_key,
                'start_timestep': negative_start_timestep,
            })
        negative_window = negative_samples[0] if self.n_neg == 1 else negative_samples
        negative_metadata = negative_metadata[0] if self.n_neg == 1 else negative_metadata

        return {
            'anchor': anchor_window,
            'anchor_metadata': anchor_metadata,
            'positive': positive_window,
            'positive_metadata': positive_metadata,
            'negative': negative_window,
            'negative_metadata': negative_metadata,
        }

def subject_dataset_collate(samples):
    """
    Collate function for the dataloader
    """
    batch = {}
    for key in samples[0].keys():
        if key.endswith("_metadata"):
            batch[key] = [sample[key] for sample in samples]
        else:
            batch[key] = torch.from_numpy(np.array([sample[key] for sample in samples]))
    return batch

def get_contrastive_subject_loader(dataset: ContrastiveSubjectDataset, batch_size: int, shuffle: bool, num_workers: int, *args, **kwargs):
    """
    Gets a dataloader for the given dataset
    """
    return torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, collate_fn=subject_dataset_collate, *args, **kwargs)

def get_contrastive_subject_loaders(datasets: Dict[str, ContrastiveSubjectDataset], config: DataloaderConfig, *args, **kwargs):
    """
    Gets the dataloaders for the given datasets
    """
    loaders = {}
    for dataset_name, dataset in datasets.items():
        if dataset is not None:
            is_train = dataset_name == "train"
            num_workers = config.num_workers_train if is_train else config.num_workers_eval
            loaders[dataset_name] = get_contrastive_subject_loader(dataset, config.batch_size_train, config.shuffle_train, num_workers, *args, **kwargs)
        else:
            loaders[dataset_name] = None
    return loaders
    

if __name__ == "__main__":
    from thesis.datasets.subject_dataset import load_subject_datasets, concatenate_datasets, get_dataset_split
    from thesis.preprocessing.meta import preprocessor_factory, MetaPreprocessorConfig
    from pathlib import Path

    dataset_folder = Path(__file__).parent / "braindecode" / "datasets"

    print("Loading Physionet dataset")
    physionet_dataset = load_subject_datasets(dataset_folder / "physionet")

    # print("Loading Shin dataset")
    # shin_dataset = load_subject_datasets(dataset_folder / "shin")

    # print("Loading Lee2019 dataset")
    # lee2019_dataset = load_subject_datasets(dataset_folder / "lee2019")

    print("Concatenating datasets")
    datasets = concatenate_datasets({
        "physionet": physionet_dataset,
        # "shin": shin_dataset,
        # "lee2019": lee2019_dataset,
    })

    split_datasets = get_dataset_split(datasets, 0.9, 0.05, 0.05, 0.05, 0.05, seed=0)

    test_preprocess_config = MetaPreprocessorConfig(
        sample_rate=120,
        target_sample_rate=None,
        use_baseline_correction=True,
        use_robust_scaler=True,
        use_clamping=True
    )
    preprocess_fn, print_preprocess_times = preprocessor_factory(test_preprocess_config, track_times=True)
    
    dataset_config = ContrastiveSubjectDatasetConfig(
        toggle_direct_sum_on=False,

        window_size_s=5,
        window_stride_s=2,

        sample_over_subjects_toggle=False,
    )
    contrastive_datasets = {key: ContrastiveSubjectDataset(dataset, dataset_config, preprocess_fn=preprocess_fn) for key, dataset in split_datasets.items()}
    contrastive_dataset = ContrastiveSubjectDataset(split_datasets["train"], dataset_config, preprocess_fn=preprocess_fn, n_pos=2, n_neg=0)

    contrastive_dataset.description()
    test_sample = contrastive_dataset[0]
    
    loaders = get_contrastive_subject_loaders(contrastive_datasets, DataloaderConfig(batch_size_train=64, shuffle_train=True, num_workers_train=0))
    test_loader = loaders["train"]
    
    count = 0
    for batch in test_loader:
        print(batch['anchor'].shape)
        print(batch['positive'].shape)
        print(batch['negative'].shape)
        count += 1
        if count > 10:
            break

    test_loader.dataset.description()
    print_preprocess_times()