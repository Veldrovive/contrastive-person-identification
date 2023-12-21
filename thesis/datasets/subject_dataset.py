"""
Defines the types and utilities for the subject dataset
"""

import os
import time
import json
import numpy as np
import torch
from pathlib import Path
from pydantic import BaseModel, RootModel, Field, validator, ValidationError
from typing import TypedDict, Dict, Tuple, List

from .mne_base import convert_dataset_to_nme, convert_dataset_to_subject
from thesis.structs.preprocessor_structs import LoadTimePreprocessorConfig

SubjectID = str
SessionID = str
RunIndex = int

class ElementMetadata(BaseModel):
    channels: List[str]
    freq: int

class ElementDict(BaseModel):
    subject: SubjectID
    session: SessionID
    run: RunIndex
    data: np.ndarray  # (channel, time)
    metadata: ElementMetadata

    @validator('data')
    def check_data_shape(cls, v, values, **kwargs):
        if v.ndim != 2:
            raise ValueError('data must be 2-dimensional (channel, time)')
        if 'metadata' in values and v.shape[0] != len(values['metadata'].channels):
            raise ValueError('data shape does not match number of channels')
        return v

    class Config:
        arbitrary_types_allowed = True  # allows np.ndarray to be used in models

SubjectDatasetKey = Tuple[SubjectID, SessionID, RunIndex]
SubjectDataset = RootModel[Dict[SubjectDatasetKey, ElementDict]]

def validate_dataset(dataset: dict) -> SubjectDataset:
    try:
        validated_dataset = SubjectDataset(dataset)
        return validated_dataset
    except ValidationError as e:
        raise e

def merge_datasets(datasets: List[SubjectDataset]) -> SubjectDataset:
    """
    Merges datasets subject-wise
    """
    merged_dataset = {}
    for dataset_index, dataset in enumerate(datasets):
        dataset_dict = dataset.dict()
        for (subject_id, session_id, run_index), element in dataset_dict.items():
            # In order to differentiate datasets, we add the dataset index to the session id
            new_session_id = f"{session_id}_{dataset_index}"
            element_dict_kwargs = { **element, "session": new_session_id }
            new_element = ElementDict(**element_dict_kwargs)
            merged_dataset[(subject_id, new_session_id, run_index)] = new_element
    return validate_dataset(merged_dataset)

def concatenate_datasets(datasets: Dict[str, Dict[str, SubjectDataset]]) -> Dict[str, SubjectDataset]:
    """
    Intended to be used to combine datasets loaded from load_subject_datasets
    So we expect conflicting subject ids that are resolved by appending the dataset key
    """
    concatenated_datasets = {}
    for dataset_name, subjects_dataset in datasets.items():
        for subject_id, subject_dataset in subjects_dataset.items():
            new_subject_id = f"{subject_id}_{dataset_name}"
            concatenated_datasets[new_subject_id] = subject_dataset
    return concatenated_datasets

def concatenate_metadata(metadata: Dict[str, Dict[str, dict | None]]) -> Dict[str, dict | None]:
    """
    Intended to be used to combine metadata loaded from load_subject_metadata
    So we expect conflicting subject ids that are resolved by appending the dataset key
    """
    concatenated_metadata = {}
    for dataset_name, subjects_metadata in metadata.items():
        for subject_id, subject_metadata in subjects_metadata.items():
            new_subject_id = f"{subject_id}_{dataset_name}"
            concatenated_metadata[new_subject_id] = subject_metadata
    return concatenated_metadata

def split_dataset_by_subject(dataset: SubjectDataset) -> Dict[SubjectID, SubjectDataset]:
    """
    Splits the dataset by subject
    """
    dataset_dict = dataset.dict()
    subject_datasets = {}
    for (subject_id, session_id), element in dataset_dict.items():
        if subject_id not in subject_datasets:
            subject_datasets[subject_id] = SubjectDataset(__root__={})
        subject_datasets[subject_id].__root__[(subject_id, session_id)] = element
    return subject_datasets

def get_dataset_split(dataset: Dict[str, SubjectDataset], train_p, extrap_val_p, extrap_test_p, intra_val_p, intra_test_p, window_size_samples: int, seed=0) -> Dict[str, Dict[str, SubjectDataset]]:
    """
    Returns the dataset split into train and evaluation sets
    The evaluation set is further split into extrapolation and interpolation sets

    The extrapolation set only includes subjects never in the training set
    The interpolation set is samples from subjects seen in training, but the samples themselves were never seen
    """
    # TODO: Use window_size_seconds instead of window_size_samples. Currently we assume that the full dataset has the same sample rate
    # but this is not necessarily true. Using time instead of samples will allow us to handle this case.
    rng = np.random.default_rng(seed)

    all_subject_ids = list(dataset.keys())
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
        extrap_val_set[subject_id] = dataset[subject_id]
    for subject_id in extrap_test_subjects:
        extrap_test_set[subject_id] = dataset[subject_id]

    # The intra sets are more difficult
    # Each dataset is a mapping from a key to a specific run
    # We are going to take a certain proportion of the total number of samples for each subject
    train_set = {  }
    intra_val_set = {  }
    intra_test_set = {  }

    # What I am going to do is dig one more level down and take samples from each session
    # The minimum size for these samples is 2 * window_size_samples
    # If it is less, then we print a warning and take the 2*window_size_samples. If this is more than the whole sample, we error
    for subject_id in train_and_intra_subjects:
        subject_dataset = dataset[subject_id]
        subject_dataset_dict = subject_dataset.model_dump()

        run_ids, runs = zip(*subject_dataset_dict.items())
        # Now we map from session id to a list of runs
        session_runs = {}
        for run_id, run in zip(run_ids, runs):
            session_id = run["session"]
            if session_id not in session_runs:
                session_runs[session_id] = []
            session_runs[session_id].append((run_id, run))

        for session_id, session_runs in session_runs.items():
            session_run_ids, session_runs = zip(*session_runs)
            session_run_lengths = [run["data"].shape[1] for run in session_runs]
            session_total_samples = np.sum(session_run_lengths)

            session_val_set_length = int(session_total_samples * intra_val_p)
            session_test_set_length = int(session_total_samples * intra_test_p)

            if session_val_set_length < 2 * window_size_samples and session_val_set_length != 0:
                # Note that if the val length is 0, this implies that the val prop was set to 0 which is an intentional choice
                print(f"Warning: session {session_id} for subject {subject_id} has less than 2 * window_size_samples samples for validation. {session_val_set_length} < {2 * window_size_samples} out of total {session_total_samples} samples")
                session_val_set_length = 2 * window_size_samples
            if session_test_set_length < 2 * window_size_samples and session_test_set_length != 0:
                # Similarly, if the test length is 0, this implies that the test prop was set to 0 which is an intentional choice
                print(f"Warning: session {session_id} for subject {subject_id} has less than 2 * window_size_samples samples for testing. {session_test_set_length} < {2 * window_size_samples} out of total {session_total_samples} samples")
                session_test_set_length = 2 * window_size_samples

            # We will take runs at random until we have enough samples
            # If the number of samples left is not an even run, then we take a proportion of that run and leave the rest for training
            run_order = np.arange(len(session_run_ids))
            rng.shuffle(run_order)
            remaining_val_samples = session_val_set_length
            remaining_test_samples = session_test_set_length

            for run_index in run_order:
                run_id = session_run_ids[run_index]
                run = session_runs[run_index]
                run_length = session_run_lengths[run_index]

                if remaining_val_samples > 0:
                    if subject_id not in intra_val_set:
                        intra_val_set[subject_id] = {}
                    if run_length <= remaining_val_samples:
                        intra_val_set[subject_id][run_id] = run
                        remaining_val_samples -= run_length
                        # Then we move on to the next run as this run has been fully used
                        continue
                    else:
                        # Then we need to split this run
                        # The first remaining_val_samples samples will be used for validation and the rest will be left for test or training
                        # To leave it for the next iteration, we overwrite run and run_length with the remaining samples
                        # And then just have an if statement instead of else if for the test set
                        val_data = run["data"][:, :remaining_val_samples]
                        remaining_data = run["data"][:, remaining_val_samples:]
                        val_run = { **run, "data": val_data }
                        intra_val_set[subject_id][run_id] = val_run

                        run = { **run, "data": remaining_data }
                        run_length = run["data"].shape[1]
                        remaining_val_samples = 0
                    
                if remaining_test_samples > 0:
                    if subject_id not in intra_test_set:
                        intra_test_set[subject_id] = {}
                    if run_length <= remaining_test_samples:
                        intra_test_set[subject_id][run_id] = run
                        remaining_test_samples -= run_length
                        # Then we move on to the next run as this run has been fully used
                        continue
                    else:
                        test_data = run["data"][:, :remaining_test_samples]
                        remaining_data = run["data"][:, remaining_test_samples:]
                        test_run = { **run, "data": test_data }
                        intra_test_set[subject_id][run_id] = test_run

                        run = { **run, "data": remaining_data }
                        run_length = run["data"].shape[1]
                        remaining_test_samples = 0

                if remaining_val_samples == 0 and remaining_test_samples == 0:
                    if subject_id not in train_set:
                        train_set[subject_id] = {}
                    train_set[subject_id][run_id] = run

        if subject_id in train_set:
            train_set[subject_id] = validate_dataset(train_set[subject_id])
        if subject_id in intra_val_set:
            intra_val_set[subject_id] = validate_dataset(intra_val_set[subject_id])
        if subject_id in intra_test_set:
            intra_test_set[subject_id] = validate_dataset(intra_test_set[subject_id])

    return {
        "train": train_set,
        "extrap_val": extrap_val_set,
        "extrap_test": extrap_test_set,
        "intra_val": intra_val_set,
        "intra_test": intra_test_set,
    }

def save_subject_dataset(dataset: SubjectDataset, path: str | Path):
    """
    Saves the dataset to the given path
    """
    if isinstance(path, Path):
        path = path.absolute().as_posix()

    # We demand that the file name ends with .pt
    if not path.endswith(".pt"):
        raise ValueError(f"File name must end with .pt, got {path}")

    dataset_dict = dataset.dict()
    torch.save(dataset_dict, path)

def save_subject_dataset_v2(dataset: SubjectDataset, path: str | Path):
    """
    Saves the dataset in the updated format to the given path
    """
    if isinstance(path, Path):
        path = path.absolute().as_posix()

    Path(path).mkdir(parents=True, exist_ok=True)

    for (subject_id, session_id, run_index), element in dataset.dict().items():
        # Save the data for the run as a numpy array and the metadata as a json file
        # We also include information about the number of samples and total time
        # The name of each file is {subject_id}_{session_id}_{run_index}.(npy|json)
        base_name_str = f"{subject_id}_{session_id}_{run_index}"
        data_path = os.path.join(path, f"{base_name_str}.npy")
        metadata_path = os.path.join(path, f"{base_name_str}.json")

        # Create the memory mapped array
        fp = np.memmap(data_path, dtype=element['data'].dtype, mode='w+', shape=element['data'].shape)
        fp[:] = element['data'][:]

        # In order to load a memory mapped, we must also excplicitly save the shape and dtype
        element['data_format'] = {}
        element['data_format']['shape'] = element['data'].shape
        element['data_format']['dtype'] = element['data'].dtype.name

        # We also include the total time in seconds
        element['data_format']['total_time_s'] = element['data'].shape[1] / element['metadata']['freq']

        # The metadata file is everything except the data
        del element['data']
        with open(metadata_path, "w") as f:
            json.dump(element, f, indent=4)


def save_subject_datasets(datasets: Dict[str, SubjectDataset], dir: str | Path):
    """
    Saves a list of datasets to the given directory

    Each is saved with the name: {dataset_key}.pt
    """
    if isinstance(dir, Path):
        dir = dir.absolute().as_posix()
    for dataset_key, dataset in datasets.items():
        save_subject_dataset(dataset, os.path.join(dir, f"{dataset_key}.pt"))

def save_subject_datasets_v2(datasets: Dict[str, SubjectDataset], dir: str | Path):
    """
    Saves a list of datasets to the given directory

    Each is saved in the subdir: /{dataset_key}
    """
    if isinstance(dir, Path):
        dir = dir.absolute().as_posix()
    for dataset_key, dataset in datasets.items():
        save_subject_dataset_v2(dataset, os.path.join(dir, dataset_key))

def load_subject_dataset(path: str | Path, load_time_preprocessor_config: LoadTimePreprocessorConfig = None) -> SubjectDataset:
    """
    Loads the dataset from the given path
    """
    if isinstance(path, Path):
        path = path.absolute().as_posix()
    try:
        dataset_dict = torch.load(path)
    except RuntimeError as e:
        print(f"Error loading dataset from {path}")
        raise e
    dataset = validate_dataset(dataset_dict)
    if load_time_preprocessor_config is not None:
        mne_datasets = convert_dataset_to_nme(dataset)
        if load_time_preprocessor_config.target_sample_rate is not None:
            # Then we check if the sample rate is different
            new_sample_rate = load_time_preprocessor_config.target_sample_rate
            for subject, mne_dataset in mne_datasets.items():
                assert mne_dataset["data"].info["sfreq"] >= new_sample_rate, f"Cannot upsample data. Current sample rate: {mne_dataset.info['sfreq']}, target sample rate: {new_sample_rate}"
                if mne_dataset["data"].info["sfreq"] != new_sample_rate:
                    mne_dataset["data"].resample(load_time_preprocessor_config.target_sample_rate)
        
        lower_cutoff = load_time_preprocessor_config.band_pass_lower_cutoff
        upper_cutoff = load_time_preprocessor_config.band_pass_upper_cutoff
        if lower_cutoff is not None or upper_cutoff is not None:
            for subject, mne_dataset in mne_datasets.items():
                mne_dataset["data"].filter(l_freq=lower_cutoff, h_freq=upper_cutoff)
        dataset = validate_dataset(convert_dataset_to_subject(mne_datasets))
    return dataset

def load_subject_dataset_v2(path: str | Path, load_time_preprocessor_config: LoadTimePreprocessorConfig | None = None, memory_map: bool = False) -> SubjectDataset:
    """
    Loads the dataset from the given directory

    If memory_map is True, then the data will be loaded in numpy memory maps
    In memory mapped mode, the load time preprocessor must be None or an error will be raised
    """
    if isinstance(path, Path):
        path = path.absolute().as_posix()
    
    dataset = {}
    file_roots = set([Path(file).stem for file in os.listdir(path)])
    for file_root in file_roots:
        # For each root, we construct a run 
        with open(os.path.join(path, f"{file_root}.json"), "r") as f:
            metadata = json.load(f)

        data_format = metadata['data_format']
        del metadata['data_format']

        data_path = os.path.join(path, f"{file_root}.npy")
        if memory_map:
            assert load_time_preprocessor_config is None, "Cannot use memory map mode with load time preprocessing"
            data = np.memmap(data_path, dtype=data_format['dtype'], mode='r', shape=tuple(data_format['shape']))
        else:
            data = np.memmap(data_path, dtype=data_format['dtype'], mode='r', shape=tuple(data_format['shape']))
            data = np.array(data)  # Convert to a normal numpy array

        # The dataset key can be constructed using metadata file ({meta.subject}, {meta.session}, {meta.run})
        dataset_key = (metadata['subject'], metadata['session'], metadata['run'])
        dataset[dataset_key] = ElementDict(data=data, **metadata)
    
    dataset = validate_dataset(dataset)
    if load_time_preprocessor_config is not None:
        mne_datasets = convert_dataset_to_nme(dataset)
        if load_time_preprocessor_config.target_sample_rate is not None:
            # Then we check if the sample rate is different
            new_sample_rate = load_time_preprocessor_config.target_sample_rate
            for subject, mne_dataset in mne_datasets.items():
                assert mne_dataset["data"].info["sfreq"] >= new_sample_rate, f"Cannot upsample data. Current sample rate: {mne_dataset.info['sfreq']}, target sample rate: {new_sample_rate}"
                if mne_dataset["data"].info["sfreq"] != new_sample_rate:
                    mne_dataset["data"].resample(load_time_preprocessor_config.target_sample_rate)
        
        lower_cutoff = load_time_preprocessor_config.band_pass_lower_cutoff
        upper_cutoff = load_time_preprocessor_config.band_pass_upper_cutoff
        if lower_cutoff is not None or upper_cutoff is not None:
            for subject, mne_dataset in mne_datasets.items():
                mne_dataset["data"].filter(l_freq=lower_cutoff, h_freq=upper_cutoff)
        dataset = validate_dataset(convert_dataset_to_subject(mne_datasets))
    return dataset


def get_subject_datasets_in_dir(dir: str | Path) -> list[str]:
    """
    Returns the dataset keys for all subjects in the given directory
    """
    if isinstance(dir, Path):
        dir = dir.absolute().as_posix()
    dataset_keys = []
    for file in os.listdir(dir):
        if file.endswith(".pt"):
            dataset_key = file[:-3]
            dataset_keys.append(dataset_key)
    return dataset_keys

def get_subject_datasets_in_dir_v2(dir: str | Path) -> list[str]:
    """
    Returns the dataset keys for all subjects in the given directory
    """
    if isinstance(dir, Path):
        dir = dir.absolute().as_posix()
    dataset_keys = []
    for file in os.listdir(dir):
        if os.path.isdir(os.path.join(dir, file)):
            dataset_key = file
            dataset_keys.append(dataset_key)
    return dataset_keys

def load_subject_datasets(dir: str, max_subjects: int | None = None, subjects: list[str] | None = None, load_time_preprocessor_config: LoadTimePreprocessorConfig = None) -> Dict[str, SubjectDataset]:
    """
    Loads a list of datasets from the given directory

    Each is loaded with the name: {dataset_key}.pt
    """
    datasets = {}
    start_time = time.time()
    for file in os.listdir(dir):
        if file.endswith(".pt"):
            dataset_key = file[:-3]
            if subjects is None or dataset_key in subjects:
                datasets[dataset_key] = load_subject_dataset(os.path.join(dir, file), load_time_preprocessor_config)
        if max_subjects is not None and len(datasets) >= max_subjects:
            break
    end_time = time.time()
    print(f"Loaded {len(datasets)} datasets in {round(end_time - start_time, 2)} seconds")
    return datasets

def load_subject_datasets_v2(dir: str, max_subjects: int | None = None, subjects: list[str] | None = None, load_time_preprocessor_config: LoadTimePreprocessorConfig | None = None, memory_map: bool = False) -> Dict[str, SubjectDataset]:
    """
    Loads a list of datasets from the given directory

    Each is loaded in the subdir: /{dataset_key}
    """
    datasets = {}
    start_time = time.time()
    for file in os.listdir(dir):
        if os.path.isdir(os.path.join(dir, file)):
            dataset_key = file
            if subjects is None or dataset_key in subjects:
                datasets[dataset_key] = load_subject_dataset_v2(os.path.join(dir, file), load_time_preprocessor_config, memory_map)
        if max_subjects is not None and len(datasets) >= max_subjects:
            break
    end_time = time.time()
    print(f"Loaded {len(datasets)} datasets in {round(end_time - start_time, 2)} seconds")
    return datasets

def load_subject_metadata(dir: str, subjects: list[str]) -> Dict[str, dict | None]:
    """
    Loads the metadata files for the given subjects.
    Expects files to be {dataset_key}.json

    if the metadata file does not exist, then the value will be None
    """
    metadata = {}
    for subject in subjects:
        metadata_path = os.path.join(dir, f"{subject}.json")
        if os.path.exists(metadata_path):
            with open(metadata_path, "r") as f:
                metadata[subject] = json.load(f)
        else:
            metadata[subject] = None
    return metadata
