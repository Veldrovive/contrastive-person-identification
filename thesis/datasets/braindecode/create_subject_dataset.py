"""
Introduces utilities to easily convert a dataset from braindecode into a subject dataset
"""

import numpy as np
from braindecode.datasets.moabb import MOABBDataset
from braindecode.preprocessing import create_fixed_length_windows
from torchaudio.transforms import Resample
import torch
from typing import Dict
from thesis.datasets.subject_dataset import validate_dataset, SubjectDataset, SubjectID
import mne

BraindecodeDatasetName = str

def construct_braindecode_dataset(dataset_name: BraindecodeDatasetName, subject_ids: list[SubjectID], target_freq: int | None = None, load_subjects_independently: bool = False, band_pass: tuple[float, float] | None = None, **kwargs) -> Dict[SubjectID, SubjectDataset]:
    """
    Loads the given dataset from braindecode and converts it into a set of subject datasets, one for each subject
    This means each dataset will contain only the set of sessions for a single subject
    """
    datasets = {}

    if load_subjects_independently:
        braindecode_datasets = None
        braindecode_subject_datasets = None
    else:
        braindecode_datasets = MOABBDataset(dataset_name=dataset_name, subject_ids=subject_ids, **kwargs)
        braindecode_subject_datasets = braindecode_datasets.split("subject")
        # braindecode may change the subject ids to strings, so we need to get the subject ids from the dataset
        subject_ids = list(braindecode_subject_datasets.keys())
    def get_subject_dataset(subject_id):
        if load_subjects_independently:
            return MOABBDataset(dataset_name=dataset_name, subject_ids=[subject_id], **kwargs)
        else:
            return braindecode_subject_datasets[subject_id]

    def process_sample(from_freq, sample: np.ndarray, channels: list[str]):
        if band_pass is not None:
            ch_types = ['eeg' for _ in channels]
            info = mne.create_info(ch_names=channels, sfreq=from_freq, ch_types=ch_types)
            raw = mne.io.RawArray(sample, info)
            raw.filter(l_freq=band_pass[0], h_freq=band_pass[1], method='iir', iir_params={'order': 4, 'ftype': 'butter'})
            sample = raw.get_data()
        
        # If the target frequency is not None and the dataset frequency is not the target frequency, then we need to downsample
        # We use torchaudio for this
        resampler = Resample(orig_freq=from_freq, new_freq=target_freq) if target_freq is not None else None
        if resampler is not None:
            t = torch.from_numpy(sample.astype(np.float32))
            sample = resampler(t)
            sample = sample.numpy()
            from_freq = target_freq

        return sample

    for subject_id in subject_ids:
        print(f"Processing subject {subject_id}")
        braindecode_dataset = get_subject_dataset(subject_id)
        subject_id = str(subject_id)  # It needs to be an int for loading, but everything else acts as if it is a string so we convert it here
        dataset_freq = braindecode_dataset.datasets[0].raw.info['sfreq']
        dataset_channels = braindecode_dataset.datasets[0].raw.info['ch_names']
        session_datasets = braindecode_dataset.split('session')
        for session_id, session_dataset in session_datasets.items():
            runs_datasets = session_dataset.split('run')
            for run_index, run_dataset in enumerate(runs_datasets.values()):
                data = run_dataset.datasets[0].raw.get_data()
                data = process_sample(dataset_freq, data, dataset_channels)
                data_freq = dataset_freq if target_freq is None else target_freq
                metadata = {
                    'channels': dataset_channels,
                    'freq': data_freq
                }
                element = {
                    'subject': subject_id,
                    'session': session_id,
                    'run': run_index,
                    'data': data,
                    'metadata': metadata
                }

                if subject_id not in datasets:
                    datasets[subject_id] = {}
                datasets[subject_id][(subject_id, session_id, run_index)] = element

    for subject_id, subject_dataset in datasets.items():
        datasets[subject_id] = validate_dataset(subject_dataset)

    return datasets

if __name__ == "__main__":
    from thesis.datasets.subject_dataset import save_subject_dataset, save_subject_datasets, load_subject_dataset, load_subject_datasets
    from pathlib import Path

    
    test_dataset = construct_braindecode_dataset('PhysionetMI', list(range(1, 2+1)), target_freq=120, load_subjects_independently=True)
    subject_0_dataset = test_dataset['1']

    current_path = Path(__file__).parent.absolute()
    dataset_dir = current_path / 'test_datasets'
    dataset_dir.mkdir(exist_ok=True)
    save_subject_datasets(test_dataset, dataset_dir)

    recalled_dataset = load_subject_datasets(dataset_dir)


    pass
