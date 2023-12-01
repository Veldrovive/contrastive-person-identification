import mne
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from thesis.datasets.subject_dataset import SubjectDataset

def convert_dataset_to_nme(dataset: 'SubjectDataset'):
    """
    Returns a dict with the same structure as a subject dataset, but where data is an nme object instead of a numpy array
    """
    dataset = dataset.model_dump()
    for dataset_key, elem in dataset.items():
        channels = elem['metadata']['channels']
        freq = elem['metadata']['freq']
        ch_types = ['eeg' for _ in channels]
        info = mne.create_info(ch_names=channels, sfreq=freq, ch_types=ch_types)
        raw = mne.io.RawArray(elem['data'], info)
        elem['data'] = raw
    return dataset

def convert_dataset_to_subject(dataset: dict) -> dict:
    """
    Converts from mne back to a subject dataset
    """
    for dataset_key, elem in dataset.items():
        # Get the frequency and channels
        freq = elem['data'].info['sfreq']
        channels = elem['data'].info['ch_names']
        # And update the metadata
        elem['metadata']['freq'] = freq
        elem['metadata']['channels'] = channels
        # Then convert the data back to a numpy array
        elem['data'] = elem['data'].get_data()
    return dataset