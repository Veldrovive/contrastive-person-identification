"""
This module allows for the creation of datasets from the MOABB dataset.
"""

from pathlib import Path
import os
import shutil
from thesis.datasets.braindecode.create_subject_dataset import BraindecodeDatasetName, construct_braindecode_dataset
from thesis.datasets.subject_dataset import SubjectDataset, SubjectID, save_subject_datasets, merge_datasets

def convert_dataset(
    save_dir: str | Path,
    dataset_name: BraindecodeDatasetName | list[BraindecodeDatasetName],
    subject_ids: list[SubjectID],
    target_freq: int | None = None,
    load_subjects_independently: bool = False,
    **kwargs
):
    """
    Converts the braindecode dataset into a set of subject datasets, one for each subject
    and then saves them to the given directory

    The save dir should be unique across datasets, as it will be used to name the dataset
    """
    Path(save_dir).mkdir(parents=True, exist_ok=True)
    if isinstance(dataset_name, list):
        dataset_list = []
        for dataset in dataset_name:
            dataset_list.append(construct_braindecode_dataset(dataset, subject_ids, target_freq, load_subjects_independently, **kwargs))
        # Merge the datasets by subject
        # First we make sure that every dataset has the same subjects
        subjects = set()
        for dataset in dataset_list:
            subjects.update(dataset.keys())
        for dataset in dataset_list:
            assert set(dataset.keys()) == subjects, "Every dataset must have the same subjects"
        # Then we merge them
        merged_dataset = {}
        for subject_id in subjects:
            subject_datasets = [dataset[subject_id] for dataset in dataset_list]
            merged_dataset[subject_id] = merge_datasets(subject_datasets)
        save_subject_datasets(merged_dataset, save_dir)
    else:
        datasets = construct_braindecode_dataset(dataset_name, subject_ids, target_freq, load_subjects_independently, **kwargs)
        save_subject_datasets(datasets, save_dir)

def delete_braindecode_local_copy(local_storage_id: str, data_dir = "~/mne_data"):
    """
    Removes braindecode's local copy of the dataset

    Note that dataset names do not necessarily match the local storage ids
    """
    data_dir = os.path.expanduser(data_dir)
    dataset_dir = os.path.join(data_dir, local_storage_id)
    if os.path.exists(dataset_dir):
        shutil.rmtree(dataset_dir)

def construct_physionet_dataset(save_dir_parent: str | Path, subject_ids: list[SubjectID], target_freq: int | None = None, delete_after: bool = False):
    """
    Constructs the physionet dataset from the given subject ids

    Automatically saved with the path (save_dir_parent / "physionet")
    """
    print("Constructing physionet dataset")
    if isinstance(save_dir_parent, Path):
        save_dir_parent = save_dir_parent.absolute().as_posix()
    save_dir = os.path.join(save_dir_parent, f"physionet_{target_freq}")
    convert_dataset(save_dir, "PhysionetMI", subject_ids, target_freq, load_subjects_independently=False)

    if delete_after:
        delete_braindecode_local_copy("MNE-eegbci-data")

def construct_lee2019_dataset(save_dir_parent: str | Path, subject_ids: list[SubjectID], target_freq: int | None = None, delete_after: bool = False):
    """
    Constructs the lee2019 dataset from the given subject ids

    Automatically saved with the path (save_dir_parent / "lee2019")
    """
    print("Constructing lee2019 dataset")
    if isinstance(save_dir_parent, Path):
        save_dir_parent = save_dir_parent.absolute().as_posix()
    save_dir = os.path.join(save_dir_parent, f"lee2019_{target_freq}")
    convert_dataset(save_dir, "Lee2019_MI", subject_ids, target_freq, load_subjects_independently=True)

    if delete_after:
        delete_braindecode_local_copy("MNE-lee2019-mi-data")

def construct_filtered_lee2019_dataset(save_dir_parent: str | Path, subject_ids: list[SubjectID], target_freq: int | None = None, delete_after: bool = False):
    """
    Currently testing using mne for filtering
    """
    print("Constructing lee2019 dataset")
    if isinstance(save_dir_parent, Path):
        save_dir_parent = save_dir_parent.absolute().as_posix()
    save_dir = os.path.join(save_dir_parent, f"lee2019_filtered_{target_freq}")
    convert_dataset(save_dir, "Lee2019_MI", subject_ids, target_freq, load_subjects_independently=True, band_pass=(2, None))

    if delete_after:
        delete_braindecode_local_copy("MNE-lee2019-mi-data")

def construct_shin_datasets(save_dir_parent: str | Path, subject_ids: list[SubjectID], target_freq: int | None = None, delete_after: bool = False):
    """
    Constructs the shin datasets from the given subject ids

    Automatically saved with the path (save_dir_parent / "shin")
    """
    print("Constructing shin datasets")
    if isinstance(save_dir_parent, Path):
        save_dir_parent = save_dir_parent.absolute().as_posix()
    save_dir = os.path.join(save_dir_parent, f"shin_{target_freq}")
    convert_dataset(save_dir, ["Shin2017A", "Shin2017B"], subject_ids, target_freq, load_subjects_independently=True, dataset_kwargs={"accept": True})

    if delete_after:
        delete_braindecode_local_copy("MNE-eegfnirs-data")  # This one location is used for both shin datasets

if __name__ == "__main__":
    save_path = Path(__file__).parent / 'test_datasets'

    # construct_physionet_dataset(save_path, list(range(1, 109+1)), target_freq=120, delete_after=False)
    construct_lee2019_dataset(save_path, list(range(32, 51+1)), target_freq=512, delete_after=False)
    # construct_shin_datasets(save_path, list(range(1, 29+1)), target_freq=120, delete_after=False)

    # construct_filtered_lee2019_dataset(save_path, list(range(1, 2+1)), target_freq=120, delete_after=False)