from thesis.structs import *
from .subject_dataset import *
from .dataset import *

def get_dataloaders(
    base_datasets: list[DatasetConfig],
    dataset_configs: dict[str, tuple[ContrastiveSubjectDatasetConfig, DataloaderConfig]],  # Keys are "train", "extrap_val", "extrap_test", "intra_val", "intra_test", "downstream". All keys must be present
    load_time_preprocessor_config: LoadTimePreprocessorConfig
):
    """
    Creates dataloaders for the 5 major datasets as well as the downstream task dataset
    """
    # We require that all dataset configs have the same window size. Enforce this now
    window_size_s = None
    for dataset_config in dataset_configs.values():
        if window_size_s is None:
            window_size_s = dataset_config[0].window_size_s
        else:
            assert window_size_s == dataset_config[0].window_size_s, "All dataset configs must have the same window size"
    datasets: dict[str, list[dict[str, SubjectDataset]]] = {
        "train": {},
        "extrap_val": {},
        "extrap_test": {},
        "intra_val": {},
        "intra_test": {},
        "downstream": {}
    }  # Maps from "train"... to the loaded datasets from each path
    downstream_dataset_metadata = {}

    for base_dataset_config in base_datasets:
        dataset_name = base_dataset_config.name
        dataset_path = base_dataset_config.path
        assert dataset_path.exists(), f"Dataset path {dataset_path} does not exist"
        split_config = base_dataset_config.split_config
        n_subjects = base_dataset_config.max_subjects

        dataset_subjects = get_subject_datasets_in_dir(dataset_path)
        if n_subjects is not None:
            dataset_subjects = dataset_subjects[:n_subjects]
        n_subjects = len(dataset_subjects)

        # Prepare for creating the downstream tasks dataset
        if split_config.downstream_num_subjects > -1:
            downstream_num_subjects = split_config.downstream_num_subjects
        else:
            downstream_num_subjects = n_subjects
        assert downstream_num_subjects <= n_subjects, "The number of downstream subjects must be less than or equal to the number of subjects in the dataset"
        downstream_subjects = dataset_subjects[:downstream_num_subjects]

        subject_datasets = load_subject_datasets(dataset_path, subjects=dataset_subjects, load_time_preprocessor_config=load_time_preprocessor_config)
        subject_metadata_dataset = load_subject_metadata(dataset_path, subjects=dataset_subjects)
        # Get the frequency of this dataset
        dataset_freq = list(subject_datasets[dataset_subjects[0]].root.values())[0].metadata.freq
        window_size_samples = int(window_size_s * dataset_freq)
        # Split the dataset
        split_datasets = get_dataset_split(subject_datasets, split_config.train_prop, split_config.extrap_val_prop, split_config.extrap_test_prop, split_config.intra_val_prop, split_config.intra_test_prop, window_size_samples=window_size_samples)
        for dataset_type, dataset in split_datasets.items():
            datasets[dataset_type][dataset_name] = dataset

        if downstream_num_subjects > 0:
            # Then we also need to create the downstream dataset
            downstream_subject_datasets = { subject: subject_datasets[subject] for subject in downstream_subjects }
            downstream_subject_metadata = { subject: subject_metadata_dataset[subject] for subject in downstream_subjects }
            downstream_dataset_metadata[dataset_name] = downstream_subject_metadata
            datasets["downstream"][dataset_name] = downstream_subject_datasets
    # Now we have sets of datasets for each type. We need to concatenate them
    full_datasets = {}
    for dataset_type, dataset_dict in datasets.items():
        full_datasets[dataset_type] = concatenate_datasets(dataset_dict)
    full_downstream_metadata = concatenate_datasets(downstream_dataset_metadata)
    # Now we have the full datasets for each split 
    # Now we need to create the contrastive datasets
    dataloaders = {}
    for dataset_type in full_datasets.keys():
        subject_dataset = full_datasets[dataset_type]
        configs = dataset_configs[dataset_type]
        contrastive_dataset_config = configs[0]
        dataloader_config = configs[1]
        subject_metadata = None

        if dataset_type != "train":
            assert contrastive_dataset_config.sample_over_subjects_toggle == False, "Only the training set can sample over subjects. Otherwise the evaluation would be invalid."
        if dataset_type == "downstream":
            assert contrastive_dataset_config.max_samples_per_subject is not None, "The downstream dataset must have a limit set for the number of samples per subject"
            subject_metadata = full_downstream_metadata


        if len(subject_dataset) == 0:
            dataloaders[dataset_type] = None
        else:
            contrastive_dataset = ContrastiveSubjectDataset(subject_dataset, contrastive_dataset_config, subject_metadata=subject_metadata)
            loader = get_contrastive_subject_loader(contrastive_dataset, batch_size=dataloader_config.batch_size, shuffle=dataloader_config.shuffle, num_workers=dataloader_config.num_workers)
            dataloaders[dataset_type] = loader

    # Sanity checks:
    channels = None
    freq = None
    for dataloader in dataloaders.values():
        if dataloader is None:
            continue
        if channels is None:
            channels = dataloader.dataset.channels
        else:
            assert channels == dataloader.dataset.channels, "All datasets must use the same channels"

        if freq is None:
            freq = dataloader.dataset.freq
        else:
            assert freq == dataloader.dataset.freq, "All datasets must use the same frequency"
    
    return dataloaders, len(channels), freq