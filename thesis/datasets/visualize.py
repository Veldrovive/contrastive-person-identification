"""
Helper function for visualizing datasets
"""

from pathlib import Path
from thesis.datasets.subject_dataset import SubjectDataset, load_subject_datasets, ElementDict
from matplotlib import pyplot as plt

def plot_signal_slice(ax, data_element: ElementDict, channel_index: int, start_sample: int, end_sample: int, title: str = "", preprocess_fn = lambda x: x):
    """
    Plots a slice of the signal from the dataset
    """
    signal = data_element.data
    sliced_signal = signal[:, start_sample:end_sample]
    sliced_signal = preprocess_fn(sliced_signal) if preprocess_fn is not None else sliced_signal
    ax.plot(sliced_signal[channel_index])
    ax.set_title(title)

def plot_example(dataset: SubjectDataset, channel_index: int, start_sample: int, end_sample: int):
    """
    Plots an example from the dataset
    """
    data_element = dataset.root[list(dataset.root.keys())[0]]
    fig, ax = plt.subplots(1, 1)
    plot_signal_slice(ax, data_element, channel_index, start_sample, end_sample)
    plt.show()

def plot_examples(examples: list[tuple[str, SubjectDataset, int, int, int]], preprocess_fn = lambda x: x):
    """
    Plots a list of examples
    """
    fig, axes = plt.subplots(len(examples), 1)
    for i, (title, dataset, channel_index, start_sample, end_sample) in enumerate(examples):
        data_element = dataset.root[list(dataset.root.keys())[0]]
        plot_signal_slice(axes[i], data_element, channel_index, start_sample, end_sample, title, preprocess_fn)
    plt.show()

if __name__ == "__main__":
    from thesis.preprocessing.meta import MetaPreprocessorConfig, preprocessor_factory

    test_preprocess_config = MetaPreprocessorConfig(
        sample_rate=120,
        clamping_sd=5,
        target_sample_rate=None,
        use_baseline_correction=True,
        use_robust_scaler=True,
        use_clamping=True
    )
    preprocess_fn = preprocessor_factory(test_preprocess_config, track_times=False)

    braindecode_dataset_path = Path(__file__).parent / "braindecode" / "datasets"
    lee2019_dataset_path = braindecode_dataset_path / "lee2019"
    lee2019_filtered_dataset_path = braindecode_dataset_path / "lee2019_filtered"
    shin_dataset_path = braindecode_dataset_path / "shin"
    physionet_dataset_path = braindecode_dataset_path / "physionet"

    lee2019_datasets = load_subject_datasets(lee2019_dataset_path, max_subjects=1)
    lee2019_filtered_datasets = load_subject_datasets(lee2019_filtered_dataset_path, max_subjects=1)
    shin_datasets = load_subject_datasets(shin_dataset_path, max_subjects=1)
    physionet_datasets = load_subject_datasets(physionet_dataset_path, max_subjects=1)

    lee2019_subject = list(lee2019_datasets.values())[0]
    lee2019_filtered_subject = list(lee2019_filtered_datasets.values())[0]
    shin_subject = list(shin_datasets.values())[0]
    physionet_subject = list(physionet_datasets.values())[0]

    # plot_example(physionet_subject, 0, 0, 120*5)
    offset = 120*0
    show_time = 200
    channel = 1
    # examples = [
    #     ("Physionet", physionet_subject, channel, 0+offset, 120*show_time + offset),
    #     ("Lee2019", lee2019_subject, channel, 0+offset, 120*show_time + offset),
    #     ("Shin", shin_subject, channel, 0+offset, 120*show_time + offset)
    # ]
    # examples = [
    #     ("Lee2019 - ch0", lee2019_subject, 0, 0+offset, 120*show_time + offset),
    #     ("Lee2019 - ch1", lee2019_subject, 1, 0+offset, 120*show_time + offset),
    #     ("Lee2019 - ch2", lee2019_subject, 2, 0+offset, 120*show_time + offset)
    # ]
    examples = [
        ("Physionet", physionet_subject, channel, 0+offset, 120*show_time + offset),
        ("Lee2019 - ch0", lee2019_subject, 0, 0+offset, 120*show_time + offset),
        ("Lee2019_filtered - ch0", lee2019_filtered_subject, 0, 0+offset, 120*show_time + offset),
    ]
    plot_examples(examples, preprocess_fn=preprocess_fn)