"""
Utilities for loading subject datasets and processing and visualizing them them with different nme methods
"""

from pathlib import Path
import mne
from matplotlib import pyplot as plt

from thesis.datasets.subject_dataset import SubjectDataset, load_subject_datasets, load_subject_dataset, ElementDict, SubjectDatasetKey, get_subject_datasets_in_dir, validate_dataset, save_subject_dataset

def convert_dataset_to_nme(dataset: SubjectDataset):
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

def convert_dataset_to_subject(dataset: dict) -> SubjectDataset:
    """
    Converts from mne back to a subject dataset
    """
    for dataset_key, elem in dataset.items():
        elem['data'] = elem['data'].get_data()
    return validate_dataset(dataset)



def load_subject_dataset_as_nme(dir: Path, max_subjects: int = None) -> dict[str, dict]:
    """
    Loads a subject dataset from the given directory and converts it to an nme dataset
    """
    dataset: dict[str, SubjectDataset] = load_subject_datasets(dir, max_subjects)
    processed_dataset = {}
    for subject_id, subject_dataset in dataset.items():
        processed_dataset[subject_id] = convert_dataset_to_nme(subject_dataset)
    return processed_dataset

def plot_highpass_signal(nme_dataset: dict[str, dict], subject_id: str, key: SubjectDatasetKey, n_channels=10, remove_dc=False, highpass=1, duration=30):
    """
    Plots the given key from the nme dataset
    """
    subject_dataset = nme_dataset[subject_id]
    data_element = subject_dataset[key]
    data_element['data'].plot(n_channels=n_channels, remove_dc=remove_dc, scalings='auto', highpass=highpass, duration=duration)
    plt.show()

def plot_power_spectrum(nme_dataset: dict[str, dict], subject_id: str, key: SubjectDatasetKey, fmin=0, fmax=None):
    """
    Plots the power spectrum of the given key from the nme dataset
    """
    subject_dataset = nme_dataset[subject_id]
    data_element = subject_dataset[key]
    if fmax is None:
        # Set fmax to the nyquist frequency
        fmax = data_element['metadata']['freq'] / 2
    data_element['data'].plot_psd(fmin=fmin, fmax=fmax, average=True, picks='data')
    plt.show()

def generate_and_plot_ica(nme_dataset: dict[str, dict], subject_id: str, key: SubjectDatasetKey):
    """
    Tests ICA on the given key from the nme dataset
    """
    subject_dataset = nme_dataset[subject_id]
    data_element = subject_dataset[key]
    raw = data_element['data']
    filt_raw = raw.copy().filter(1, None)
    filt_raw_cropped = filt_raw.copy().crop(tmin=60, tmax=120)

    # filt_raw.plot(scalings="auto", n_channels=10, duration=60)

    ica = mne.preprocessing.ICA(n_components=15, random_state=97, max_iter="auto")
    ica.fit(filt_raw_cropped)

    # raw.plot(scalings="auto", n_channels=10)
    # plt.show()

    fig = ica.plot_sources(filt_raw, show_scrollbars=True)
    plt.show()
    plt.close(fig)
    print("To be excluded: ", ica.exclude)

    reconst_raw = filt_raw.copy()
    ica.apply(reconst_raw)
    fig_1 = filt_raw.plot(scalings="auto", n_channels=10, start=60, duration=60)
    fig_2 = reconst_raw.plot(scalings="auto", n_channels=10, start=60, duration=60)
    plt.show()
    plt.close(fig_1)
    plt.close(fig_2)

def create_ica(nme_dataset: dict[str, dict], subject_id: str, key: SubjectDatasetKey, tmin=60, tmax=120, n_components=15):
    """
    Uses manual input to create an ICA object for the given key from the nme dataset
    """
    subject_dataset = nme_dataset[subject_id]
    data_element = subject_dataset[key]
    raw = data_element['data']
    return create_ica_from_raw(raw, tmin, tmax, n_components)

def create_ica_from_raw(raw: mne.io.Raw, tmin=60, tmax=120, n_components=15):
    """
    Uses manual input to create an ICA object for the given key from the nme dataset
    """
    filt_raw = raw.copy().filter(1, None)
    filt_raw_cropped = filt_raw.copy().crop(tmin=60, tmax=120)

    ica = mne.preprocessing.ICA(n_components=15, random_state=97, max_iter="auto")
    ica.fit(filt_raw_cropped)

    fig = ica.plot_sources(filt_raw, show_scrollbars=True)
    plt.show()
    plt.close(fig)

    print("Excluded: ", ica.exclude)
    return ica

def plot_ica_sources(ica, nme_dataset: dict[str, dict], subject_id: str, key: SubjectDatasetKey):
    """
    Visualizes a pretrained ica on a given key from the nme dataset
    """
    subject_dataset = nme_dataset[subject_id]
    data_element = subject_dataset[key]
    raw = data_element['data']
    filt_raw = raw.copy().filter(1, None)

    fig = ica.plot_sources(filt_raw, show_scrollbars=True)
    plt.show()
    plt.close(fig)

def remove_dataset_artifacts(subjects: list[str], input_dataset_path: Path, output_dataset_dir: Path, highpass_cutoff=1.0, resample_freq=120):
    """
    Takes an nme dataset, runs a low pass filter over it, and then removes artifacts

    A new ICA is trained for each session, but within one session the same ica is used to remove artifacts from all runs

    As subjects are processed we save them to disk. If there is already a subject.pt file for a subject, we skip them

    Since the datasets can get large, we load each subjects individually
    """
    output_dataset_dir.mkdir(parents=True, exist_ok=True)
    for subject_id in subjects:
        file_name = f"{subject_id}.pt"
        input_file_path = input_dataset_path / file_name
        output_file_path = output_dataset_dir / file_name
        assert input_file_path.exists(), f"Input file {input_file_path} does not exist"

        if output_file_path.exists():
            print(f"Skipping subject {subject_id}")
            continue

        print(f"Processing subject {subject_id}")
        subject_dataset = load_subject_dataset(input_file_path)
        mne_dataset = convert_dataset_to_nme(subject_dataset)
        subject_icas = {}
        for dataset_key, elem in mne_dataset.items():
            raw = elem['data']
            session_id = dataset_key[1]
            if session_id not in subject_icas:
                ica = create_ica_from_raw(raw, n_components=15)
                subject_icas[session_id] = ica

        # Now we have an ica for each session and we can process the dataset
        new_nme_dataset = {}
        for dataset_key, elem in mne_dataset.items():
            raw = elem['data']
            filtered_raw = raw.copy().filter(highpass_cutoff, None)
            session_id = dataset_key[1]
            ica = subject_icas[session_id]
            ica.apply(filtered_raw)
            elem['data'] = filtered_raw

            # Resample the data
            if resample_freq is not None:
                elem['data'].resample(resample_freq)
                elem['metadata']['freq'] = resample_freq
            new_nme_dataset[dataset_key] = elem
        
        # Convert the mne dataset back to a subject dataset
        subject_dataset = convert_dataset_to_subject(new_nme_dataset)
        save_subject_dataset(subject_dataset, output_file_path)

def crop_dataset(subjects: list[str], input_dataset_path: Path, output_dataset_dir: Path):
    """
    Visualizes the dataset and then allows the user to crop the beginning of each run 
    """
    output_dataset_dir.mkdir(parents=True, exist_ok=True)
    for subject_id in subjects:
        file_name = f"{subject_id}.pt"
        input_file_path = input_dataset_path / file_name
        output_file_path = output_dataset_dir / file_name
        assert input_file_path.exists(), f"Input file {input_file_path} does not exist"

        if output_file_path.exists():
            print(f"Skipping subject {subject_id}")
            continue

        print(f"Processing subject {subject_id}")
        subject_dataset = load_subject_dataset(input_file_path)
        mne_dataset = convert_dataset_to_nme(subject_dataset)

        for dataset_key, elem in mne_dataset.items():
            raw = elem['data']
            session_id = dataset_key[1]

            fig = raw.plot(scalings="auto", n_channels=10, duration=60)
            plt.show()
            start_time = float(input("Enter start time: "))
            elem['data'] = raw.copy().crop(tmin=start_time)
            plt.close(fig)

        # Convert the mne dataset back to a subject dataset
        subject_dataset = convert_dataset_to_subject(mne_dataset)
        save_subject_dataset(subject_dataset, output_file_path)


if __name__ == "__main__":
    braindecode_dataset_path = Path(__file__).parent / "braindecode" / "datasets"
    lee2019_dataset_path = braindecode_dataset_path / "lee2019_512"
    shin_dataset_path = braindecode_dataset_path / "shin"
    physionet_dataset_path = braindecode_dataset_path / "physionet"
    lee_2019_filtered_dataset_path = braindecode_dataset_path / "lee2019_512_ica_highpass_filtered_resampled_120"

    large_eeg_dataset_path = Path(__file__).parent / "large_eeg"
    kaya_dataset_path = large_eeg_dataset_path / "processed_data"
    kaya_filtered_dataset_path = large_eeg_dataset_path / "processed_data_ica_highpass_filtered_resampled_120"

    # # lee2019_nme_dataset = load_subject_dataset_as_nme(lee2019_dataset_path, max_subjects=2)
    # lee2019_filtered_nme_dataset = load_subject_dataset_as_nme(lee_2019_filtered_dataset_path, max_subjects=2)
    # # physionet_nme_dataset = load_subject_dataset_as_nme(physionet_dataset_path, max_subjects=1)
    # # shin_nme_dataset = load_subject_dataset_as_nme(shin_dataset_path, max_subjects=1)

    # example_dataset = lee2019_filtered_nme_dataset
    # subjects = list(example_dataset.keys())
    # subject_keys: dict[str, list[SubjectDatasetKey]] = {}
    # for subject in subjects:
    #     run_keys = list(example_dataset[subject].keys())
    #     subject_keys[subject] = run_keys
    #     print(f"Subject: {subject}")
    #     for run_key in run_keys:
    #         print(f"\tRun: {run_key}")

    # test_subject = subjects[0]
    # test_run_key = subject_keys[test_subject][0]

    # # plot_highpass_signal(example_dataset, test_subject, test_run_key, highpass=None, duration=120)
    # # plot_power_spectrum(example_dataset, test_subject, test_run_key)
    # generate_and_plot_ica(example_dataset, test_subject, test_run_key)

    # ica = create_ica(example_dataset, test_subject, test_run_key)
    # # Try plotting the ica on the next run
    # test_run_key_2 = subject_keys[test_subject][1]
    # plot_ica_sources(ica, example_dataset, test_subject, test_run_key_2)

    # test_subject_2 = subjects[1]
    # test_run_key_3 = subject_keys[test_subject_2][0]
    # plot_ica_sources(ica, example_dataset, test_subject_2, test_run_key_3)



    input_dataset_path = kaya_filtered_dataset_path

    if False:
        input_mne_dataset = load_subject_dataset_as_nme(input_dataset_path, max_subjects=2)

        subjects = list(input_mne_dataset.keys())
        subject_keys: dict[str, list[SubjectDatasetKey]] = {}
        for subject in subjects:
            run_keys = list(input_mne_dataset[subject].keys())
            subject_keys[subject] = run_keys
            print(f"Subject: {subject}")
            for run_key in run_keys:
                print(f"\tRun: {run_key}")
            
        test_subject = subjects[0]
        test_run_key = subject_keys[test_subject][0]

        plot_highpass_signal(input_mne_dataset, test_subject, test_run_key, highpass=None, duration=120)
        plot_power_spectrum(input_mne_dataset, test_subject, test_run_key)
        generate_and_plot_ica(input_mne_dataset, test_subject, test_run_key)
    

    if False:
        output_dataset_path = input_dataset_path.parent / f"{input_dataset_path.name}_ica_highpass_filtered_resampled_120"
        subjects = get_subject_datasets_in_dir(input_dataset_path)
        remove_dataset_artifacts(subjects, input_dataset_path, output_dataset_path)
    
    if True:
        output_dataset_path = input_dataset_path.parent / f"{input_dataset_path.name}_ica_highpass_filtered_resampled_120_cropped"
        # subjects = get_subject_datasets_in_dir(input_dataset_path)
        subjects = ["SubjectI", "SubjectF"]
        crop_dataset(subjects, input_dataset_path, output_dataset_path)
