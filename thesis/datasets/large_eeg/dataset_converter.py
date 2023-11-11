"""
Uses files downloaded from https://figshare.com/collections/A_large_electroencephalographic_motor_imagery_dataset_for_electroencephalographic_brain_computer_interfaces/3917698
to create subject datasets

Electrode order is Fp1 Fp2 F3 F4 C3 C4 P3 P4 O1 O2 A1 A2 F7 F8 T3 T4 T5 T6 Fz Cz Pz X3
"""

channels = ['Fp1', 'Fp2', 'F3', 'F4', 'C3', 'C4', 'P3', 'P4', 'O1', 'O2', 'A1', 'A2', 'F7', 'F8', 'T3', 'T4', 'T5', 'T6', 'Fz', 'Cz', 'Pz', 'X3']

from pathlib import Path
import numpy as np
from scipy.io import loadmat

import requests
from pathlib import Path

from thesis.datasets.subject_dataset import SubjectDataset, validate_dataset, save_subject_dataset

def download_figshare_collection(output_path: Path, collection_url: str):
    """
    Download all files from a Figshare collection to a specified output directory, skipping existing files.

    Parameters:
    - output_path: A pathlib Path object for the directory where files should be saved.
    - collection_url: The Figshare API URL for the collection.
    """
    # Step 1: Make a request to the collection API endpoint
    response = requests.get(collection_url)
    articles = response.json()

    for article in articles:
        # Step 2: Parse the article's public API URL and get the file information
        article_details_response = requests.get(article['url_public_api'])
        article_details = article_details_response.json()

        # Step 3: Download each file
        for file in article_details['files']:
            file_name = file['name']
            file_url = file['download_url']
            file_path = output_path / file_name

            # Skip download if file already exists
            if not file_path.exists():
                # Make a GET request to download the file
                file_response = requests.get(file_url, stream=True)

                # Save the file to the specified output path
                with open(file_path, 'wb') as f:
                    for chunk in file_response.iter_content(chunk_size=8192):
                        if chunk:  # filter out keep-alive new chunks
                            f.write(chunk)
            else:
                print(f"File {file_name} already exists. Skipping download.")

def load_eeg_data(file_path: Path):
    """
    Load EEG data from a .mat file and return the data with channels as the first dimension.

    Parameters:
    - file_path: A pathlib Path object pointing to the .mat file

    Returns:
    - eeg_data: A numpy array with channels as the first dimension and time as the second
    - samp_freq: The sampling frequency of the EEG data
    """
    # Load the .mat file
    mat_contents = loadmat(file_path)
    
    # Access the 'o' structure
    o_structure = mat_contents['o']
    
    # Get the data and the sampling frequency
    eeg_data = o_structure['data'][0, 0].T  # Transpose to make channels the first dimension
    samp_freq = o_structure['sampFreq'][0, 0][0, 0]
    
    return eeg_data, samp_freq

def get_file_metadata(file_path: Path) -> dict:
    """
    We get the metadata directly from the name. Chunks are separated by -
    The first chunk is the task. The second is the subject. The third is the date. The rest we do not care about
    """
    file_name = file_path.stem
    chunks = file_name.split("-")
    task = chunks[0]
    subject = chunks[1]
    date = chunks[2]
    return {
        "task": task,
        "subject": subject,
        "date": date
    }

def convert_dataset(dataset_dir: Path, output_dir: Path):
    """
    Converts the datasets into the SubjectDataset format.
    Each file in the same subject is assumed to be a separate session.
    To prevent running out of memory we work one subject at a time.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    raw_data_files = dataset_dir.glob("*.mat")
    # Now we group the files by subject
    subject_files = {}
    for file in raw_data_files:
        metadata = get_file_metadata(file)
        subject = metadata["subject"]
        if subject not in subject_files:
            subject_files[subject] = []
        subject_files[subject].append(file)

    # For each subject we load the data
    for subject, subject_file_list in subject_files.items():
        subject_dataset = {}
        output_file = output_dir / f"{subject}.pt"

        for file in subject_file_list:
            metadata = get_file_metadata(file)
            session_id = f"{metadata['task']}_{metadata['date']}"
            subject_id = metadata["subject"]
            dataset_key = (subject_id, session_id, 0)
            assert dataset_key not in subject_dataset, f"Duplicate dataset key {dataset_key}"

            eeg_data, samp_freq = load_eeg_data(file)
            assert eeg_data.shape[0] == len(channels), f"Expected {len(channels)} channels, got {eeg_data.shape[0]}"

            element_metadata = { 'channels': channels, 'freq': samp_freq }
            element = {
                'subject': subject_id,
                'session': session_id,
                'run': 0,
                'data': eeg_data,
                'metadata': element_metadata
            }
            subject_dataset[dataset_key] = element
        
        subject_dataset = validate_dataset(subject_dataset)
        save_subject_dataset(subject_dataset, output_file)

        

if __name__ == "__main__":
    raw_data_dir = Path(__file__).parent / "raw_data"
    output_dir = Path(__file__).parent / "datasets"

    figshare_url = "https://api.figshare.com/v2/collections/3917698/articles?page_size=80"

    download_figshare_collection(raw_data_dir, figshare_url)
    convert_dataset(raw_data_dir, Path(__file__).parent / "processed_data")