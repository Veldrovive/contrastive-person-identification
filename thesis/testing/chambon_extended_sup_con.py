"""
Loads a chambon model and a linear head with a very small dataset and overfits it
"""

import torch
from pathlib import Path
import json
import umap
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.neighbors import NearestNeighbors
from pydantic import RootModel

from thesis.structs.chambon_structs import ChambonExtendableConfig, ChambonConfig
from thesis.structs.contrastive_head_structs import ContrastiveHeadConfig, HeadStyle
# from thesis.datasets.braindecode.PhysionetMI import get_physionet_datasets, PhysionetMIDatasetConfig, get_physionet_dataloaders, DataloaderConfig
from thesis.datasets.subject_dataset import load_subject_datasets, concatenate_datasets, get_dataset_split, SubjectDataset, get_subject_datasets_in_dir
from thesis.datasets.dataset import ContrastiveSubjectDataset, ContrastiveSubjectDatasetConfig, get_contrastive_subject_loaders, get_contrastive_subject_loader
from thesis.preprocessing import MetaPreprocessorConfig
from thesis.augmentation import SessionVarianceAugmentationConfig
from thesis.structs.training_structs import TrainingConfig, TrainingState
from thesis.structs.dataset_structs import DatasetSplitConfig, DataloaderConfig
from thesis.loss_functions.sup_con import SupConLoss
from thesis.evaluation.knn import get_embeddings, get_cross_session_knn_metrics, visualize_embeddings, visualize_confusion_matrix

from thesis.models.chambon_extendable import ChambonNetExtendable
from thesis.models.chambon import ChambonNet
from thesis.models.contrastive_head import HeadStyle, create_contrastive_module

import wandb
from tqdm import tqdm

def create_model(t_config: TrainingConfig, m_config: ChambonExtendableConfig, contrastive_loss_size: int = 128):
    model = ChambonNetExtendable(m_config)
    test_data = torch.randn(1, m_config.C, m_config.T)
    test_output = model(test_data)
    embedding_size = test_output.shape[1]
    head_config = ContrastiveHeadConfig(logit_dimension=embedding_size, c_loss_dimension=contrastive_loss_size, head_style=HeadStyle.LINEAR, layer_sizes=[contrastive_loss_size])
    full_model = create_contrastive_module(model, head_config)

    # Move the model to the device
    full_model = full_model.to(t_config.device)

    # Create the optimizer
    optimizer = torch.optim.Adam(full_model.parameters(), lr=1e-4)

    return full_model, optimizer

def create_model_old(t_config: TrainingConfig, m_config: ChambonConfig, contrastive_loss_size: int = 128):
    model = ChambonNet(m_config)
    test_data = torch.randn(1, m_config.C, m_config.T)
    test_output = model(test_data)
    embedding_size = test_output.shape[1]
    head_config = ContrastiveHeadConfig(logit_dimension=embedding_size, c_loss_dimension=contrastive_loss_size, head_style=HeadStyle.LINEAR, layer_sizes=[contrastive_loss_size])
    full_model = create_contrastive_module(model, head_config)

    # Move the model to the device
    full_model = full_model.to(t_config.device)

    # Create the optimizer
    optimizer = torch.optim.Adam(full_model.parameters(), lr=1e-3)

    return full_model, optimizer

def evaluate_model(training_state: TrainingState, training_config: TrainingConfig, model, dataloaders):
    extrap_dataloader = dataloaders["extrap_val"]
    intra_dataloader = dataloaders["intra_val"]
    visualization_path = training_config.evaluation_visualization_path
    visualization_path.mkdir(parents=True, exist_ok=True)
    epoch = training_state.epoch

    # Prep the paths for all visualizations
    epoch_visualization_path = visualization_path / f"epoch_{epoch}"
    intra_all_session_embeddings_visualization_path = epoch_visualization_path / "intra_all_session_embeddings.png"
    intra_subject_session_embeddings_visualization_path = epoch_visualization_path / "intra_subject_session_embeddings"
    intra_top_1_confusion_matrix_path = epoch_visualization_path / "intra_top_1_confusion_matrix.png"
    intra_consensus_k_confusion_matrix_path = epoch_visualization_path / "intra_consensus_k_confusion_matrix.png"
    intra_subject_session_embeddings_visualization_path.mkdir(parents=True, exist_ok=True)

    extrap_all_session_embeddings_visualization_path = epoch_visualization_path / "extrap_all_session_embeddings.png"
    extrap_subject_session_embeddings_visualization_path = epoch_visualization_path / "extrap_subject_session_embeddings"
    extrap_top_1_confusion_matrix_path = epoch_visualization_path / "extrap_top_1_confusion_matrix.png"
    extrap_consensus_k_confusion_matrix_path = epoch_visualization_path / "extrap_consensus_k_confusion_matrix.png"
    extrap_subject_session_embeddings_visualization_path.mkdir(parents=True, exist_ok=True)

    res = {}

    if training_config.run_intra_val:
        # Run the intra evaluation
        intra_embeddings, intra_labels, intra_sessions = get_embeddings(training_config, model, intra_dataloader, limit=None)
        intra_cross_session_knn_metrics = get_cross_session_knn_metrics(
            intra_embeddings, intra_labels, intra_sessions,
            k=training_config.evaluation_k, metric='cosine'
        )
        res["intra_top_1_accuracy"] = intra_cross_session_knn_metrics["total_top_1_accuracy"]
        res[f"intra_top_{training_config.evaluation_k}_accuracy"] = intra_cross_session_knn_metrics["total_top_k_accuracy"]

        session_embeddings = intra_cross_session_knn_metrics["session_embeddings"]
        intra_all_session_embeddings_visualization = visualize_embeddings(session_embeddings, title="UMAP Projection of Intra-Training Set Embeddings", separate_sessions=False)
        intra_all_session_embeddings_visualization.savefig(intra_all_session_embeddings_visualization_path)
        plt.close(intra_all_session_embeddings_visualization)
        res["intra_embedding_visualization"] = wandb.Image(intra_all_session_embeddings_visualization_path.absolute().as_posix())

        # We also visualize embeddings for some individual subjects to see how the sessions cluster
        # To do this we first need to re-organize the session_embeddings into a map from subject id to a subset of the session_embeddings
        subject_session_embeddings = {}  # Maps from subject id to a dictionary from (subject_id, session_id) to embeddings
        for (subject_id, session_id), embedding in session_embeddings.items():
            if subject_id not in subject_session_embeddings:
                subject_session_embeddings[subject_id] = {}
            subject_session_embeddings[subject_id][(subject_id, session_id)] = embedding
        # Now we take the first n subjects and visualize their embeddings
        n_subjects = 5
        for subject_id, session_embeddings in list(subject_session_embeddings.items())[:n_subjects]:
            subject_embeddings_visualization = visualize_embeddings(session_embeddings, title=f"UMAP Projection of Intra-Training Set Embeddings for Subject {subject_id}", separate_sessions=True)
            subject_embeddings_visualization_path = intra_subject_session_embeddings_visualization_path / f"subject_{subject_id}.png"
            subject_embeddings_visualization.savefig(subject_embeddings_visualization_path)
            plt.close(subject_embeddings_visualization)
            res[f"intra_embedding_visualization_subject_{subject_id}"] = wandb.Image(subject_embeddings_visualization_path.absolute().as_posix())

        # And finally we plot the confusion matrix for both the top 1 and consensus labels
        true_labels = intra_cross_session_knn_metrics["true_labels"]
        top_1_labels = intra_cross_session_knn_metrics["top_1_labels"]
        consensus_k_labels = intra_cross_session_knn_metrics["consensus_k_labels"]

        top_1_confusion_matrix = visualize_confusion_matrix(true_labels=true_labels, predicted_labels=top_1_labels, title="Confusion Matrix for Intra-Training Set Top 1 Labels")
        top_1_confusion_matrix.savefig(intra_top_1_confusion_matrix_path)
        plt.close(top_1_confusion_matrix)
        res["intra_top_1_confusion_matrix"] = wandb.Image(intra_top_1_confusion_matrix_path.absolute().as_posix())

        consensus_k_confusion_matrix = visualize_confusion_matrix(true_labels=true_labels, predicted_labels=consensus_k_labels, title="Confusion Matrix for Intra-Training Set Consensus Labels")
        consensus_k_confusion_matrix.savefig(intra_consensus_k_confusion_matrix_path)
        plt.close(consensus_k_confusion_matrix)
        res[f"intra_consensus_{training_config.evaluation_k}_confusion_matrix"] = wandb.Image(intra_consensus_k_confusion_matrix_path.absolute().as_posix())

    if training_config.run_extrap_val:
        # Now we do the same thing for the extrapolation set
        extrap_embeddings, extrap_labels, extrap_sessions = get_embeddings(training_config, model, extrap_dataloader, limit=None)
        extrap_cross_session_knn_metrics = get_cross_session_knn_metrics(
            extrap_embeddings, extrap_labels, extrap_sessions,
            k=training_config.evaluation_k, metric='cosine'
        )
        res["extrap_top_1_accuracy"] = extrap_cross_session_knn_metrics["total_top_1_accuracy"]
        res[f"extrap_top_{training_config.evaluation_k}_accuracy"] = extrap_cross_session_knn_metrics["total_top_k_accuracy"]

        session_embeddings = extrap_cross_session_knn_metrics["session_embeddings"]
        extrap_all_session_embeddings_visualization = visualize_embeddings(session_embeddings, title="UMAP Projection of Extrapolation Set Embeddings", separate_sessions=False)
        extrap_all_session_embeddings_visualization.savefig(extrap_all_session_embeddings_visualization_path)
        plt.close(extrap_all_session_embeddings_visualization)
        res["extrap_embedding_visualization"] = wandb.Image(extrap_all_session_embeddings_visualization_path.absolute().as_posix())

        # We also visualize embeddings for some individual subjects to see how the sessions cluster
        # To do this we first need to re-organize the session_embeddings into a map from subject id to a subset of the session_embeddings
        subject_session_embeddings = {}  # Maps from subject id to a dictionary from (subject_id, session_id) to embeddings
        for (subject_id, session_id), embedding in session_embeddings.items():
            if subject_id not in subject_session_embeddings:
                subject_session_embeddings[subject_id] = {}
            subject_session_embeddings[subject_id][(subject_id, session_id)] = embedding
        # Now we take the first n subjects and visualize their embeddings
        n_subjects = 5
        for subject_id, session_embeddings in list(subject_session_embeddings.items())[:n_subjects]:
            subject_embeddings_visualization = visualize_embeddings(session_embeddings, title=f"UMAP Projection of Extrapolation Set Embeddings for Subject {subject_id}", separate_sessions=True)
            subject_embeddings_visualization_path = extrap_subject_session_embeddings_visualization_path / f"subject_{subject_id}.png"
            subject_embeddings_visualization.savefig(subject_embeddings_visualization_path)
            plt.close(subject_embeddings_visualization)
            res[f"extrap_embedding_visualization_subject_{subject_id}"] = wandb.Image(subject_embeddings_visualization_path.absolute().as_posix())

        # And finally we plot the confusion matrix for both the top 1 and consensus labels
        true_labels = extrap_cross_session_knn_metrics["true_labels"]
        top_1_labels = extrap_cross_session_knn_metrics["top_1_labels"]
        consensus_k_labels = extrap_cross_session_knn_metrics["consensus_k_labels"]

        top_1_confusion_matrix = visualize_confusion_matrix(true_labels=true_labels, predicted_labels=top_1_labels, title="Confusion Matrix for Extrapolation Set Top 1 Labels")
        top_1_confusion_matrix.savefig(extrap_top_1_confusion_matrix_path)
        plt.close(top_1_confusion_matrix)
        res["extrap_top_1_confusion_matrix"] = wandb.Image(extrap_top_1_confusion_matrix_path.absolute().as_posix())

        consensus_k_confusion_matrix = visualize_confusion_matrix(true_labels=true_labels, predicted_labels=consensus_k_labels, title="Confusion Matrix for Extrapolation Set Consensus Labels")
        consensus_k_confusion_matrix.savefig(extrap_consensus_k_confusion_matrix_path)
        plt.close(consensus_k_confusion_matrix)
        res[f"extrap_consensus_{training_config.evaluation_k}_confusion_matrix"] = wandb.Image(extrap_consensus_k_confusion_matrix_path.absolute().as_posix())

    print("Evaluation:", res)
    return res

    
def get_dataloaders(
    base_datasets: dict[str, tuple[Path, DatasetSplitConfig]],
    dataset_configs: dict[str, tuple[ContrastiveSubjectDatasetConfig, DataloaderConfig]],  # Keys are "train", "extrap_val", "extrap_test", "intra_val", "intra_test". All keys must be present
):
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
        "intra_test": {}
    }  # Maps from "train"... to the loaded datasets from each path
    for dataset_name, (dataset_path, split_config) in base_datasets.items():
        print(f"Loading dataset {dataset_name}")
        dataset_subjects = get_subject_datasets_in_dir(dataset_path)
        subject_datasets = load_subject_datasets(dataset_path, subjects=dataset_subjects)
        # Get the frequency of this dataset
        dataset_freq = list(subject_datasets[dataset_subjects[0]].root.values())[0].metadata.freq
        window_size_samples = int(window_size_s * dataset_freq)
        # Split the dataset
        split_datasets = get_dataset_split(subject_datasets, split_config.train_prop, split_config.extrap_val_prop, split_config.extrap_test_prop, split_config.intra_val_prop, split_config.intra_test_prop, window_size_samples=window_size_samples)
        for dataset_type, dataset in split_datasets.items():
            datasets[dataset_type][dataset_name] = dataset
    # Now we have sets of datasets for each type. We need to concatenate them
    full_datasets = {}
    for dataset_type, dataset_dict in datasets.items():
        full_datasets[dataset_type] = concatenate_datasets(dataset_dict)
    # Now we have the full datasets for each split 
    # Now we need to create the contrastive datasets
    dataloaders = {}
    for dataset_type in full_datasets.keys():
        subject_dataset = full_datasets[dataset_type]
        configs = dataset_configs[dataset_type]
        contrastive_dataset_config = configs[0]
        dataloader_config = configs[1]

        if len(subject_dataset) == 0:
            dataloaders[dataset_type] = None
        else:
            contrastive_dataset = ContrastiveSubjectDataset(subject_dataset, contrastive_dataset_config)
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


# def get_dataloaders(
#     dataset_paths: dict[str, Path],
#     dataset_config: ContrastiveSubjectDatasetConfig,
#     dataloader_config: DataloaderConfig,
#     split_config: DatasetSplitConfig,
#     n_anchors: int = 2,
#     preprocess_config = None,
#     train_subject_limit: int | None = None
# ):
#     datasets = {}
#     for dataset_name, dataset_path in dataset_paths.items():
#         print(f"Loading dataset {dataset_name}")
#         datasets[dataset_name] = load_subject_datasets(dataset_path)
#     dataset = concatenate_datasets(datasets)

#     freq = list(list(dataset.values())[0].model_dump().values())[0]["metadata"]["freq"]  # Don't question it
#     window_size_samples = int(dataset_config.window_size_s * freq)

#     split_datasets = get_dataset_split(dataset, split_config.train_prop, split_config.extrap_val_prop, split_config.extrap_test_prop, split_config.intra_val_prop, split_config.intra_test_prop, window_size_samples=window_size_samples)
#     if train_subject_limit is not None:
#         # Limit the number of subjects in the training set
#         train_subjects = list(split_datasets["train"].keys())
#         train_subjects = train_subjects[:train_subject_limit]
#         split_datasets["train"] = {subject: split_datasets["train"][subject] for subject in train_subjects}
#     # contrastive_datasets = {key: ContrastiveSubjectDataset(dataset, dataset_config, preprocess_fn=preprocess_fn) for key, dataset in split_datasets.items()}
#     contrastive_datasets = {}
#     for key, dataset in split_datasets.items():
#         if key != "train":
#             # Then we are actually going to change the config slightly because resampling over subjects in the eval
#             # sets is not supported
#             indiv_dataset_config = dataset_config.copy()
#             indiv_dataset_config.sample_over_subjects_toggle = False
#         else:
#             indiv_dataset_config = dataset_config

#         if len(dataset) == 0:
#             contrastive_datasets[key] = None
#         else:
#             contrastive_datasets[key] = ContrastiveSubjectDataset(dataset, indiv_dataset_config, n_pos=n_anchors - 1, n_neg=0, preprocess_config=preprocess_config)
#     # For a sanity check, make sure that all the datasets are using the same channels
#     # I think it is possible for the extrap set to have chosen different channels
#     channels = None
#     for contrastive_dataset in contrastive_datasets.values():
#         if contrastive_dataset is None:
#             continue
#         if channels is None:
#             channels = contrastive_dataset.channels
#         else:
#             assert channels == contrastive_dataset.channels, "All datasets must use the same channels"
#     # Get the window size in samples
#     loaders = get_contrastive_subject_loaders(contrastive_datasets, dataloader_config)
#     return loaders, len(channels), window_size_samples

def save_checkpoint(training_state: TrainingState, training_config: TrainingConfig, model, optimizer):
    """
    Saves the model and optimizer state to the checkpoint path
    """
    checkpoint_path = training_config.checkpoint_path
    checkpoint_path.mkdir(parents=True, exist_ok=True)
    checkpoint_file = checkpoint_path / f"checkpoint_{training_state.epoch}.pth"
    torch.save({
        "epoch": training_state.epoch,
        "step": training_state.step,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "config": training_state.config
    }, checkpoint_file)

def load_checkpoint(training_state: TrainingState, training_config: TrainingConfig, model, optimizer):
    """
    Loads the model and optimizer state from the checkpoint path
    """
    checkpoint_file = training_config.checkpoint_load_path
    checkpoint = torch.load(checkpoint_file)
    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    # Move the model and optimizer to the device
    model = model.to(training_config.device)

    # Update the training state
    epoch = checkpoint.get("epoch", 0)
    step = checkpoint.get("step", 0)
    training_state.epoch = epoch
    training_state.step = step

    if training_config.upload_wandb_checkpoint:
        wandb.save(checkpoint_file)

def train(training_state: TrainingState, training_config: TrainingConfig, model, optimizer, dataloaders):
    # Get the total number of training parameters in the model
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Training model with {num_params} parameters")

    unique_subject_id_to_label = { id: i for i, id in enumerate(dataloaders['train'].dataset.unique_subjects) }
    loss_func = SupConLoss(temperature=training_config.loss_temperature)

    if training_config.evaluate_first:
        # Run an initial evaluation
        eval_res = evaluate_model(training_state, training_config, model, dataloaders)
        wandb.log(eval_res, step=training_state.step)

    for epoch in range(training_state.epoch, training_config.epochs):
        training_state.epoch = epoch
        wandb.log({"epoch": epoch}, step=training_state.step)
        save_checkpoint(training_state, training_config, model, optimizer)

        epoch_length = min(training_config.epoch_length, len(dataloaders["train"]))
        progress_bar = tqdm(range(epoch_length), desc=f"Epoch {epoch}")
        iterator = iter(dataloaders["train"])
        for batch_index in progress_bar:
            batch = next(iterator)
            optimizer.zero_grad()
            anchors = batch["anchor"]
            anchor_labels = torch.tensor([unique_subject_id_to_label[data['unique_subject_id']] for data in batch["anchor_metadata"]])
            positives = batch["positive"]
            if positives.ndim == 4:
                # The positive labels are just the anchor labels replicated across a new axis 1, n_pos times
                positive_labels = anchor_labels.unsqueeze(1).repeat(1, positives.shape[1])
                positives = positives.reshape(-1, positives.shape[-2], positives.shape[-1])
                positive_labels = positive_labels.reshape(-1)
            else:
                # The the positive labels are just the anchor labels
                positive_labels = anchor_labels
            
            # Now we can create the full batch by concatenating the anchors and positives
            batch_windows = torch.cat([anchors, positives], dim=0)
            batch_labels = torch.cat([anchor_labels, positive_labels], dim=0)
            # Convert all samples to float
            batch_windows = batch_windows.float()
            batch_labels = batch_labels.float()
            # Move the samples to the device
            batch_windows = batch_windows.to(training_config.device)
            batch_labels = batch_labels.to(training_config.device)

            embeddings = model(batch_windows)
            # Normalize the embeddings for the loss function
            embeddings = torch.nn.functional.normalize(embeddings, dim=1)
            loss = loss_func(embeddings, batch_labels)
            loss.backward()
            # torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            # Update the progress bar
            progress_bar.set_postfix({"loss": loss.item()})
            # Log the loss
            wandb.log({"loss": loss.item()}, step=training_state.step)
            training_state.step += 1
        print(f"Epoch {epoch} completed")
        # Run an evaluation
        eval_res = evaluate_model(training_state, training_config, model, dataloaders)
        wandb.log(eval_res, step=training_state.step)

if __name__ == "__main__":
    training_state = TrainingState()
    wandb.init(project="thesis-sup-con", mode="online")
    training_state.run_id = wandb.run.id

    base_path = Path(__file__).parent / f"experiments/chambon_sup_con/kaya/{training_state.run_id}"
    checkpoints_path = base_path / "checkpoints"
    checkpoints_path.mkdir(parents=True, exist_ok=True)
    visuals_path = base_path / "visuals"
    visuals_path.mkdir(parents=True, exist_ok=True)

    braindecode_dataset_base_path = Path(__file__).parent.parent / 'datasets' / 'braindecode' / 'datasets_data'
    kaya_dataset_path = Path(__file__).parent.parent / 'datasets' / 'large_eeg' / 'datasets_data'
    # raw_datasets = {
    #     # 'physionet': (braindecode_dataset_base_path / 'physionet', DatasetSplitConfig(train_prop=1, extrap_val_prop=0.0, extrap_test_prop=0, intra_val_prop=0.0, intra_test_prop=0)),  # only has one session so invalid for evaluation
    #     'lee2019': (braindecode_dataset_base_path / 'lee2019_512_ica_highpass_filtered_resampled_120', DatasetSplitConfig(train_prop=0.8, extrap_val_prop=0.1, extrap_test_prop=0, intra_val_prop=0.1, intra_test_prop=0)),
    #     'kaya': (kaya_dataset_path, DatasetSplitConfig(train_prop=0.80, extrap_val_prop=0.15, extrap_test_prop=0, intra_val_prop=0.05, intra_test_prop=0))
    # }
    # Update: Now using a consistent extrap set for all experiments. This also forces a consistent intra set, but the exact split in each run is not consistent
    raw_datasets = {
        'lee2019_train': (braindecode_dataset_base_path / 'lee2019_filtered_train_intra_set', DatasetSplitConfig(train_prop=0.9, extrap_val_prop=0, extrap_test_prop=0, intra_val_prop=0.1, intra_test_prop=0)),
        'lee2019_extrap': (braindecode_dataset_base_path / 'lee2019_filtered_extrap_set', DatasetSplitConfig(train_prop=0, extrap_val_prop=1, extrap_test_prop=0, intra_val_prop=0, intra_test_prop=0)),
        'kaya_train': (kaya_dataset_path / 'filtered_train_intra_set', DatasetSplitConfig(train_prop=0.95, extrap_val_prop=0, extrap_test_prop=0, intra_val_prop=0.05, intra_test_prop=0)),
        'kaya_extrap': (kaya_dataset_path / 'filtered_extrap_set', DatasetSplitConfig(train_prop=0, extrap_val_prop=1, extrap_test_prop=0, intra_val_prop=0, intra_test_prop=0))
    }

    preprocess_config = MetaPreprocessorConfig(
        sample_rate=120,
        clamping_sd=5,
        target_sample_rate=None,
        use_baseline_correction=True,
        use_robust_scaler=True,
        use_clamping=True
    )

    augmentation_config = SessionVarianceAugmentationConfig(
        apply_gaussian_noise=False,
        relative_noise_amplitude=0.01,
        apply_per_channel_amplitude_scaling=False,
        min_amplitude_scaling=0.9,
        max_amplitude_scaling=1.1,
    )

    train_dataloader_config = DataloaderConfig(
        batch_size = 64,
        shuffle = True,
    )
    eval_dataloader_config = DataloaderConfig(
        batch_size = 16,
        shuffle = False,
    )
    base_dataset_config = ContrastiveSubjectDatasetConfig(
        toggle_direct_sum_on=False,
        window_size_s=64,
        window_stride_s=2,
        sample_over_subjects_toggle=True,
        positive_separate_session=True,
        n_pos=1,
        n_neg=0,
        channel_blacklist=["EMG1", "EMG2", "EMG3", "EMG4", "AFz", "STI 014", "Nz"],

        preprocessor_config=preprocess_config,
        augmentation_config=augmentation_config,
    )
    train_dataset_config = base_dataset_config.copy()
    train_dataset_config.sample_over_subjects_toggle = True
    train_dataset_config.n_pos = 1  # So anchor+positive = total number of positive samples for SupCon

    eval_dataset_config = base_dataset_config.copy()
    eval_dataset_config.sample_over_subjects_toggle = False
    eval_dataset_config.n_pos = 0  # We only use anchors for evaluation
    eval_dataset_config.augmentation_config = None  # Don't use augmentation for evaluation
    dataset_configs = {
        "train": (train_dataset_config, train_dataloader_config),
        "extrap_val": (eval_dataset_config, eval_dataloader_config),
        "extrap_test": (eval_dataset_config, eval_dataloader_config),
        "intra_val": (eval_dataset_config, eval_dataloader_config),
        "intra_test": (eval_dataset_config, eval_dataloader_config),
    }
    loaders, num_channels, freq = get_dataloaders(raw_datasets, dataset_configs)

    print("Train Dataset:")
    loaders["train"].dataset.description()

    print("Intra-Val Dataset:")
    loaders["intra_val"].dataset.description()

    # for i in range(10):
    #     loaders["train"].dataset.visualize_item(i)

    training_config = TrainingConfig(
        device="mps",
        checkpoint_path=checkpoints_path,
        evaluation_visualization_path=visuals_path,
        load_checkpoint=False,
        checkpoint_load_path=None,
        evaluate_first=True,
        epochs=500,
        epoch_length=512,
        evaluation_k=5,

        run_extrap_val=True,
        run_intra_val=True,

        loss_temperature=0.1,
    )

    # model_config = ChambonExtendableConfig(C=num_channels, T=base_dataset_config.window_size_s*freq, k=63, m=2, num_blocks=9, D=512)
    # print("Creating model")
    # model, optimizer = create_model(training_config, model_config)

    model_config = ChambonConfig(C=num_channels, T=base_dataset_config.window_size_s*freq, k=63, m=16)
    print("Creating model")
    model, optimizer = create_model_old(training_config, model_config)

    if training_config.load_checkpoint:
        load_checkpoint(training_state, training_config, model, optimizer)

    all_configs = {
        'train_dataset_config': train_dataset_config,
        'eval_dataset_config': eval_dataset_config,
        'train_dataloader_config': train_dataloader_config,
        'eval_dataloader_config': eval_dataloader_config,
        'raw_datasets': raw_datasets,
        'model_config': model_config,
        'training_config': training_config,
    }
    all_configs_model = RootModel(all_configs)
    all_configs_dict = all_configs_model.model_dump()
    wandb.config.update(all_configs_dict)

    print("Starting training")
    train(training_state, training_config, model, optimizer, loaders)

    