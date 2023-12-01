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
from thesis.datasets.subject_dataset import load_subject_datasets, concatenate_datasets, get_dataset_split, SubjectDataset, get_subject_datasets_in_dir, load_subject_metadata
from thesis.datasets.dataset import ContrastiveSubjectDataset, ContrastiveSubjectDatasetConfig, get_contrastive_subject_loaders, get_contrastive_subject_loader
from thesis.preprocessing import MetaPreprocessorConfig, LoadTimePreprocessorConfig
from thesis.augmentation import SessionVarianceAugmentationConfig
from thesis.structs.training_structs import TrainingConfig, TrainingState
from thesis.structs.dataset_structs import DatasetSplitConfig, DataloaderConfig
from thesis.loss_functions.sup_con import SupConLoss
from thesis.loss_functions.weighted_sup_con import WeightedSupConLoss
from thesis.evaluation.knn import get_embeddings, get_cross_session_knn_metrics, visualize_embeddings, visualize_confusion_matrix
from thesis.evaluation.downstream import compute_LDA_metrics, get_embeddings as get_downstream_embeddings

from thesis.models.chambon_extendable import ChambonNetExtendable
from thesis.models.chambon import ChambonNet
from thesis.models.contrastive_head import HeadStyle, ContrastiveModel

import wandb
from tqdm import tqdm

def create_model(t_config: TrainingConfig, m_config: ChambonExtendableConfig, contrastive_loss_size: int = 128):
    model = ChambonNetExtendable(m_config)
    test_data = torch.randn(1, m_config.C, m_config.T)
    test_output = model(test_data)
    embedding_size = test_output.shape[1]
    head_config = ContrastiveHeadConfig(logit_dimension=embedding_size, c_loss_dimension=contrastive_loss_size, head_style=HeadStyle.LINEAR, layer_sizes=[contrastive_loss_size])
    # full_model = create_contrastive_module(model, head_config)
    full_model = ContrastiveModel(model, head_config)

    # Move the model to the device
    full_model = full_model.to(t_config.device)

    # Create the optimizer
    optimizer = torch.optim.Adam(full_model.parameters(), lr=1e-4)

    return full_model, optimizer

def create_model_old(t_config: TrainingConfig, m_config: ChambonConfig, contrastive_loss_size: int = 128) -> tuple[ContrastiveModel, torch.optim.Optimizer]:
    model = ChambonNet(m_config)
    test_data = torch.randn(1, m_config.C, m_config.T)
    test_output = model(test_data)
    embedding_size = test_output.shape[1]
    print("Logit Size", embedding_size)
    head_config = ContrastiveHeadConfig(logit_dimension=embedding_size, c_loss_dimension=contrastive_loss_size, head_style=HeadStyle.LINEAR, layer_sizes=[contrastive_loss_size])
    # full_model = create_contrastive_module(model, head_config)
    full_model = ContrastiveModel(model, head_config)

    # Move the model to the device
    full_model = full_model.to(t_config.device)

    # Create the optimizer
    optimizer = torch.optim.Adam(full_model.parameters(), lr=1e-3)

    return full_model, optimizer

def evaluate_model(training_state: TrainingState, training_config: TrainingConfig, model: ContrastiveModel, dataloaders):
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
        intra_embeddings, intra_labels, intra_sessions = get_embeddings(training_config, model, intra_dataloader, limit=None)
        intra_cross_session_knn_metrics = get_cross_session_knn_metrics(
            intra_embeddings, intra_labels, intra_sessions,
            k=training_config.evaluation_k, metric='cosine'
        )
        res["intra_top_1_accuracy"] = intra_cross_session_knn_metrics["total_top_1_accuracy"]
        res["intra_improved_top_1_accuracy"] = intra_cross_session_knn_metrics["improved_total_top_1_accuracy"]
        res[f"intra_top_{training_config.evaluation_k}_accuracy"] = intra_cross_session_knn_metrics["total_top_k_accuracy"]
        res[f"intra_improved_top_{training_config.evaluation_k}_accuracy"] = intra_cross_session_knn_metrics["improved_total_top_k_accuracy"]

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
        res["extrap_improved_top_1_accuracy"] = extrap_cross_session_knn_metrics["improved_total_top_1_accuracy"]
        res[f"extrap_top_{training_config.evaluation_k}_accuracy"] = extrap_cross_session_knn_metrics["total_top_k_accuracy"]
        res[f"extrap_improved_top_{training_config.evaluation_k}_accuracy"] = extrap_cross_session_knn_metrics["improved_total_top_k_accuracy"]

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

    if training_config.run_downstream_eval:
        embedding_model = model.embedding_module
        downstream_embeddings, downstream_metadatas = get_downstream_embeddings(training_config, embedding_model, dataloaders["downstream"])
        # Now we need to compute the LDA metrics
        for metadata_key in training_config.downstream_lda_metadata_keys:
            lda_metrics = compute_LDA_metrics(downstream_embeddings, downstream_metadatas, metadata_key, separate_subjects=True, num_folds=training_config.downstream_num_folds)
            for metric_name, metric_value in lda_metrics.items():
                res[f"downstream_{metric_name}_{metadata_key}"] = metric_value

    print("Evaluation:", res)
    return res

    
def get_dataloaders(
    base_datasets: dict[str, tuple[Path, DatasetSplitConfig] | tuple[Path, DatasetSplitConfig, int]],  # Path, split config, and optionally the number of subjects to use
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
    for dataset_name, load_config in base_datasets.items():
        if len(load_config) == 2:
            dataset_path, split_config = load_config
            n_subjects = None
        elif len(load_config) == 3:
            dataset_path, split_config, n_subjects = load_config
        print(f"Loading dataset {dataset_name}")
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
    # loss_func = SupConLoss(temperature=training_config.loss_temperature)
    loss_func = WeightedSupConLoss(temperature=training_config.loss_temperature)

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
            anchor_subject_labels = torch.tensor([unique_subject_id_to_label[data['unique_subject_id']] for data in batch["anchor_metadata"]])
            anchor_session_hashes = torch.tensor([hash(data['dataset_key'][1]) for data in batch["anchor_metadata"]])
            positives = batch["positive"]
            if positives.ndim == 4:  # If there is more than one positive sample per anchor
                # The positive labels are the anchor labels replicated across a new axis 1, n_pos times
                positive_subject_labels = anchor_subject_labels.unsqueeze(1).repeat(1, positives.shape[1])
                positives = positives.reshape(-1, positives.shape[-2], positives.shape[-1])
                positive_subject_labels = positive_labels.reshape(-1)

                # The session are different between the anchors and positives, so we need to compute them separately
                positive_session_hashes = torch.tensor([hash(data['dataset_key'][1]) for data in batch["positive_metadata"]] * positives.shape[1])
                positives = positives.reshape(-1, positives.shape[-2], positives.shape[-1])
                positive_session_hashes = positive_session_hashes.reshape(-1)
            else:  # If there is only one positive sample per anchor
                # The the positive subject ids are just the anchor labels
                positive_subject_labels = anchor_subject_labels

                # Again, we need to compute the session hashes separately
                positive_session_hashes = torch.tensor([hash(data['dataset_key'][1]) for data in batch["positive_metadata"]])
            
            # Now we can create the full batch by concatenating the anchors and positives
            batch_windows = torch.cat([anchors, positives], dim=0)
            batch_subject_labels = torch.cat([anchor_subject_labels, positive_subject_labels], dim=0)
            # The session labels are more complicated. Each label is [subject_label, session_hash] so we need to produce an nx2 tensor
            batch_session_hashes = torch.cat([anchor_session_hashes, positive_session_hashes], dim=0)
            batch_session_labels = torch.stack([batch_subject_labels, batch_session_hashes], dim=1)
            # Convert all samples to float
            batch_windows = batch_windows.float()
            batch_subject_labels = batch_subject_labels.float()
            batch_session_labels = batch_session_labels.float()
            # Move the samples to the device
            batch_windows = batch_windows.to(training_config.device)
            batch_subject_labels = batch_subject_labels.to(training_config.device)
            batch_session_labels = batch_session_labels.to(training_config.device)

            embeddings = model(batch_windows)
            # Normalize the embeddings for the loss function
            embeddings = torch.nn.functional.normalize(embeddings, dim=1)
            # loss = loss_func(embeddings, batch_subject_labels)
            loss = loss_func(embeddings, [
                (1.0, batch_subject_labels),  # Baseline positive weight for same-subject samples
                (-training_config.same_session_suppression, batch_session_labels)  # Reduce the weight on same-session samples to promote cross-session invariance
            ])
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

def get_channel_whitelist(whitelist_str: str) -> list[str]:
    """
    Parses a comma separated string of channel names into a list of channel names
    """
    return [channel.strip() for channel in whitelist_str.split(",")]

if __name__ == "__main__":
    import argparse
    args = argparse.ArgumentParser()
    # Do we evaluate before training?
    args.add_argument("--evaluate_first", action="store_true")
    # Which evaluations do we run?
    args.add_argument("--run_intra_val", action="store_true")
    args.add_argument("--run_extrap_val", action="store_true")
    args.add_argument("--run_downstream_eval", action="store_true")
    # How many epochs do we train for? How many iterations per epoch? What train batch size?
    args.add_argument("--epochs", type=int, default=20)
    args.add_argument("--epoch_length", type=int, default=1024)
    args.add_argument("--train_batch_size", type=int, default=64)
    # What loss temperature do we use?
    args.add_argument("--loss_temperature", type=float, default=0.05)
    args.add_argument("--same_session_suppression", type=float, default=0.0)
    # Load time preprocessing config
    args.add_argument("--sample_rate", type=int, default=120)
    args.add_argument("--low_freq_cutoff", type=float, default=None)
    args.add_argument("--high_freq_cutoff", type=float, default=None)
    # Channel whitelist
    args.add_argument("--channel_whitelist", type=str, default="C3,C4,Cz,F3,F4,F7,F8,Fp1,Fp2,Fz,O1,O2,P3,P4,P7,P8,Pz,T7,T8")
    # What window size do we use?
    args.add_argument("--window_size_s", type=float, default=8)
    # Which checkpoint and should we load it?
    args.add_argument("--checkpoint_load_path", type=str, default=None)
    args.add_argument("--load_checkpoint", action="store_true")
    # Go online? What project?
    args.add_argument("--wandb_online", action="store_true")
    args.add_argument("--wandb_project", type=str, default="thesis")
    args.add_argument("--wandb_run_name", type=str, default=None)
    # And testing parameters
    args.add_argument("--hack", action="store_true")

    # Parse the arguments
    args = args.parse_args()

    if args.channel_whitelist is not None:
        channel_whitelist = get_channel_whitelist(args.channel_whitelist)
    else:
        channel_whitelist = None


    training_state = TrainingState()
    wandb.init(project=args.wandb_project, mode="online" if args.wandb_online else "offline", name=args.wandb_run_name)
    training_state.run_id = wandb.run.id

    base_path = Path(__file__).parent / f"experiments/chambon/{training_state.run_id}"
    checkpoints_path = base_path / "checkpoints"
    checkpoints_path.mkdir(parents=True, exist_ok=True)
    visuals_path = base_path / "visuals"
    visuals_path.mkdir(parents=True, exist_ok=True)

    braindecode_dataset_base_path = Path(__file__).parent.parent / 'datasets' / 'braindecode' / 'datasets_data'
    kaya_dataset_path = Path(__file__).parent.parent / 'datasets' / 'large_eeg' / 'datasets_data'
    # Update: Now using a consistent extrap set for all experiments. This also forces a consistent intra set, but the exact split in each run is not consistent
    v1_raw_datasets = {
        'physionet': (braindecode_dataset_base_path / 'physionet', DatasetSplitConfig(train_prop=1, extrap_val_prop=0.0, extrap_test_prop=0, intra_val_prop=0.0, intra_test_prop=0, downstream_num_subjects=0)),  # only has one session so invalid for evaluation
        'lee2019_train': (braindecode_dataset_base_path / 'lee2019_filtered_train_intra_set', DatasetSplitConfig(train_prop=0.9, extrap_val_prop=0, extrap_test_prop=0, intra_val_prop=0.1, intra_test_prop=0)),
        # 'lee2019_train_intra_only': (braindecode_dataset_base_path / 'lee2019_filtered_train_intra_set', DatasetSplitConfig(train_prop=0, extrap_val_prop=0, extrap_test_prop=0, intra_val_prop=1, intra_test_prop=0), 6),
        'lee2019_extrap': (braindecode_dataset_base_path / 'lee2019_filtered_extrap_set', DatasetSplitConfig(train_prop=0, extrap_val_prop=1, extrap_test_prop=0, intra_val_prop=0, intra_test_prop=0)),
        'kaya_train': (kaya_dataset_path / 'filtered_train_intra_set', DatasetSplitConfig(train_prop=0.95, extrap_val_prop=0, extrap_test_prop=0, intra_val_prop=0.05, intra_test_prop=0, downstream_num_subjects=0)),
        # 'kaya_train_intra_only': (kaya_dataset_path / 'filtered_train_intra_set', DatasetSplitConfig(train_prop=0, extrap_val_prop=0, extrap_test_prop=0, intra_val_prop=1, intra_test_prop=0), 2),
        'kaya_extrap': (kaya_dataset_path / 'filtered_extrap_set', DatasetSplitConfig(train_prop=0, extrap_val_prop=1, extrap_test_prop=0, intra_val_prop=0, intra_test_prop=0, downstream_num_subjects=0))
    }
    v1_raw_hacking_datasets = {
        # 'physionet': (braindecode_dataset_base_path / 'physionet', DatasetSplitConfig(train_prop=1, extrap_val_prop=0.0, extrap_test_prop=0, intra_val_prop=0.0, intra_test_prop=0, downstream_num_subjects=0), 10),  # only has one session so invalid for evaluation
        'lee2019_train': (braindecode_dataset_base_path / 'lee2019_filtered_train_intra_set', DatasetSplitConfig(train_prop=0.8, extrap_val_prop=0.1, extrap_test_prop=0, intra_val_prop=0.1, intra_test_prop=0), 10),
    }

    v2_braindecode_dataset_base_path = Path(__file__).parent.parent / 'datasets' / 'braindecode' / 'datasets_data_v2'
    v2_kaya_dataset_path = Path(__file__).parent.parent / 'datasets' / 'large_eeg' / 'datasets_data' / 'kaya_v2'

    v2_raw_datasets = {
        'physionet': (v2_braindecode_dataset_base_path / 'physionet_160', DatasetSplitConfig(train_prop=1, extrap_val_prop=0.0, extrap_test_prop=0, intra_val_prop=0.0, intra_test_prop=0, downstream_num_subjects=0)),  # only has one session so invalid for evaluation
        'lee2019_train': (v2_braindecode_dataset_base_path / 'lee2019_512_highpass_filtered' / 'train_intra_set', DatasetSplitConfig(train_prop=0.9, extrap_val_prop=0, extrap_test_prop=0, intra_val_prop=0.1, intra_test_prop=0)),
        'lee2019_extrap': (v2_braindecode_dataset_base_path / 'lee2019_512_highpass_filtered' / 'extrap_set', DatasetSplitConfig(train_prop=0, extrap_val_prop=1, extrap_test_prop=0, intra_val_prop=0, intra_test_prop=0)),
        'kaya_train': (v2_kaya_dataset_path / 'train_intra_set', DatasetSplitConfig(train_prop=0.95, extrap_val_prop=0, extrap_test_prop=0, intra_val_prop=0.05, intra_test_prop=0, downstream_num_subjects=0)),
        'kaya_extrap': (v2_kaya_dataset_path / 'extrap_set', DatasetSplitConfig(train_prop=0, extrap_val_prop=1, extrap_test_prop=0, intra_val_prop=0, intra_test_prop=0, downstream_num_subjects=0))
    }

    v2_raw_hacking_datasets = {
        # 'physionet': (v2_braindecode_dataset_base_path / 'physionet_160', DatasetSplitConfig(train_prop=1, extrap_val_prop=0.0, extrap_test_prop=0, intra_val_prop=0.0, intra_test_prop=0, downstream_num_subjects=0), 10),  # only has one session so invalid for evaluation
        'lee2019_train': (v2_braindecode_dataset_base_path / 'lee2019_512_highpass_filtered' / 'train_intra_set', DatasetSplitConfig(train_prop=0.8, extrap_val_prop=0.1, extrap_test_prop=0, intra_val_prop=0.1, intra_test_prop=0), 3),
    }

    raw_datasets = v2_raw_hacking_datasets if args.hack else v2_raw_datasets

    preprocess_config = MetaPreprocessorConfig(
        sample_rate=args.sample_rate,
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

        apply_per_channel_time_offset=False,
        min_time_offset = -1,
        max_time_offset = 1
    )

    load_time_preprocess_config = LoadTimePreprocessorConfig(
        target_sample_rate=args.sample_rate,
        band_pass_lower_cutoff=args.low_freq_cutoff,
        band_pass_upper_cutoff=args.high_freq_cutoff,
    )

    train_dataloader_config = DataloaderConfig(
        batch_size = args.train_batch_size,
        shuffle = True,
    )
    eval_dataloader_config = DataloaderConfig(
        batch_size = 16,
        shuffle = False,
    )
    base_dataset_config = ContrastiveSubjectDatasetConfig(
        # target_channels=['C3', 'C4', 'Cz', 'F3', 'F4', 'F7', 'F8', 'Fp1', 'Fp2', 'Fz', 'O1', 'O2', 'P3', 'P4', 'P7', 'P8', 'Pz', 'T7', 'T8'],
        target_channels=channel_whitelist,
        toggle_direct_sum_on=False,
        window_size_s=args.window_size_s,
        window_stride_s=2,
        sample_over_subjects_toggle=True,
        positive_separate_session=True,
        n_pos=1,
        n_neg=0,
        channel_blacklist=["EMG1", "EMG2", "EMG3", "EMG4", "AFz", "STI 014", "Nz"],

        preprocessor_config=preprocess_config,
        augmentation_config=augmentation_config,
    )
    train_dataset_config = base_dataset_config.model_copy()
    train_dataset_config.sample_over_subjects_toggle = True
    train_dataset_config.n_pos = 1  # So anchor+positive = total number of positive samples for SupCon

    eval_dataset_config = base_dataset_config.model_copy()
    eval_dataset_config.sample_over_subjects_toggle = False
    eval_dataset_config.n_pos = 0  # We only use anchors for evaluation
    eval_dataset_config.augmentation_config = None  # Don't use augmentation for evaluation

    downstream_dataset_config = base_dataset_config.model_copy()
    downstream_dataset_config.sample_over_subjects_toggle = False
    downstream_dataset_config.n_pos = 0  # We only use anchors for evaluation
    downstream_dataset_config.augmentation_config = None  # Don't use augmentation for evaluation
    downstream_dataset_config.max_samples_per_subject = 1000

    dataset_configs = {
        "train": (train_dataset_config, train_dataloader_config),
        "extrap_val": (eval_dataset_config, eval_dataloader_config),
        "extrap_test": (eval_dataset_config, eval_dataloader_config),
        "intra_val": (eval_dataset_config, eval_dataloader_config),
        "intra_test": (eval_dataset_config, eval_dataloader_config),
        "downstream": (downstream_dataset_config, eval_dataloader_config)
    }
    loaders, num_channels, freq = get_dataloaders(raw_datasets, dataset_configs, load_time_preprocessor_config=load_time_preprocess_config)

    print("Train Dataset:")
    loaders["train"].dataset.description()

    if loaders["intra_val"] is not None:
        print("Intra-Val Dataset:")
        loaders["intra_val"].dataset.description()

    if loaders["extrap_val"] is not None:
        print("Extrap-Val Dataset:")
        loaders["extrap_val"].dataset.description()

    # for i in range(10):
    #     loaders["train"].dataset.visualize_item(i)

    training_config = TrainingConfig(
        device="mps",
        checkpoint_path=checkpoints_path,
        evaluation_visualization_path=visuals_path,
        load_checkpoint=args.load_checkpoint,
        checkpoint_load_path=args.checkpoint_load_path,
        evaluate_first=args.evaluate_first,
        epochs=args.epochs,
        epoch_length=args.epoch_length,
        evaluation_k=5,

        run_extrap_val=args.run_extrap_val,
        run_intra_val=args.run_intra_val,

        run_downstream_eval=args.run_downstream_eval,
        downstream_lda_metadata_keys=["sex"],

        loss_temperature=args.loss_temperature,
        same_session_suppression=args.same_session_suppression,
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
        'args': vars(args),
        'train_dataset_config': train_dataset_config,
        'eval_dataset_config': eval_dataset_config,
        'train_dataloader_config': train_dataloader_config,
        'eval_dataloader_config': eval_dataloader_config,
        'raw_datasets': raw_datasets,
        'model_config': model_config,
        'training_config': training_config,
        'load_time_preprocess_config': load_time_preprocess_config,
    }
    all_configs_model = RootModel(all_configs)
    all_configs_dict = all_configs_model.model_dump()
    wandb.config.update(all_configs_dict)

    print("Starting training")
    train(training_state, training_config, model, optimizer, loaders)

    