from pathlib import Path

import wandb
import matplotlib.pyplot as plt
import torch
import numpy as np
from tqdm import tqdm
from torchinfo import summary

from thesis.structs import *

from thesis.datasets import get_dataloaders
from thesis.models import construct_model, save_checkpoint, load_checkpoint, ContrastiveModel
from thesis.loss_functions import WeightedSupConLoss
from thesis.evaluation import *


def evaluate_model(training_state: TrainingState, training_config: TrainingConfig, model: ContrastiveModel, dataloaders):
    extrap_dataloader = dataloaders["extrap_val"]
    intra_dataloader = dataloaders["intra_val"]
    visualization_path = training_state.visualization_path
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
        intra_embeddings, intra_labels, intra_sessions = get_knn_embeddings(training_config, model, intra_dataloader, limit=None)
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
        extrap_embeddings, extrap_labels, extrap_sessions = get_knn_embeddings(training_config, model, extrap_dataloader, limit=None)
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

    return res


def train(training_state: TrainingState, config: Config, model, optimizer, dataloaders):
    training_config = config.training_config

    unique_subject_id_to_label = { id: i for i, id in enumerate(dataloaders['train'].dataset.unique_subjects) }
    loss_func = WeightedSupConLoss(temperature=training_config.loss_temperature)

    if training_config.evaluate_first:
        # Run an initial evaluation
        eval_res = evaluate_model(training_state, training_config, model, dataloaders)
        wandb.log(eval_res, step=training_state.step)

    for epoch in range(training_state.epoch, training_config.epochs):
        training_state.epoch = epoch
        wandb.log({"epoch": epoch}, step=training_state.step)
        checkpoint_name = f"epoch_{epoch-1}.pt"
        checkpoint_path = training_state.checkpoint_path / checkpoint_name
        print(f"Saving checkpoint to {checkpoint_path}")
        save_checkpoint(checkpoint_path, model, optimizer, config, training_state)

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

    args.add_argument("--config-path", type=Path, required=False)
    args.add_argument("--checkpoint-path", type=Path, required=False)

    args.add_argument("--train-batch-size-override", type=int, required=False)
    args.add_argument("--loss-temperature-override", type=float, required=False)
    args.add_argument("--same-session-suppression-override", type=float, required=False)
    args.add_argument("--lr-override", type=float, required=False)
    args.add_argument("--sample-rate-override", type=float, required=False)
    args.add_argument("--low-freq-cutoff-override", type=float, required=False)
    args.add_argument("--high-freq-cutoff-override", type=float, required=False)
    args.add_argument("--channel-whitelist-override", type=str, required=False)
    args.add_argument("--window-size-s-override", type=float, required=False)

    args.add_argument("--epochs-override", type=int, required=False)
    args.add_argument("--epoch-length-override", type=int, required=False)

    args.add_argument("--wandb-project-override", type=str, required=False)
    args.add_argument("--wandb-run-name-override", type=str, required=False)

    args.add_argument("--wandb-offline", action="store_true")
    args.add_argument("--wandb-online", action="store_true")

    args.add_argument("--hack", action="store_true")

    args = args.parse_args()

    # Basic validation
    if args.config_path is None and args.checkpoint_path is None:
        raise ValueError("Must specify either a config path or a checkpoint path")

    if args.wandb_offline and args.wandb_online:
        raise ValueError("Cannot specify both --wandb-offline and --wandb-online")

    if args.config_path is not None:
        # Read the config file into a dict
        if not args.config_path.exists():
            raise FileNotFoundError(f"Config file {args.config_path} does not exist")
        if args.config_path.suffix == ".json":
            import json
            with open(args.config_path, "r") as f:
                config = json.load(f)
        elif args.config_path.suffix == ".yaml" or args.config_path.suffix == ".yml":
            import yaml
            with open(args.config_path, "r") as f:
                config = yaml.safe_load(f)
        config: Config = Config.model_validate(config)
    else:
        config = None

    if args.checkpoint_path is not None:
        # Then we are loading from a checkpoint
        if not args.checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint file {args.checkpoint_path} does not exist")
        if args.checkpoint_path.suffix != ".pt":
            raise ValueError(f"Checkpoint file {args.checkpoint_path} must be a .pt file")
        
        model, optimizer, checkpoint_config, checkpoint_training_state = load_checkpoint(args.checkpoint_path, current_config=config)

        if config is None:
            # Then we need to load the config from the checkpoint
            config = checkpoint_config
    else:
        model = None
        optimizer = None
        checkpoint_config = None
        checkpoint_training_state = None

    training_config = config.training_config

    # Update the config with overrides from the args
    if args.hack:
        # Then we overwrite the wandb config to be offline and update the datasets to be only a small subset
        training_config.wandb_online = False

        v2_braindecode_dataset_base_path = Path(__file__).parent.parent / 'datasets' / 'braindecode' / 'datasets_data_v2'
        hacking_dataset_config = DatasetConfig(
            name = "lee2019_train",
            path = v2_braindecode_dataset_base_path / 'lee2019_512_highpass_filtered' / 'train_intra_set',
            split_config = DatasetSplitConfig(train_prop=0.65, extrap_val_prop=0.25, extrap_test_prop=0, intra_val_prop=0.1, intra_test_prop=0),
            max_subjects = 4
        )
        config.datasets = [hacking_dataset_config]

    if args.train_batch_size_override is not None:
        config.train_dataloader_config.batch_size = args.train_batch_size_override

    if args.loss_temperature_override is not None:
        training_config.loss_temperature = args.loss_temperature_override

    if args.same_session_suppression_override is not None:
        training_config.same_session_suppression = args.same_session_suppression_override

    if args.lr_override is not None:
        training_config.lr = args.lr_override

    if args.sample_rate_override is not None:
        config.load_time_preprocess_config.target_sample_rate = args.sample_rate_override

    if args.low_freq_cutoff_override is not None:
        config.load_time_preprocess_config.band_pass_lower_cutoff = args.low_freq_cutoff_override

    if args.high_freq_cutoff_override is not None:
        config.load_time_preprocess_config.band_pass_upper_cutoff = args.high_freq_cutoff_override

    if args.channel_whitelist_override is not None:
        config.target_channels = get_channel_whitelist(args.channel_whitelist_override)

    if args.window_size_s_override is not None:
        config.window_size_s = args.window_size_s_override

    if args.epochs_override is not None:
        training_config.epochs = args.epochs_override

    if args.epoch_length_override is not None:
        training_config.epoch_length = args.epoch_length_override

    if args.wandb_project_override is not None:
        training_config.wandb_project = args.wandb_project_override

    if args.wandb_run_name_override is not None:
        training_config.wandb_run_name = args.wandb_run_name_override

    # Set the random seed from the config
    seed = config.random_seed
    torch.manual_seed(seed)
    np.random.seed(seed)

    # Verify that the datasets exist
    for dataset_config in config.datasets:
        if not dataset_config.path.exists():
            raise FileNotFoundError(f"Dataset path {dataset_config.path} does not exist")

    # Set up datasets
    base_dataset_config = ContrastiveSubjectDatasetConfig(
        target_channels=config.target_channels,
        toggle_direct_sum_on=False,
        window_size_s=config.window_size_s,
        window_stride_s=config.window_stride_s,
        sample_over_subjects_toggle=True,
        positive_separate_session=True,
        n_pos=1,
        n_neg=0,
        channel_blacklist=config.channel_blacklist,

        preprocessor_config=config.inference_time_preprocess_config,
        augmentation_config=config.augmentation_config,
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
        "train": (train_dataset_config, config.train_dataloader_config),
        "extrap_val": (eval_dataset_config, config.eval_dataloader_config),
        "extrap_test": (eval_dataset_config, config.eval_dataloader_config),
        "intra_val": (eval_dataset_config, config.eval_dataloader_config),
        "intra_test": (eval_dataset_config, config.eval_dataloader_config),
        "downstream": (downstream_dataset_config, config.eval_dataloader_config)
    }

    loaders, num_channels, freq = get_dataloaders(config.datasets, dataset_configs, load_time_preprocessor_config=config.load_time_preprocess_config)

    # Set up model if it was not previously loaded from a checkpoints
    
    if model is None:
        # Check if we need to update the model config given the frequency of the dataset and the channels
        model_config = config.embedding_model_config
        if model_config.C is None:
            model_config.C = num_channels
        if model_config.T is None:
            model_config.T = int(base_dataset_config.window_size_s*freq)
        # Then we are training from scratch
        model, optimizer = construct_model(config.embedding_model_config, config.head_config, device=training_config.device, lr=training_config.lr)
        summary(model, (1, model_config.C, model_config.T), col_names=("input_size", "output_size", "num_params"))
        checkpoint_config = None
        checkpoint_training_state = None
    
    training_state = TrainingState() if checkpoint_training_state is None else checkpoint_training_state
    wandb_mode = "online" if training_config.wandb_online else "offline"
    if args.wandb_offline:
        wandb_mode = "offline"
    if args.wandb_online:
        wandb_mode = "online"
    wandb.init(project=training_config.wandb_project, mode=wandb_mode, name=training_config.wandb_run_name)
    training_state.run_id = wandb.run.id

    # Create the data folder if it doesn't exist
    training_config.data_path.mkdir(parents=True, exist_ok=True)
    save_path = training_config.data_path / training_state.run_id
    # And the /checkpoints and /visualizations folders
    checkpoint_path = save_path / "checkpoints"
    visualization_path = save_path / "visualizations"
    checkpoint_path.mkdir(parents=True, exist_ok=True)
    visualization_path.mkdir(parents=True, exist_ok=True)

    training_state.checkpoint_path = checkpoint_path
    training_state.visualization_path = visualization_path

    config_dict = config.model_dump()
    wandb.config.update(config_dict)

    train(training_state, config, model, optimizer, loaders)