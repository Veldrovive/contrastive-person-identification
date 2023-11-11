"""
Loads a chambon model and a linear head with a very small dataset and overfits it
"""

import torch
from pathlib import Path
import umap
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.neighbors import NearestNeighbors

from thesis.structs.chambon_structs import ChambonConfig
from thesis.structs.contrastive_head_structs import ContrastiveHeadConfig, HeadStyle
from thesis.datasets.braindecode.PhysionetMI import get_physionet_datasets, PhysionetMIDatasetConfig, get_physionet_dataloaders, DataloaderConfig
from thesis.preprocessing.meta import MetaPreprocessorConfig, preprocessor_factory
from thesis.structs.training_structs import TrainingConfig, TrainingState

from thesis.models.chambon import ChambonNet
from thesis.models.contrastive_head import HeadStyle, create_contrastive_module

import wandb
from tqdm import tqdm

def create_model(t_config: TrainingConfig, m_config: ChambonConfig, contrastive_loss_size: int = 128):
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

def triple_loss(anchor, positive, negative, margin: float = 1.0):
    """
    Assumes that the input is already embedded
    """
    loss = torch.nn.TripletMarginLoss(margin=margin)
    return loss(anchor, positive, negative)

def visualize_embeddings(embeddings, labels, title="UMAP Projection of Embeddings"):
    """
    Uses UMAP to visualize the manifold of the embeddings
    Returns a matplotlib figure

    Embeddings is a tensor of shape (num_samples, embedding_size)
    Labels is a list of length num_samples where each label is an integer
    """
    # Reduce dimensionality using UMAP
    reducer = umap.UMAP()
    embeddings_2d = reducer.fit_transform(embeddings)

    # Create a matplotlib figure
    fig, ax = plt.subplots()

    # Create a scatter plot
    scatter = ax.scatter(
        embeddings_2d[:, 0],
        embeddings_2d[:, 1],
        c=labels,
        cmap='Spectral',
        s=5
    )

    # Add a color bar
    colorbar = plt.colorbar(scatter)

    # Set titles and labels
    plt.title(title)

    return fig

def visualize_embeddings_confusion(all_embeddings, all_labels, nbrs, title="Confusion Matrix based on Nearest Neighbors"):
    # Step 1: Find the k nearest neighbors for each embedding
    distances, indices = nbrs.kneighbors(all_embeddings)

    all_labels = [int(label) for label in all_labels]
    
    # Step 2: Predict labels through majority voting
    predicted_labels_k = []
    predicted_labels_1 = []
    for idx in indices:
        # Remove the first index because it is the anchor itself
        idx = idx[1:]

        # Fetch the labels of the k nearest neighbor embeddings
        neighbor_labels = [all_labels[i] for i in idx]
        
        # Use majority voting to determine the label of the anchor
        most_common_label = max(set(neighbor_labels), key=neighbor_labels.count)
        predicted_labels_k.append(most_common_label)

        # Use the first neighbor as the prediction
        predicted_labels_1.append(neighbor_labels[0])
    
    # Step 3: Create the confusion matrix
    cm_k = confusion_matrix(all_labels, predicted_labels_k, labels=list(set(all_labels)))
    cm_1 = confusion_matrix(all_labels, predicted_labels_1, labels=list(set(all_labels)))

    num_labels = len(set(all_labels))
    use_annotations = num_labels < 10
    x_labels = list(set(all_labels)) if num_labels < 20 else False
    y_labels = list(set(all_labels)) if num_labels < 20 else False
    
    # Step 4: Visualize the confusion matrix
    fig_k, ax_k = plt.subplots(figsize=(10, 8))
    
    # Use seaborn to make the visualization more appealing
    sns.heatmap(cm_k, annot=use_annotations, fmt="d", cmap="Blues", ax=ax_k, xticklabels=x_labels, yticklabels=y_labels)
    
    ax_k.set_xlabel('Predicted labels with k nearest neighbors')
    ax_k.set_ylabel('True labels')
    ax_k.set_title(title)

    fig_1, ax_1 = plt.subplots(figsize=(10, 8))
    sns.heatmap(cm_1, annot=use_annotations, fmt="d", cmap="Blues", ax=ax_1, xticklabels=x_labels, yticklabels=y_labels)
    ax_1.set_xlabel('Predicted labels with 1 nearest neighbor')
    ax_1.set_ylabel('True labels')
    ax_1.set_title(title)
    
    return fig_k, fig_1

def evaluate_model(training_state: TrainingState, training_config: TrainingConfig, model, dataloaders):
    """
    Evaluates the model on the validation and test sets
    We have two evaluation sets that we evaluate in parallel. The extrapolation set and the intra-training set.
    The extrapolation set is the more difficult of the two and involves matching samples from subjects that the model has never seen before.
    The intra-training set checks if the model can match samples from individuals it has seen in the training set and should be the easier task.

    The evaluation is done by taking the precision@k and recall@k for the k most similar samples to the anchor sample. k defined by the training config
    """
    visualization_path = training_config.evaluation_visualization_path
    visualization_path.mkdir(parents=True, exist_ok=True)
    epoch = training_state.epoch

    intra_labeled_embeddings = {}  # Maps from the true label to a list of embeddings for the intra_val set
    extrap_labeled_embeddings = {}  # Maps from the true label to a list of embeddings for the extrap_val set
    # Now we iterate over the entire evaluation set and get the embeddings for each sample
    for batch in tqdm(dataloaders["intra_val"], desc="Evaluating (intra)"):
        anchors = batch["anchor_data"]
        labels = batch["anchor_label"]
        # Convert all samples to float
        anchors = anchors.float()
        # Move the samples to the device
        anchors = anchors.to(training_config.device)
        embeddings = model(anchors)
        for i in range(len(labels)):
            label = labels[i]
            embedding = embeddings[i]
            if label not in intra_labeled_embeddings:
                intra_labeled_embeddings[label] = []
            intra_labeled_embeddings[label].append(embedding)
    
    all_embeddings = []
    all_labels = []
    for label in intra_labeled_embeddings:
        all_embeddings.extend(intra_labeled_embeddings[label])
        all_labels.extend([label] * len(intra_labeled_embeddings[label]))

    # Now we have all the embeddings and labels for the intra set. Let's train a nearest neighbors model
    # Convert the embeddings to a numpy array
    all_embeddings = torch.stack(all_embeddings).detach().cpu().numpy()
    nbrs = NearestNeighbors(n_neighbors=training_config.evaluation_k + 1, algorithm="ball_tree").fit(all_embeddings)  # +1 because the nearest neighbor is the anchor itself

    # We use UMAP to visualize the embeddings
    visualized_intra_embeddings = visualize_embeddings(all_embeddings, all_labels, title="UMAP Projection of Intra-Training Set Embeddings")
    visualized_intra_embeddings_path = visualization_path / f"intra_embeddings_{epoch}.png"
    visualized_intra_embeddings.savefig(visualized_intra_embeddings_path)
    plt.close(visualized_intra_embeddings)
    # And then plot the confusion matrix
    visualized_intra_embeddings_confusion_k, visualized_intra_embeddings_confusion_1 = visualize_embeddings_confusion(all_embeddings, all_labels, nbrs, title="Confusion Matrix based on Nearest Neighbors for Intra-Training Set")
    visualized_intra_embeddings_confusion_path_k = visualization_path / f"intra_embeddings_confusion_{epoch}_k.png"
    visualized_intra_embeddings_confusion_path_1 = visualization_path / f"intra_embeddings_confusion_{epoch}_1.png"
    visualized_intra_embeddings_confusion_k.savefig(visualized_intra_embeddings_confusion_path_k)
    visualized_intra_embeddings_confusion_1.savefig(visualized_intra_embeddings_confusion_path_1)
    plt.close(visualized_intra_embeddings_confusion_k)
    plt.close(visualized_intra_embeddings_confusion_1)

    # Now we iterate over every sample in the intra set and get the k nearest neighbors labels as well as the true label
    # For now, let's just compute the proportion of the top k that are the true label on average
    # We will also compute the top 1 score
    total_correct = 0
    total_comparisons = 0
    top_1_correct_count = 0
    sum_total = 0
    for embedding, anchor_label in tqdm(zip(all_embeddings, all_labels), desc="Crunching (intra)"):
        distances, indices = nbrs.kneighbors([embedding])
        # Remove the first index because it is the anchor itself
        indices = indices[:, 1:]
        distances = distances[:, 1:]

        # Get the labels for the indices
        nearest_labels = [all_labels[i] for i in indices[0]]
        
        num_correct = sum([1 if label == anchor_label else 0 for label in nearest_labels])
        total = len(nearest_labels)
        top_1_score = 1 if nearest_labels[0] == anchor_label else 0

        total_correct += num_correct
        total_comparisons += total
        top_1_correct_count += top_1_score
        sum_total += 1

    intra_precision_at_k = total_correct / total_comparisons
    top_1_prop = top_1_correct_count / sum_total

    res = {
        f'intra_precision_at_{training_config.evaluation_k}': intra_precision_at_k,
        'intra_top_1': top_1_prop,
        'intra_total': sum_total,
        'intra_embedding_visualization': wandb.Image(visualized_intra_embeddings_path.absolute().as_posix()),
        'intra_embedding_confusion_k': wandb.Image(visualized_intra_embeddings_confusion_path_k.absolute().as_posix()),
        'intra_embedding_confusion_1': wandb.Image(visualized_intra_embeddings_confusion_path_1.absolute().as_posix()),
    }

    # Delete some stuff to try to get python to free up some memory
    del intra_labeled_embeddings
    del all_embeddings
    del all_labels
    del nbrs

    # Now we do the same thing for the extrapolation set
    for batch in tqdm(dataloaders["extrap_val"], desc="Evaluating (extrap)"):
        anchors = batch["anchor_data"]
        labels = batch["anchor_label"]
        # Convert all samples to float
        anchors = anchors.float()
        # Move the samples to the device
        anchors = anchors.to(training_config.device)
        embeddings = model(anchors)
        for i in range(len(labels)):
            label = labels[i]
            embedding = embeddings[i]
            if label not in extrap_labeled_embeddings:
                extrap_labeled_embeddings[label] = []
            extrap_labeled_embeddings[label].append(embedding)

    all_embeddings = []
    all_labels = []
    for label in extrap_labeled_embeddings:
        all_embeddings.extend(extrap_labeled_embeddings[label])
        all_labels.extend([label] * len(extrap_labeled_embeddings[label]))

    # Now we have all the embeddings and labels for the intra set. Let's train a nearest neighbors model
    # Convert the embeddings to a numpy array
    all_embeddings = torch.stack(all_embeddings).detach().cpu().numpy()
    nbrs = NearestNeighbors(n_neighbors=training_config.evaluation_k + 1, algorithm="ball_tree").fit(all_embeddings)  # +1 because the nearest neighbor is the anchor itself

    # We again use UMAP to visualize the embeddings
    visualized_extrap_embeddings = visualize_embeddings(all_embeddings, all_labels, title="UMAP Projection of Extrapolation Set Embeddings")
    visualized_extrap_embeddings_path = visualization_path / f"extrap_embeddings_{epoch}.png"
    visualized_extrap_embeddings.savefig(visualized_extrap_embeddings_path)
    plt.close(visualized_extrap_embeddings)
    # And then plot the confusion matrix
    visualized_extrap_embeddings_confusion_k, visualized_extrap_embeddings_confusion_1 = visualize_embeddings_confusion(all_embeddings, all_labels, nbrs, title="Confusion Matrix based on Nearest Neighbors for Extrapolation Set")
    visualized_extrap_embeddings_confusion_path_k = visualization_path / f"extrap_embeddings_confusion_{epoch}_k.png"
    visualized_extrap_embeddings_confusion_path_1 = visualization_path / f"extrap_embeddings_confusion_{epoch}_1.png"
    visualized_extrap_embeddings_confusion_k.savefig(visualized_extrap_embeddings_confusion_path_k)
    visualized_extrap_embeddings_confusion_1.savefig(visualized_extrap_embeddings_confusion_path_1)
    plt.close(visualized_extrap_embeddings_confusion_k)
    plt.close(visualized_extrap_embeddings_confusion_1)


    # Now we iterate over every sample in the intra set and get the k nearest neighbors labels as well as the true label
    # For now, let's just compute the proportion of the top k that are the true label on average
    # We will also compute the top 1 score
    total_correct = 0
    total_comparisons = 0
    top_1_correct_count = 0
    sum_total = 0
    for embedding, anchor_label in tqdm(zip(all_embeddings, all_labels), desc="Crunching (extrap)"):
        distances, indices = nbrs.kneighbors([embedding])
        # Remove the first index because it is the anchor itself
        indices = indices[:, 1:]
        distances = distances[:, 1:]

        # Get the labels for the indices
        nearest_labels = [all_labels[i] for i in indices[0]]
        
        num_correct = sum([1 if label == anchor_label else 0 for label in nearest_labels])
        total = len(nearest_labels)
        top_1_score = 1 if nearest_labels[0] == anchor_label else 0

        total_correct += num_correct
        total_comparisons += total
        top_1_correct_count += top_1_score
        sum_total += 1

    extrap_precision_at_k = total_correct / total_comparisons
    top_1_prop = top_1_correct_count / sum_total

    res.update({
        f'extrap_precision_at_{training_config.evaluation_k}': extrap_precision_at_k,
        'extrap_top_1': top_1_prop,
        'extrap_total': sum_total,
        'extrap_embedding_visualization': wandb.Image(visualized_extrap_embeddings_path.absolute().as_posix()),
        'extrap_embedding_confusion_k': wandb.Image(visualized_extrap_embeddings_confusion_path_k.absolute().as_posix()),
        'extrap_embedding_confusion_1': wandb.Image(visualized_extrap_embeddings_confusion_path_1.absolute().as_posix()),
    })

    print("Evaluation:", res)
    return res

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
    """
    Trains the model for the specified number of epochs
    """
    # Get the total number of training parameters in the model
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Training model with {num_params} parameters")
    if training_config.evaluate_first:
        # Run an initial evaluation
        eval_res = evaluate_model(training_state, training_config, model, dataloaders)
        wandb.log(eval_res, step=training_state.step)
    for epoch in range(training_state.epoch, training_config.epochs):
        # Log the epoch
        training_state.epoch = epoch
        wandb.log({"epoch": epoch}, step=training_state.step)
        save_checkpoint(training_state, training_config, model, optimizer)

        epoch_length = min(training_config.epoch_length, len(dataloaders["train"]))
        progress_bar = tqdm(range(epoch_length), desc=f"Epoch {epoch}")
        iterator = iter(dataloaders["train"])
        for batch_index in progress_bar:
            batch = next(iterator)
            optimizer.zero_grad()
            anchors = batch["anchor_data"]
            positives = batch["positive_data"].reshape(anchors.shape)
            negatives = batch["negative_data"].reshape(anchors.shape)
            all_samples = torch.cat([anchors, positives, negatives], dim=0)
            # Convert all samples to float
            all_samples = all_samples.float()
            # Move the samples to the device
            all_samples = all_samples.to(training_config.device)
            embeddings = model(all_samples)
            anchor_embeddings = embeddings[:anchors.shape[0]]
            positive_embeddings = embeddings[anchors.shape[0]:anchors.shape[0] + positives.shape[0]]
            negative_embeddings = embeddings[anchors.shape[0] + positives.shape[0]:]
            loss = triple_loss(anchor_embeddings, positive_embeddings, negative_embeddings)
            loss.backward()
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
    batch_size = 64
    n_channels = 64
    window_size_s = 3
    sample_freq_hz = 120
    samples_per_window = window_size_s * sample_freq_hz

    model_config = ChambonConfig(C=64, T=samples_per_window, k=63, m=16)
    dataset_config = PhysionetMIDatasetConfig(
        subject_ids=list(range(1, 109+1)),
        window_size_s=window_size_s,
        window_stride_s=1,
        train_prop=0.90,
        extrap_val_prop=0.5,
        extrap_test_prop=0.0,
        intra_val_prop=0.05,
        intra_test_prop=0.00,
        n_positive_train=1,
        m_negative_train=1,
        seed=0
    )
    dataloader_config = DataloaderConfig(
        batch_size_train=batch_size,
        batch_size_eval=32,
        shuffle_train=True,
        shuffle_eval=False,
        num_workers_train=0,
        num_workers_eval=0
    )
    preprocessor_config = MetaPreprocessorConfig(
        sample_rate=160,
        target_sample_rate=sample_freq_hz,
        use_baseline_correction=True,
        use_robust_scaler=True,
        use_clamping=True
    )
    training_config = TrainingConfig(
        device="mps",
        checkpoint_path=Path("./checkpoints/overfit_chambon"),
        evaluation_visualization_path=Path("./visualizations/overfit_chambon"),
        load_checkpoint=True,
        checkpoint_load_path=Path("/Users/aidan/projects/engsci/year4/thesis/implementation/thesis/testing/checkpoints/overfit_chambon/checkpoint_34.pth"),
        evaluate_first=True,
        epochs=50,
        epoch_length=512,
        evaluation_k=5
    )

    all_configs = {
        "model_config": model_config.model_dump(),
        "dataset_config": dataset_config.model_dump(),
        "dataloader_config": dataloader_config.model_dump(),
        "preprocessor_config": preprocessor_config.model_dump(),
        "training_config": training_config.model_dump()
    }
    training_state = TrainingState()
    training_state.config = all_configs

    wandb.init(project="thesis", config=training_state.config)

    model, optimizer = create_model(training_config, model_config)
    if training_config.load_checkpoint:
        load_checkpoint(training_state, training_config, model, optimizer)
    preprocess_fn = preprocessor_factory(preprocessor_config)
    datasets = get_physionet_datasets(dataset_config, preprocess_fn)  # TODO: This preprocessor system doesn't work with multiple workers because it needs to pickle the dataloader
    dataloaders = get_physionet_dataloaders(datasets, dataloader_config)

    train(training_state, training_config, model, optimizer, dataloaders)

    # print(model)
    # test_input = torch.randn(32, model_config.C, model_config.T)
    # output = model(test_input)
    # print(f"Input shape: {test_input.shape}")
    # print(f"Output shape: {output.shape}")