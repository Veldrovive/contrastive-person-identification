"""

"""

from thesis.structs.training_structs import TrainingConfig

import torch
import numpy as np
from tqdm import tqdm

from sklearn.neighbors import NearestNeighbors
import umap
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

def get_embeddings(training_config: TrainingConfig, model, dataloader, limit=None) -> tuple[list[torch.Tensor], list[str], list[str]]:
    """
    Gets the embedding (unique_subject_id, session) pairs for every anchor in the dataloader
    if limit is not None, only the first limit pairs are returned.
    """
    # If the limit is set, then we need to create a new dataloader with the same dataset that takes a consistent subset of the data
    torch.manual_seed(0)

    if limit is not None:
        dataset = dataloader.dataset
        subset_indices = torch.randperm(len(dataset))[:limit]
        dataset = torch.utils.data.Subset(dataset, subset_indices)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=dataloader.batch_size, shuffle=False, num_workers=dataloader.num_workers)
    
    embeddings = []
    labels = []
    sessions = []
    with torch.no_grad():
        for batch in tqdm(dataloader):
            anchors = batch["anchor"]
            metadata = batch["anchor_metadata"]

            anchors = anchors.float().to(training_config.device)
            anchor_embeddings = model(anchors)
            embeddings.extend(anchor_embeddings.cpu().numpy())

            batch_labels = [data['unique_subject_id'] for data in metadata]
            labels.extend(batch_labels)

            batch_sessions = [data['dataset_key'][1] for data in metadata]
            sessions.extend(batch_sessions)

    return embeddings, labels, sessions

def get_representative_embedding(embeddings: np.ndarray, z_score_threshold: float = 1.5):
    """
    Computes a more reliable embedding out of a set of samples from the distribution of embeddings for a single subject
    
    We take an array of shape (n, m) where there are n samples of m-dimensional embeddings
    We compute the cosine similarity between each embedding and take the average
    Embeddings with a Z-score above the threshold are considered outliers and are removed
    The remaining embeddings are averaged to get the representative embedding

    This has the flaw that if the distribution of embeddings is not unimodal we could get a bad representative embedding
    We could fix this by using a more advanced clustering technique to detect outliers, but for a first test this should be fine
    """
    # Normalize the embeddings
    embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)

    # Calculate pairwise cosine similarity since embeddings are normalized
    cosine_similarity = embeddings @ embeddings.T
    avg_similarity = np.mean(cosine_similarity, axis=1)

    z_scores = (avg_similarity - np.mean(avg_similarity)) / np.std(avg_similarity)
    outliers = np.abs(z_scores) > z_score_threshold

    # Filter out the outliers
    filtered_embeddings = embeddings[~outliers]

    # Compute the average of the remaining embeddings
    representative_embedding = np.mean(filtered_embeddings, axis=0)

    # Normalize the representative embedding
    representative_embedding = representative_embedding / np.linalg.norm(representative_embedding)

    return representative_embedding

def select_random_subsets(embeddings, subset_size):
    """
    Select random subsets of embeddings with no overlap, adjusting for divisibility.
    There is no overlap between subsets.
    """
    n, m = embeddings.shape
    
    # Adjust n to make it divisible by subset_size
    n_adj = n - (n % subset_size)
    adjusted_embeddings = embeddings[:n_adj]

    # Shuffle the adjusted embeddings
    np.random.shuffle(adjusted_embeddings)

    # Split the array into subsets
    num_subsets = n_adj // subset_size
    subsets = np.array(np.array_split(adjusted_embeddings, num_subsets))

    return subsets

def get_cross_session_knn_metrics(embeddings: list[np.ndarray], labels: list[str], sessions: list[str], k: int = 2, metric: str = 'cosine'):
    """
    Computes the knn metrics for the given embedding label pairs, but exclude neighbors from the same session
    """
    session_embeddings = {}  # Maps from (label, session) to a list of embeddings
    for embedding, label, session in zip(embeddings, labels, sessions):
        key = (label, session)
        if key not in session_embeddings:
            session_embeddings[key] = []
        session_embeddings[key].append(embedding)

    # As a sanity check, we ensure that every subject has at least 2 sessions. Otherwise we need to issue
    # a warning as the results will be meaningless
    label_session_counts = {}
    for label, session in session_embeddings.keys():
        if label not in label_session_counts:
            label_session_counts[label] = 0
        label_session_counts[label] += 1
    num_subjects_with_one_session = sum([1 for count in label_session_counts.values() if count == 1])
    if num_subjects_with_one_session > 0:
        print(f"WARNING: {num_subjects_with_one_session} subjects have only one session. The results will be meaningless. This evaluation should not enforce cross-session sampling.")

    # Now we create a nearest neighbors model for each session
    session_models = {}
    session_predictions = {}  # Maps from (label, session) to the output of the model run over all embeddings
    for (label, session), subject_session_embeddings in session_embeddings.items():
        session_models[(label, session)] = NearestNeighbors(n_neighbors=k, metric=metric).fit(subject_session_embeddings)
        session_predictions[(label, session)] = session_models[(label, session)].kneighbors(embeddings, return_distance=True)

    # Predict the labels for each embedding over all session models
    label_predictions = []  # Each element is a list of tuples (distance, label, session) for the corresponding embedding
    for embedding_index, (embedding, label, session) in enumerate(zip(embeddings, labels, sessions)):
        embedding_predictions = []
        for session_key, subject_session_predictions in session_predictions.items():
            if session_key == (label, session):
                continue
            # Otherwise we add the predictions from this session to the list
            for distance, neighbor_index in zip(subject_session_predictions[0][embedding_index], subject_session_predictions[1][embedding_index]):
                neighbor_label = session_key[0]
                neighbor_session = session_key[1]
                embedding_predictions.append((distance, neighbor_label, neighbor_session))
        label_predictions.append(embedding_predictions)

    # Now for each embedding, we sort the predictions by distance and take the top k
    label_predictions = [sorted(predictions, key=lambda x: x[0])[:k] for predictions in label_predictions]
    # NOTE: We don't have any problems with the first neighbor being the same as the anchor because we exclude neighbors from the same session

    # We also compute metrics for representative embeddings. For now we choose subset sizes of 10.
    improved_embeddings = []
    improved_labels_gt = []
    improved_sessions_gt = []
    for (label, session), subject_session_embeddings in session_embeddings.items():
        # Split the session embeddings into subsets
        subsets = select_random_subsets(np.array(subject_session_embeddings), 10)
        
        improved_embeddings.extend([get_representative_embedding(subset) for subset in subsets])
        improved_labels_gt.extend([label for _ in subsets])
        improved_sessions_gt.extend([session for _ in subsets])

    # We still use the original models to predict the neighbors
    improved_session_predictions = {}  # Maps from (label, session) to the output of the model run over all embeddings
    for (label, session), subject_session_embeddings in session_embeddings.items():
        improved_session_predictions[(label, session)] = session_models[(label, session)].kneighbors(improved_embeddings, return_distance=True)

    # Predict the labels for each embedding over all session models
    improved_label_predictions = []  # Each element is a list of tuples (distance, label, session) for the corresponding embedding
    for embedding_index, (embedding, label, session) in enumerate(zip(improved_embeddings, improved_labels_gt, improved_sessions_gt)):
        embedding_predictions = []
        for session_key, subject_session_predictions in improved_session_predictions.items():
            if session_key == (label, session):
                continue
            # Otherwise we add the predictions from this session to the list
            for distance, neighbor_index in zip(subject_session_predictions[0][embedding_index], subject_session_predictions[1][embedding_index]):
                neighbor_label = session_key[0]
                neighbor_session = session_key[1]
                embedding_predictions.append((distance, neighbor_label, neighbor_session))
        improved_label_predictions.append(embedding_predictions)

    # Now for each embedding, we sort the predictions by distance and take the top k
    improved_label_predictions = [sorted(predictions, key=lambda x: x[0])[:k] for predictions in improved_label_predictions]

    # Compute metrics
    top_1_count_map = {}  # Maps from label to the number of times the top 1 neighbor is from the same subject
    top_k_count_map = {}  # Maps from label to the number of times the correct label appears in the top k neighbors
    total_count_map = {}  # Maps from label to the total number of times the label was seen

    top_1_labels = []  # The same order as embeddings. The predicted label for each embedding
    consensus_k_labels = []  # The same order as embeddings. The predicted label for each embedding given by the consensus of the top k neighbors

    for embedding, true_label, true_session, label_prediction in zip(embeddings, labels, sessions, label_predictions):
        # Get the top 1 label
        top_1_label = label_prediction[0][1]
        top_1_labels.append(top_1_label)
        if true_label not in top_1_count_map:
            top_1_count_map[true_label] = 0
        if top_1_label == true_label:
            top_1_count_map[true_label] += 1

        # Get the top k label
        top_k_labels = [label for _, label, _ in label_prediction]
        consensus_k_label = max(set(top_k_labels), key=top_k_labels.count)
        consensus_k_labels.append(consensus_k_label)
        
        if true_label not in top_k_count_map:
            top_k_count_map[true_label] = 0
        if true_label in top_k_labels:
            top_k_count_map[true_label] += 1

        # Update the total count map
        if true_label not in total_count_map:
            total_count_map[true_label] = 0
        total_count_map[true_label] += 1

    # Compute the accuracy for each label
    top_1_accuracies = {}
    top_k_accuracies = {}
    for label in total_count_map.keys():
        top_1_accuracies[label] = top_1_count_map[label] / total_count_map[label]
        top_k_accuracies[label] = top_k_count_map[label] / total_count_map[label]

    # Compute the total accuracy
    total_top_1_accuracy = sum(top_1_count_map.values()) / sum(total_count_map.values())
    total_top_k_accuracy = sum(top_k_count_map.values()) / sum(total_count_map.values())

    
    # And then again for the improved set
    improved_top_1_count_map = {}  # Maps from label to the number of times the top 1 neighbor is from the same subject
    improved_top_k_count_map = {}  # Maps from label to the number of times the correct label appears in the top k neighbors
    improved_total_count_map = {}  # Maps from label to the total number of times the label was seen

    improved_top_1_labels = []  # The same order as embeddings. The predicted label for each embedding
    improved_consensus_k_labels = []  # The same order as embeddings. The predicted label for each embedding given by the consensus of the top k neighbors

    for embedding, true_label, true_session, label_prediction in zip(improved_embeddings, improved_labels_gt, improved_sessions_gt, improved_label_predictions):
        # Get the top 1 label
        top_1_label = label_prediction[0][1]
        improved_top_1_labels.append(top_1_label)
        if true_label not in improved_top_1_count_map:
            improved_top_1_count_map[true_label] = 0
        if top_1_label == true_label:
            improved_top_1_count_map[true_label] += 1

        # Get the top k label
        top_k_labels = [label for _, label, _ in label_prediction]
        consensus_k_label = max(set(top_k_labels), key=top_k_labels.count)
        improved_consensus_k_labels.append(consensus_k_label)
        
        if true_label not in improved_top_k_count_map:
            improved_top_k_count_map[true_label] = 0
        if true_label in top_k_labels:
            improved_top_k_count_map[true_label] += 1

        # Update the total count map
        if true_label not in improved_total_count_map:
            improved_total_count_map[true_label] = 0
        improved_total_count_map[true_label] += 1

    # Compute the accuracy for each label
    improved_top_1_accuracies = {}
    improved_top_k_accuracies = {}
    for label in improved_total_count_map.keys():
        improved_top_1_accuracies[label] = improved_top_1_count_map[label] / improved_total_count_map[label]
        improved_top_k_accuracies[label] = improved_top_k_count_map[label] / improved_total_count_map[label]

    # Compute the total accuracy
    improved_total_top_1_accuracy = sum(improved_top_1_count_map.values()) / sum(improved_total_count_map.values())
    improved_total_top_k_accuracy = sum(improved_top_k_count_map.values()) / sum(improved_total_count_map.values())

    res = {
        # Used for visualization of embeddings
        'session_embeddings': session_embeddings,

        # Finalized metrics
        'top_1_accuracies': top_1_accuracies,
        'top_k_accuracies': top_k_accuracies,
        'total_top_1_accuracy': total_top_1_accuracy,
        'total_top_k_accuracy': total_top_k_accuracy,

        # Improved metrics
        'improved_top_1_accuracies': improved_top_1_accuracies,
        'improved_top_k_accuracies': improved_top_k_accuracies,
        'improved_total_top_1_accuracy': improved_total_top_1_accuracy,
        'improved_total_top_k_accuracy': improved_total_top_k_accuracy,

        # Used for confusion matrix
        'true_labels': labels,
        'top_1_labels': top_1_labels,
        'consensus_k_labels': consensus_k_labels
    }

    return res

def visualize_embeddings(session_embeddings: dict[tuple[str, str], list[np.ndarray]], title: str = "Embeddings", separate_sessions: bool = False):
    """
    Uses UMAP to visualize the embeddings in 2D.
    If separate_sessions is True, then the label is (label, session) instead of just label

    Returns a matplotlib figure
    """
    # Convert the embeddings into a numpy array and a parallel list of labels
    embeddings = []
    labels = []
    for (label, session), session_embeddings in session_embeddings.items():
        embeddings.extend(session_embeddings)
        if separate_sessions:
            labels.extend([(label, session) for _ in session_embeddings])
        else:
            labels.extend([label for _ in session_embeddings])
    embeddings = np.array(embeddings)

    # Reduce dimensionality using UMAP
    reducer = umap.UMAP()
    embeddings = reducer.fit_transform(embeddings)

    # Get numerical values for the labels
    unique_labels = sorted(list(set(labels)))
    label_map = {label: i for i, label in enumerate(unique_labels)}
    numerical_labels = [label_map[label] for label in labels]

    labeled_embeddings = {}
    for embedding, label in zip(embeddings, numerical_labels):
        if label not in labeled_embeddings:
            labeled_embeddings[label] = []
        labeled_embeddings[label].append(embedding)
    labeled_embeddings = {label: np.array(embeddings) for label, embeddings in labeled_embeddings.items()}

    # Defining size. If there are more than 1000 embeddings, we use size 3. More than 10000 and we use size 1
    size = 5
    if len(embeddings) > 10000:
        size = 1
    elif len(embeddings) > 1000:
        size = 3

    # Plot the embeddings
    fig, ax = plt.subplots(1, 1)
    for label, embeddings in labeled_embeddings.items():
        ax.scatter(
            embeddings[:, 0],
            embeddings[:, 1],
            label=label,
            s=size
        )
    ax.set_title(title)

    include_legend = len(unique_labels) < 10
    if include_legend:
        ax.legend(unique_labels)

    return fig

def visualize_confusion_matrix(true_labels: list[str], predicted_labels: list[str], pred_label: str = "Predicted Label", true_label: str = "True Label", title: str = "Confusion Matrix"):
    """
    Plots a confusion matrix for the given labels
    
    Returns a matplotlib figure
    """
    # Get numerical values for the labels
    unique_labels = sorted(list(set(true_labels)))
    label_map = {label: i for i, label in enumerate(unique_labels)}
    numerical_true_labels = [label_map[label] for label in true_labels]
    numerical_predicted_labels = [label_map[label] for label in predicted_labels]

    # Compute the confusion matrix
    confusion = confusion_matrix(numerical_true_labels, numerical_predicted_labels)
    # Normalize the confusion matrix
    confusion = confusion.astype('float') / confusion.sum(axis=1)[:, np.newaxis]

    # Configuration
    num_labels = len(unique_labels)
    use_annotations = num_labels < 10

    x_labels = unique_labels if num_labels < 20 else False
    y_labels = unique_labels if num_labels < 20 else False

    # Plot the confusion matrix
    fig, ax = plt.subplots(1, 1)
    sns.heatmap(confusion, annot=use_annotations, ax=ax, cmap='Blues', xticklabels=x_labels, yticklabels=y_labels)
    ax.set_xlabel("Predicted Label")
    ax.set_ylabel("True Label")
    ax.set_title(title)

    return fig