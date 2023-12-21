"""
Implements required methods for computing an evaluation of the models based on a set of downstream tasks
"""

import torch
import numpy as np
from tqdm import tqdm
import random
from sklearn.metrics import precision_recall_fscore_support
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

from thesis.structs.training_structs import TrainingConfig
from thesis.models.contrastive_head import ContrastiveHead


def get_embeddings(training_config: TrainingConfig, logit_model, dataloader):
    """
    Computes embeddings and matching metadata from a dataset
    """
    if type(logit_model) == ContrastiveHead:
        raise ValueError("Downstream tasks should use the embedding module, not the contrastive head")
    # If the limit is set, then we need to create a new dataloader with the same dataset that takes a consistent subset of the data
    torch.manual_seed(0)

    embeddings = []
    metadatas = []
    with torch.no_grad():
        for batch in tqdm(dataloader):
            anchors = batch["anchor"]
            metadata = batch["anchor_metadata"]

            # Move to device
            anchors = anchors.float().to(training_config.device)

            # Compute the embeddings
            embeddings.append(logit_model(anchors).cpu().numpy())

            metadatas.extend(metadata)

    embeddings = np.concatenate(embeddings)
    assert embeddings.shape[0] == len(metadatas), f"Metadata does not match with embeddings: {embeddings.shape[0]} != {len(metadata)}"

    return embeddings, metadatas

def compute_LDA_metrics(embeddings: np.ndarray, metadatas: list[dict], metadata_key: str, num_folds: int = 5, bayes_n: int = 5, separate_subjects: bool = False):
    """
    Computes the precision, recall, and f1 score using LDA on the embeddings

    num_folds: The number of folds to use for cross validation
    separate_subjects: If true, the training and test sets will be split by the subject. If false, they will be split by the sample.
    """
    # Generate the folds
    if separate_subjects:
        # Then we need to decide the subjects that go in each fold
        unique_subjects = np.unique([metadata["unique_subject_id"] for metadata in metadatas])
        np.random.shuffle(unique_subjects)
        folds_subjects = np.array_split(unique_subjects, num_folds)
        subject_to_fold_index_map = {subject: fold_index for fold_index, fold in enumerate(folds_subjects) for subject in fold}
        # For each fold, we then need to generate 2 arrays for the embeddings, metadata, and subject
        folds: list[tuple[list[np.ndarray], list, list[str]]] = [(list(), list(), list()) for _ in range(num_folds)]
        for embedding, metadata in zip(embeddings, metadatas):
            # Find the fold that the subject is in
            subject = metadata["unique_subject_id"]
            fold_index = subject_to_fold_index_map[subject]
            folds[fold_index][0].append(embedding)
            folds[fold_index][1].append(metadata[metadata_key])
            folds[fold_index][2].append(subject)
    else:
        # Then we just shuffle the data and split it into folds
        indices = np.arange(len(embeddings))
        np.random.shuffle(indices)
        folds: list[tuple[list[np.ndarray], list, list[str]]] = [(list(), list(), list()) for _ in range(num_folds)]
        for index, (embedding, metadata) in enumerate(zip(embeddings, metadatas)):
            # Find the fold that the sample is in by evenly distributing the samples
            fold_index = index % folds
            folds[fold_index][0].append(embedding)
            folds[fold_index][1].append(metadata[metadata_key])
            folds[fold_index][2].append(metadata["unique_subject_id"])

    individual_total_precision = 0
    individual_total_recall = 0
    individual_total_f1 = 0
    individual_total_cross_entropy = 0

    consensus_total_precision = 0
    consensus_total_recall = 0
    consensus_total_f1 = 0
    consensus_total_cross_entropy = 0

    subject_consensus_scores = {}
    for test_fold_index in range(num_folds):
        test_set = folds[test_fold_index]
        train_set = (list(), list(), list())
        for train_fold_index in range(num_folds):
            if train_fold_index != test_fold_index:
                train_set[0].extend(folds[train_fold_index][0])
                train_set[1].extend(folds[train_fold_index][1])
                train_set[2].extend(folds[train_fold_index][2])
        
        # Now we train the LDA
        lda = LinearDiscriminantAnalysis()
        lda.fit(train_set[0], train_set[1])

        # And compute the metrics for individual samples
        try:
            predictions = lda.predict(test_set[0])
        except ValueError as e:
            print(f"This error generally occurs when you have more folds than subjects in the downstream dataset.")
            raise e
        precision, recall, f1, _ = precision_recall_fscore_support(test_set[1], predictions, average="binary", zero_division=0)
        individual_total_precision += precision
        individual_total_recall += recall
        individual_total_f1 += f1

        # We also compute the cross entropy between the posterior and the true labels
        # So we are going to construct a one hot vector for the true labels as the target distribution
        target_distribution = np.zeros_like(lda.predict_proba(test_set[0]))
        target_distribution[np.arange(len(test_set[1])), test_set[1]] = 1

        # And then compute the cross entropy
        cross_entropy = -np.sum(target_distribution * np.log(lda.predict_proba(test_set[0])), axis=1)
        # And then take the mean
        mean_cross_entropy = np.mean(cross_entropy)

        individual_total_cross_entropy += mean_cross_entropy

        # However, LDA also outputs probabilities, so for each subject we can use naive bayes to compute the probability of the subject
        # using a larger set of samples defined by the bayes_n parameter
        # First, we split the test set into subjects
        subject_to_samples_map = dict()
        for sample, metadata, subject in zip(test_set[0], test_set[1], test_set[2]):
            if subject not in subject_to_samples_map:
                subject_to_samples_map[subject] = (list(), list(), list())
            subject_to_samples_map[subject][0].append(sample)
            subject_to_samples_map[subject][1].append(metadata)
            subject_to_samples_map[subject][2].append(subject)
        # We shuffle the samples within each subject to reduce overlap effects
        for subject, samples in subject_to_samples_map.items():
            indices = np.arange(len(samples[0]))
            np.random.shuffle(indices)
            subject_to_samples_map[subject] = (np.array(samples[0])[indices], np.array(samples[1])[indices], np.array(samples[2])[indices])
        # Then we compute the probability of each subject
        bayes_posterior_probability_sets = []  # List of numpy arrays of shape (n_samples, n_classes)
        target_distributions_sets = []  # List of numpy arrays of shape (n_samples, n_classes)
        true_labels_sets = []  # List of numpy arrays of shape (n_samples,)
        predicted_labels_sets = []  # List of numpy arrays of shape (n_samples,)
        for subject, samples in subject_to_samples_map.items():
            # We need to compute the probability of each sample
            sample_probabilities = lda.predict_proba(samples[0])
            # Prep for multiplying the likelihood by constructing a n x n_classes x bayes_n matrix of the probabilities
            # where each row is rolled by 1 more than the previous
            n_samples, n_classes = sample_probabilities.shape
            bayes_matrix = np.zeros((n_samples, n_classes, bayes_n))
            for i in range(bayes_n):
                bayes_matrix[:, :, i] = np.roll(sample_probabilities, i, axis=0)
            # We also construct a nx2 prior vector which is uniform for each sample
            prior_vector = np.ones((n_samples, n_classes)) / 2
            # To compute the posterior, we first multiply all the likelihoods together
            likelihood = np.prod(bayes_matrix, axis=2)
            # And then multiply by the prior
            posterior = likelihood * prior_vector
            # And normalize to a distribution
            posterior = posterior / np.sum(posterior, axis=1, keepdims=True)
            bayes_posterior_probability_sets.append(posterior)

            # We want to compute the cross entropy between the posterior and the true labels
            # So we are going to construct a one hot vector for the true labels as the target distribution
            target_distribution = np.zeros_like(posterior)
            target_distribution[np.arange(n_samples), samples[1]] = 1
            target_distributions_sets.append(target_distribution)

            # And then compute the cross entropy
            subject_cross_entropy = -np.sum(target_distribution * np.log(posterior), axis=1)
            # And then take the mean
            subject_mean_cross_entropy = np.mean(cross_entropy)

            # We also extract the predicted labels to compute the precision, recall, and f1
            subject_predictions = np.argmax(posterior, axis=1)
            true_labels_sets.append(samples[1])
            predicted_labels_sets.append(subject_predictions)
            
            subject_precision, subject_recall, subject_f1, _ = precision_recall_fscore_support(samples[1], subject_predictions, average="binary", zero_division=0)

            subject_scores = {
                "precision": subject_precision,
                "recall": subject_recall,
                "f1": subject_f1,
                "cross_entropy": subject_mean_cross_entropy
            }
            subject_id = samples[2][0]
            assert subject_id not in subject_consensus_scores, f"Subject {subject_id} already in subject_consensus_scores"
            subject_consensus_scores[subject_id] = subject_scores
        bayes_posterior_probability = np.concatenate(bayes_posterior_probability_sets)
        target_distributions = np.concatenate(target_distributions_sets)
        true_labels = np.concatenate(true_labels_sets)
        predicted_labels = np.concatenate(predicted_labels_sets)

        fold_cross_entropy = -np.sum(target_distributions * np.log(bayes_posterior_probability), axis=1)
        fold_mean_cross_entropy = np.mean(fold_cross_entropy)

        fold_precision, fold_recall, fold_f1, _ = precision_recall_fscore_support(true_labels, predicted_labels, average="binary")

        consensus_total_precision += fold_precision
        consensus_total_recall += fold_recall
        consensus_total_f1 += fold_f1
        consensus_total_cross_entropy += fold_mean_cross_entropy

        

            

    individual_avg_precision = individual_total_precision / num_folds
    individual_avg_recall = individual_total_recall / num_folds
    individual_avg_f1 = individual_total_f1 / num_folds
    individual_avg_cross_entropy = individual_total_cross_entropy / num_folds

    consensus_avg_precision = consensus_total_precision / num_folds
    consensus_avg_recall = consensus_total_recall / num_folds
    consensus_avg_f1 = consensus_total_f1 / num_folds
    consensus_avg_cross_entropy = consensus_total_cross_entropy / num_folds

    return {
        "individual_avg_precision": individual_avg_precision,
        "individual_avg_recall": individual_avg_recall,
        "individual_avg_f1": individual_avg_f1,
        "individual_avg_cross_entropy": individual_avg_cross_entropy,

        "consensus_avg_precision": consensus_avg_precision,
        "consensus_avg_recall": consensus_avg_recall,
        "consensus_avg_f1": consensus_avg_f1,
        "consensus_avg_cross_entropy": consensus_avg_cross_entropy,

        "subject_consensus_scores": subject_consensus_scores
    }

