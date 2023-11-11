"""
A dataset that implements the AbstractContrastiveDataset must provide methods for choosing anchors, positive samples, and negative samples
of arbitrary length. This provides flexibility in the contrastive framework used downstream.
"""

from abc import ABC, abstractmethod
from typing import List, Tuple, Any, Callable
import numpy as np

class AbstractContrastiveDataset(ABC):
    def __init__(self, n_positive: int = 1, m_negative: int = 1, preprocess_fn: Callable = None, transform: Callable = None, *args, **kwargs):
        """
        Parameters:
            n_positive: The number of positive examples to sample for each anchor
            m_negative: The number of negative examples to sample for each anchor
            preprocess_fn: A function that takes in a data point and returns a preprocessed version of it
            transform: A function that takes in a data point and returns a transformed version of it

        The intention of the preprocess function is to apply things like normalization, filtering, etc. that are standard in EEG processing.
        The intention of the transform function is to apply things like dropout, augmentation, etc.
        """
        self.n_positive = n_positive
        self.m_negative = m_negative
        self.preprocess_fn = preprocess_fn
        self.transform = transform

        self.is_anchor_only = n_positive == 0 and m_negative == 0

    @abstractmethod
    def get_anchor(self, index: int) -> Tuple[Any, Any]:
        """
        Returns an anchor data point and its label based on the given index.
        """
        pass

    @abstractmethod
    def get_positive_samples(self, anchor_label: Any, num_samples: int) -> Tuple[List[Any], List[Any]]:
        """
        Returns a list containing 'num_samples' positive examples corresponding to the anchor label along with their labels.
        Which in the positive case are the same as the anchor label, but we do this for consistency.
        """
        pass

    @abstractmethod
    def get_negative_samples(self, anchor_label: Any, num_samples: int) -> Tuple[List[Any], List[Any]]:
        """
        Returns a list containing 'num_samples' negative examples that do not match the anchor label along with their labels.
        """
        pass

    def __getitem__(self, index: int) -> dict:
        anchor_data, anchor_label = self.get_anchor(index)

        attempt_preprocess = lambda data: self.preprocess_fn(data) if self.preprocess_fn is not None else data
        attempt_transform = lambda data: self.transform(data) if self.transform is not None else data
        process_data = lambda data: attempt_transform(attempt_preprocess(data))
        
        if self.is_anchor_only:
            return {
                'anchor_data': process_data(anchor_data),
                'anchor_label': anchor_label
            }

        positive_data, positive_labels = self.get_positive_samples(anchor_label, self.n_positive)
        negative_data, negative_labels = self.get_negative_samples(anchor_label, self.m_negative)
        
        return {
            'anchor_data': np.array(process_data(anchor_data)),
            'anchor_label': anchor_label,
            'positive_data': np.array([process_data(data) for data in positive_data]),
            'positive_labels': positive_labels,
            'negative_data': np.array([process_data(data) for data in negative_data]),
            'negative_labels': negative_labels
        }