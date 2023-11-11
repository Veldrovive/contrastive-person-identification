from thesis.structs.augmentation_structs import *
from typing import Callable

from thesis.augmentation.session_variance_augmentation import preprocessor_factory as session_variance_preprocessor_factory

def construct_augmentation_fn(config: AugmentationConfig) -> Callable:
    """
    Creates the correct augmentation function for the given config
    """
    if type(config) == SessionVarianceAugmentationConfig:
        return session_variance_preprocessor_factory(config)
    else:
        raise Exception(f"Unknown preprocessor config type {type(config)}")