from thesis.structs.preprocessor_structs import *
from .meta import preprocessor_factory as meta_preprocessor_factory
from .in_house import InHousePreprocessor

def construct_preprocess_fn(config: PreprocessorConfig):
    """
    Creates the correct preprocessing function for the given config
    """
    if type(config) == MetaPreprocessorConfig:
        return meta_preprocessor_factory(config)
    elif type(config) == InHousePreprocessorConfig:
        preprocessor = InHousePreprocessor(config)
        return preprocessor.process_sample
    else:
        raise Exception(f"Unknown preprocessor config type {type(config)}")