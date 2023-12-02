from pathlib import Path
import time

import torch

from thesis.structs import Config, TrainingState
from thesis.utils import get_git_hash

from .chambon import ChambonConfig, ChambonNet
from .chambon_extendable import ChambonExtendableConfig, ChambonNetExtendable
from .contrastive_head import ContrastiveHeadConfig, ContrastiveHead, ContrastiveModel

def construct_model(model_config, head_config, device='cpu', lr=1e-3):
    """
    Constructs a model and contrastive head given the model and head configs
    """
    if type(model_config) == ChambonConfig:
        model = ChambonNet(model_config)
    elif type(model_config) == ChambonExtendableConfig:
        model = ChambonNetExtendable(model_config)
    else:
        raise ValueError(f"Model config type {type(model_config)} not supported")

    test_data = torch.randn(1, model_config.C, model_config.T)
    test_output = model(test_data)
    embedding_size = test_output.shape[1]
    print("Logit Size", embedding_size)

    if head_config.logit_dimension != embedding_size:
        if head_config.logit_dimension is not None:
            print(f"Warning: Logit dimension {head_config.logit_dimension} does not match embedding size {embedding_size}. Setting logit dimension to embedding size")
        head_config.logit_dimension = embedding_size

    full_model = ContrastiveModel(model, head_config)

    full_model = full_model.to(device)
    optimizer = torch.optim.Adam(full_model.parameters(), lr=lr)

    return full_model, optimizer

def save_checkpoint(checkpoint_path: Path, model: torch.nn.Module, optimizer: torch.optim.Optimizer, config: Config, training_state: TrainingState):
    """
    Saves a full checkpoint including the model and the optimizer state as well as a serialized config and training state
    """
    checkpoint_dict = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "config": config.model_dump(),
        "training_state": training_state.model_dump()
    }
    torch.save(checkpoint_dict, checkpoint_path)

    
def load_checkpoint(checkpoint_path: Path, current_config: Config | None) -> tuple[torch.nn.Module, torch.optim.Optimizer, Config, TrainingState]:
    """
    Loads a checkpoint from the path, constructs the model based on the saved config, and returns the model and optimizer

    Prints a warning if the current git hash does not match the one saved in the checkpoint
    """
    checkpoint_dict = torch.load(checkpoint_path)
    loaded_config: Config = Config.model_validate(checkpoint_dict["config"])
    loaded_training_state: TrainingState = TrainingState.model_validate(checkpoint_dict["training_state"])

    if current_config is None:
        config = loaded_config
    else:
        config = current_config

    model, optimizer = construct_model(config.embedding_model_config, config.head_config, device=config.training_config.device, lr=config.training_config.lr)
    model.load_state_dict(checkpoint_dict["model_state_dict"])
    optimizer.load_state_dict(checkpoint_dict["optimizer_state_dict"])

    found_issue = False
    if loaded_config.git_hash != get_git_hash():
        print("WARNING: The git hash of the checkpoint does not match the current git hash. This may cause issues.")
        print(f"Checkpoint Git Hash: {loaded_config.git_hash}")
        print(f"Current Git Hash: {get_git_hash()}\n\n")
        found_issue = True

    if loaded_config.model_config != config.model_config:
        print("WARNING: The model config of the checkpoint does not match the current model config. This will almost certainly cause issues.")
        print(f"Checkpoint Model Config: {loaded_config.model_config}")
        print(f"Current Model Config: {config.model_config}\n\n")
        inp = input("Do you want to continue? (y/n) ")
        if inp != "y":
            raise ValueError("User chose to cancel")
    
    if loaded_config.head_config != config.head_config:
        print("WARNING: The head config of the checkpoint does not match the current head config. This will almost certainly cause issues.")
        print(f"Checkpoint Head Config: {loaded_config.head_config}")
        print(f"Current Head Config: {config.head_config}\n\n")
        inp = input("Do you want to continue? (y/n) ")
        if inp != "y":
            raise ValueError("User chose to cancel")

    if loaded_config.model_dump() != config.model_dump():
        print("WARNING: The model config of the checkpoint does not match the current model config. This may cause issues.")
        print(f"Checkpoint Model Config: {loaded_config.model_config}")
        print(f"Current Model Config: {config.model_config}\n\n")
        found_issue = True

    if found_issue:
        print("Pausing for 5 seconds to allow you to cancel...")
        time.sleep(5)

    return model, optimizer, loaded_config, loaded_training_state