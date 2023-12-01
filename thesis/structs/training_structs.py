

from pydantic import BaseModel, Field, validator
from pathlib import Path

class TrainingConfig(BaseModel):
    device: str = Field(..., description="The device to use for training")
    evaluate_first: bool = Field(False, description="Whether to evaluate the model before training")

    # Checkpointing
    checkpoint_path: Path = Field(..., description="The path to save the checkpoints to")
    upload_wandb_checkpoint: bool = Field(False, description="Whether to update the wandb checkpoint")
    load_checkpoint: bool = Field(False, description="Whether to load a checkpoint")
    checkpoint_load_path: Path | None = Field(None, description="The path to load the checkpoint from")

    # Training
    epochs: int = Field(10, description="The number of epochs to train for")
    epoch_length: int = Field(100, description="The number of batches to train for each epoch")

    # Evaluation
    evaluation_k: int = Field(10, description="The number of samples to use when computing the precision@k and recall@k")
    evaluation_visualization_path: Path = Field(..., description="The path to save the evaluation visualizations to")
    run_extrap_val: bool = Field(True, description="Whether to run the evaluation on the extrapolation validation set")
    run_intra_val: bool = Field(True, description="Whether to run the evaluation on the intra validation set")

    # Downstream evaluation
    run_downstream_eval: bool = Field(False, description="Whether to run the evaluation on the downstream evaluation set")
    downstream_lda_metadata_keys: list[str] = Field([], description="The metadata keys to use for the LDA evaluation. \"unique_subject_id\" and \"dataset_key\" are always available. Other metadata is loaded from .json subject files.")
    downstream_num_folds: int = Field(10, description="The number of folds to use for the downstream evaluation")

    # Loss function
    loss_temperature: float = Field(0.1, description="The temperature to use for the loss function")
    same_session_suppression: float = Field(0.0, description="The amount to suppress the same session pairs in the loss function")

class TrainingState(BaseModel):
    epoch: int = Field(0, description="The current epoch")
    step: int = Field(0, description="The current wandb step")

    config: dict = Field({}, description="The config used for training")
    run_id: str = Field("", description="The id of the run")