"""

"""

from enum import Enum
from pydantic import BaseModel, Field, validator

class HeadStyle(Enum):
    """
    Enum for the different types of heads that can be used
    """
    LINEAR = 1
    MLP = 2
    SIMCLR = 3
    MOCO = 4

class ContrastiveHeadConfig(BaseModel):
    logit_dimension: int = Field(..., description="The dimension of the input to the contrastive head")
    c_loss_dimension: int = Field(..., description="The dimension of the output of the contrastive head")
    head_style: HeadStyle = Field(..., description="The style of the contrastive head")
    layer_sizes: list[int] = Field(..., description="The sizes of the layers in the contrastive head")
    normalize: bool = Field(False, description="Whether to normalize the output of the contrastive head")

    @validator("layer_sizes")
    def check_layer_sizes(cls, v, values):
        assert v[-1] == values["c_loss_dimension"], "Last layer size must be equal to the c_loss_dimension"
        return v