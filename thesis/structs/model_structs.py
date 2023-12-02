"""

"""

from pydantic import BaseModel, Field, validator
from typing import Literal

class ChambonConfig(BaseModel):
    type: Literal["base_chambon"]
    C: int | None = Field(None, description="The number of channels in the input. If None, then the number of channels will be inferred from the input")
    T: int | None = Field(None, description="The number of timesteps in the input. If None, then the number of timesteps will be inferred from the input")
    k: int = Field(63, description="The kernel size for the temporal convolutions")
    m: int = Field(16, description="The width & stride for the max pooling")

class ChambonExtendableConfig(BaseModel):
    type: Literal["extendable_chambon"]
    C: int = Field(..., description="The number of channels in the input")
    T: int = Field(..., description="The number of timesteps in the input")
    k: int = Field(63, description="The kernel size for the temporal convolutions")
    m: int = Field(16, description="The width & stride for the max pooling")
    D: int = Field(100, description="The dimension of the embedding")
    num_blocks: int = Field(1, description="The number of layers to use in the model")

ModelConfig = ChambonConfig | ChambonExtendableConfig  # Union of all model configs