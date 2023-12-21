"""

"""

from pydantic import BaseModel, Field, validator
from typing import Literal

class ChambonConfig(BaseModel):
    type: Literal["base_chambon"] = "base_chambon"
    C: int | None = Field(None, description="The number of channels in the input. If None, then the number of channels will be inferred from the input")
    C_prime: int | None = Field(None, description="The number of virtual channels to use in the first convolution. If None, then C_prime will be set to C")
    T: int | None = Field(None, description="The number of timesteps in the input. If None, then the number of timesteps will be inferred from the input")
    k: int = Field(63, description="The kernel size for the temporal convolutions")
    m: int = Field(16, description="The width & stride for the max pooling")

class ChambonExtendableConfig(BaseModel):
    type: Literal["extendable_chambon"] = "extendable_chambon"
    C: int = Field(..., description="The number of channels in the input")
    T: int = Field(..., description="The number of timesteps in the input")
    k: int = Field(63, description="The kernel size for the temporal convolutions")
    m: int = Field(16, description="The width & stride for the max pooling")
    D: int = Field(100, description="The dimension of the embedding")
    num_blocks: int = Field(1, description="The number of layers to use in the model")

class ChambonWithLinearConfig(BaseModel):
    type: Literal["chambon_with_linear"] = "chambon_with_linear"
    C: int | None = Field(None, description="The number of channels in the input. If None, then the number of channels will be inferred from the input")
    C_prime: int | None = Field(None, description="The number of virtual channels to use in the first convolution. If None, then C_prime will be set to C")
    T: int | None = Field(None, description="The number of timesteps in the input. If None, then the number of timesteps will be inferred from the input")
    k: int = Field(63, description="The kernel size for the temporal convolutions")
    m: int = Field(16, description="The width & stride for the max pooling")
    linear_sizes: list[int] = Field([], description="The sizes of the linear layers to be used at the end of the model")

# Union of all model configs
ModelConfig = ChambonConfig | ChambonExtendableConfig | ChambonWithLinearConfig