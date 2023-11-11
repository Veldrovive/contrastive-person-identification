"""

"""

from pydantic import BaseModel, Field, validator

class ChambonConfig(BaseModel):
    C: int = Field(..., description="The number of channels in the input")
    T: int = Field(..., description="The number of timesteps in the input")
    k: int = Field(63, description="The kernel size for the temporal convolutions")
    m: int = Field(16, description="The width & stride for the max pooling")
    D: int = Field(100, description="The dimension of the embedding")

class ChambonExtendableConfig(BaseModel):
    C: int = Field(..., description="The number of channels in the input")
    T: int = Field(..., description="The number of timesteps in the input")
    k: int = Field(63, description="The kernel size for the temporal convolutions")
    m: int = Field(16, description="The width & stride for the max pooling")
    D: int = Field(100, description="The dimension of the embedding")
    num_blocks: int = Field(1, description="The number of layers to use in the model")