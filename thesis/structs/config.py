"""
The full config for a training run
"""

from pydantic import BaseModel, Field, validator

from .contrastive_head_structs import ContrastiveHeadConfig

class OverfitChambonConfig(BaseModel):
    contrastive_head_config: ContrastiveHeadConfig = Field(..., description="The config for the contrastive head")