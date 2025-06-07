"""Data schemas and type definitions."""

from typing import Optional, Literal, Dict, Any
from datetime import datetime
from pydantic import BaseModel, Field, validator
import numpy as np


class BrandProfile(BaseModel):
    """Schema for brand characteristics."""
    
    brand_id: int
    brand_name: str
    brand_size: Literal["large", "medium", "small"]
    innovation_score: float = Field(..., ge=0, le=1)
    market_segment: Literal["luxury", "mass", "emerging"]
    base_visibility: float = Field(..., ge=0, le=1)
    
    @validator("innovation_score", "base_visibility")
    def validate_probability(cls, v: float) -> float:
        """Ensure values are valid probabilities."""
        return np.clip(v, 0, 1)
    
    class Config:
        """Pydantic configuration."""
        json_encoders = {
            np.float64: lambda v: float(v),
            np.int64: lambda v: int(v),
        }


class ActionSet(BaseModel):
    """Schema for marketing actions."""
    
    wikipedia_update: bool = False
    youtube_content: bool = False
    press_release: bool = False
    
    @property
    def any_action_taken(self) -> bool:
        """Check if any action was taken."""
        return any([self.wikipedia_update, self.youtube_content, self.press_release])
    
    def to_dict(self) -> Dict[str, bool]:
        """Convert to dictionary."""
        return self.dict()


class ObservationRecord(BaseModel):
    """Schema for time-series observations."""
    
    brand_id: int
    brand_name: str
    week: int
    date: datetime
    visibility_score: float = Field(..., ge=0, le=1)
    
    # Actions
    wikipedia_update: bool
    youtube_content: bool
    press_release: bool
    
    # Confounders
    market_trend: float
    competitor_action: bool
    news_event: bool
    
    # Brand characteristics (denormalized for efficiency)
    brand_size: Literal["large", "medium", "small"]
    innovation_score: float = Field(..., ge=0, le=1)
    market_segment: Literal["luxury", "mass", "emerging"]
    
    @validator("visibility_score")
    def validate_visibility(cls, v: float) -> float:
        """Ensure visibility is valid probability."""
        return np.clip(v, 0, 1)
    
    class Config:
        """Pydantic configuration."""
        json_encoders = {
            datetime: lambda v: v.isoformat(),
            np.float64: lambda v: float(v),
            np.int64: lambda v: int(v),
        }


class CausalEffect(BaseModel):
    """Schema for causal effect estimates."""
    
    treatment: str
    effect_size: float
    confidence_interval_lower: float
    confidence_interval_upper: float
    p_value: float
    sample_size: int
    method: str
    
    @property
    def is_significant(self) -> bool:
        """Check if effect is statistically significant."""
        return self.p_value < 0.05
    
    @property
    def confidence_interval(self) -> tuple[float, float]:
        """Get confidence interval as tuple."""
        return (self.confidence_interval_lower, self.confidence_interval_upper)