"""Application configuration settings."""

from typing import Dict, Any, Optional
from functools import lru_cache
from pydantic import Field

try:
    from pydantic_settings import BaseSettings
except ImportError:
    # Fallback for older pydantic versions
    from pydantic import BaseSettings


class Settings(BaseSettings):
    """Application settings with environment variable support."""
    
    # Application settings
    app_name: str = "Recommendation Impact Simulator"
    app_version: str = "0.1.0"
    debug: bool = Field(False, env="DEBUG")
    
    # Data generation settings
    n_brands: int = Field(50, env="N_BRANDS")
    n_time_periods: int = Field(52, env="N_TIME_PERIODS")
    random_seed: int = Field(42, env="RANDOM_SEED")
    
    # Causal inference settings
    bootstrap_iterations: int = Field(100, env="BOOTSTRAP_ITERATIONS")
    min_samples_leaf: int = Field(10, env="MIN_SAMPLES_LEAF")
    confidence_level: float = Field(0.95, env="CONFIDENCE_LEVEL")
    
    # Business settings
    default_monthly_searches: int = Field(100000, env="DEFAULT_MONTHLY_SEARCHES")
    default_conversion_rate: float = Field(0.02, env="DEFAULT_CONVERSION_RATE")
    default_revenue_per_conversion: float = Field(50.0, env="DEFAULT_REVENUE_PER_CONVERSION")
    
    # Visualization settings
    plot_theme: str = Field("plotly_white", env="PLOT_THEME")
    color_palette: Dict[str, str] = Field(
        default_factory=lambda: {
            "primary": "#1f77b4",
            "secondary": "#ff7f0e",
            "success": "#2ca02c",
            "danger": "#d62728",
            "warning": "#ff9800",
        }
    )
    
    # Logging settings
    log_level: str = Field("INFO", env="LOG_LEVEL")
    log_format: str = Field(
        "{time:YYYY-MM-DD HH:mm:ss} | {level} | {name}:{function}:{line} - {message}",
        env="LOG_FORMAT"
    )
    
    class Config:
        """Pydantic configuration."""
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False


@lru_cache()
def get_settings() -> Settings:
    """Get cached settings instance."""
    return Settings()