"""Utility module for the recommendation impact simulator."""

# Logger configuration
from .logger import (
    setup_logging,
    get_logger,
    log_performance,
    log_business_metric,
    log_causal_estimate,
    log_context,
    log_duration,
)

# Decorators
from .helpers import (
    timeit,
    validate_dataframe,
    handle_errors,
)

# Statistical functions
from .helpers import (
    calculate_confidence_interval,
    calculate_effect_size,
    calculate_statistical_power,
)

# Data processing functions
from .helpers import (
    clean_dataframe,
    encode_categorical_variables,
    calculate_summary_statistics,
)

# Time series functions
from .helpers import (
    generate_date_range,
    calculate_rolling_statistics,
    detect_trend,
)

# Causal inference helpers
from .helpers import (
    calculate_propensity_scores,
    check_covariate_balance,
    calculate_sample_size,
)

# Business metrics
from .helpers import (
    calculate_roi,
    calculate_conversion_metrics,
    calculate_ltv,
)

# Validation functions
from .helpers import (
    validate_treatment_data,
)

# Utility functions
from .helpers import (
    format_number,
    safe_divide,
    chunk_list,
    flatten_dict,
)

__all__ = [
    # Logger
    "setup_logging",
    "get_logger",
    "log_performance",
    "log_business_metric",
    "log_causal_estimate",
    "log_context",
    "log_duration",
    # Decorators
    "timeit",
    "validate_dataframe",
    "handle_errors",
    # Statistical
    "calculate_confidence_interval",
    "calculate_effect_size",
    "calculate_statistical_power",
    # Data processing
    "clean_dataframe",
    "encode_categorical_variables",
    "calculate_summary_statistics",
    # Time series
    "generate_date_range",
    "calculate_rolling_statistics",
    "detect_trend",
    # Causal inference
    "calculate_propensity_scores",
    "check_covariate_balance",
    "calculate_sample_size",
    # Business metrics
    "calculate_roi",
    "calculate_conversion_metrics",
    "calculate_ltv",
    # Validation
    "validate_treatment_data",
    # Utilities
    "format_number",
    "safe_divide",
    "chunk_list",
    "flatten_dict",
]
