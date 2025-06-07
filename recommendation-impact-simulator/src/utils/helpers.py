"""Utility functions for the recommendation impact simulator."""

import pandas as pd
import numpy as np
from typing import List, Dict, Tuple, Optional, Union, Any, Callable
from datetime import datetime, timedelta
from functools import wraps
import time
from scipy import stats
from loguru import logger

from ..config import get_settings


# ============================================================================
# Decorators
# ============================================================================

def timeit(func: Callable) -> Callable:
    """Decorator to measure function execution time."""
    @wraps(func)
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        duration = time.time() - start
        logger.debug(f"{func.__name__} took {duration:.3f}s")
        return result
    return wrapper


def validate_dataframe(required_columns: List[str]) -> Callable:
    """Decorator to validate dataframe has required columns."""
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(df: pd.DataFrame, *args, **kwargs):
            missing = set(required_columns) - set(df.columns)
            if missing:
                raise ValueError(f"Missing required columns: {missing}")
            return func(df, *args, **kwargs)
        return wrapper
    return decorator


def handle_errors(default_return=None, log_errors=True) -> Callable:
    """Decorator to handle errors gracefully."""
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                if log_errors:
                    logger.error(f"Error in {func.__name__}: {str(e)}")
                return default_return
        return wrapper
    return decorator


# ============================================================================
# Statistical Functions
# ============================================================================

def calculate_confidence_interval(
    data: Union[np.ndarray, pd.Series],
    confidence_level: float = 0.95,
    method: str = "percentile"
) -> Tuple[float, float]:
    """
    Calculate confidence interval for data.
    
    Args:
        data: Input data
        confidence_level: Confidence level (default: 0.95)
        method: Method to use ('percentile', 't', 'bootstrap')
        
    Returns:
        Tuple of (lower_bound, upper_bound)
    """
    data = np.array(data)
    alpha = 1 - confidence_level
    
    if method == "percentile":
        lower = np.percentile(data, alpha/2 * 100)
        upper = np.percentile(data, (1 - alpha/2) * 100)
    
    elif method == "t":
        mean = np.mean(data)
        se = stats.sem(data)
        interval = stats.t.interval(confidence_level, len(data)-1, loc=mean, scale=se)
        lower, upper = interval
    
    elif method == "bootstrap":
        # Simple bootstrap
        n_bootstrap = 1000
        bootstrap_means = []
        for _ in range(n_bootstrap):
            sample = np.random.choice(data, size=len(data), replace=True)
            bootstrap_means.append(np.mean(sample))
        lower = np.percentile(bootstrap_means, alpha/2 * 100)
        upper = np.percentile(bootstrap_means, (1 - alpha/2) * 100)
    
    else:
        raise ValueError(f"Unknown method: {method}")
    
    return float(lower), float(upper)


def calculate_effect_size(
    treatment_group: Union[np.ndarray, pd.Series],
    control_group: Union[np.ndarray, pd.Series],
    effect_type: str = "cohen_d"
) -> float:
    """
    Calculate effect size between treatment and control groups.
    
    Args:
        treatment_group: Treatment group data
        control_group: Control group data
        effect_type: Type of effect size ('cohen_d', 'glass_delta', 'hedges_g')
        
    Returns:
        Effect size value
    """
    treatment = np.array(treatment_group)
    control = np.array(control_group)
    
    mean_diff = np.mean(treatment) - np.mean(control)
    
    if effect_type == "cohen_d":
        # Pooled standard deviation
        n1, n2 = len(treatment), len(control)
        var1, var2 = np.var(treatment, ddof=1), np.var(control, ddof=1)
        pooled_std = np.sqrt(((n1-1)*var1 + (n2-1)*var2) / (n1+n2-2))
        effect_size = mean_diff / pooled_std
    
    elif effect_type == "glass_delta":
        # Use control group standard deviation
        effect_size = mean_diff / np.std(control, ddof=1)
    
    elif effect_type == "hedges_g":
        # Cohen's d with small sample correction
        cohen_d = calculate_effect_size(treatment, control, "cohen_d")
        n = len(treatment) + len(control)
        correction = 1 - (3 / (4 * n - 9))
        effect_size = cohen_d * correction
    
    else:
        raise ValueError(f"Unknown effect type: {effect_type}")
    
    return float(effect_size)


def calculate_statistical_power(
    effect_size: float,
    n_treatment: int,
    n_control: int,
    alpha: float = 0.05,
    test_type: str = "two_tailed"
) -> float:
    """
    Calculate statistical power for a given effect size and sample size.
    
    Args:
        effect_size: Expected effect size (Cohen's d)
        n_treatment: Treatment group sample size
        n_control: Control group sample size
        alpha: Significance level
        test_type: 'two_tailed' or 'one_tailed'
        
    Returns:
        Statistical power (0-1)
    """
    from statsmodels.stats.power import tt_ind_solve_power
    
    # Calculate effective sample size
    n_eff = (n_treatment * n_control) / (n_treatment + n_control)
    
    # Adjust alpha for test type
    if test_type == "two_tailed":
        alpha_adj = alpha
    else:
        alpha_adj = alpha / 2
    
    power = tt_ind_solve_power(
        effect_size=effect_size,
        nobs1=n_eff,
        alpha=alpha_adj,
        ratio=n_treatment/n_control
    )
    
    return float(power)


# ============================================================================
# Data Processing Functions
# ============================================================================

def clean_dataframe(
    df: pd.DataFrame,
    drop_na_columns: Optional[List[str]] = None,
    fill_na_values: Optional[Dict[str, Any]] = None,
    drop_duplicates: bool = True,
    subset_columns: Optional[List[str]] = None
) -> pd.DataFrame:
    """
    Clean dataframe with various options.
    
    Args:
        df: Input dataframe
        drop_na_columns: Columns to check for NA values
        fill_na_values: Dictionary mapping columns to fill values
        drop_duplicates: Whether to drop duplicate rows
        subset_columns: Columns to consider for duplicate detection
        
    Returns:
        Cleaned dataframe
    """
    df_clean = df.copy()
    
    # Drop rows with NA in specified columns
    if drop_na_columns:
        df_clean = df_clean.dropna(subset=drop_na_columns)
    
    # Fill NA values
    if fill_na_values:
        df_clean = df_clean.fillna(fill_na_values)
    
    # Drop duplicates
    if drop_duplicates:
        df_clean = df_clean.drop_duplicates(subset=subset_columns)
    
    logger.debug(f"Cleaned dataframe: {len(df)} -> {len(df_clean)} rows")
    
    return df_clean


def encode_categorical_variables(
    df: pd.DataFrame,
    categorical_columns: List[str],
    method: str = "onehot",
    drop_first: bool = True
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Encode categorical variables.
    
    Args:
        df: Input dataframe
        categorical_columns: Columns to encode
        method: Encoding method ('onehot', 'label', 'target')
        drop_first: Whether to drop first category (for onehot)
        
    Returns:
        Tuple of (encoded dataframe, encoding mapping)
    """
    df_encoded = df.copy()
    encoding_map = {}
    
    for col in categorical_columns:
        if col not in df.columns:
            continue
            
        if method == "onehot":
            dummies = pd.get_dummies(
                df_encoded[col], 
                prefix=col, 
                drop_first=drop_first
            )
            df_encoded = pd.concat([df_encoded.drop(col, axis=1), dummies], axis=1)
            encoding_map[col] = dummies.columns.tolist()
            
        elif method == "label":
            from sklearn.preprocessing import LabelEncoder
            le = LabelEncoder()
            df_encoded[col] = le.fit_transform(df_encoded[col])
            encoding_map[col] = dict(zip(le.classes_, le.transform(le.classes_)))
            
    return df_encoded, encoding_map


def calculate_summary_statistics(
    df: pd.DataFrame,
    groupby_columns: Optional[List[str]] = None,
    numeric_columns: Optional[List[str]] = None,
    statistics: Optional[List[str]] = None
) -> pd.DataFrame:
    """
    Calculate summary statistics for dataframe.
    
    Args:
        df: Input dataframe
        groupby_columns: Columns to group by
        numeric_columns: Numeric columns to summarize
        statistics: Statistics to calculate
        
    Returns:
        Summary statistics dataframe
    """
    if numeric_columns is None:
        numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
    
    if statistics is None:
        statistics = ['mean', 'std', 'min', 'max', 'median']
    
    if groupby_columns:
        summary = df.groupby(groupby_columns)[numeric_columns].agg(statistics)
    else:
        summary = df[numeric_columns].agg(statistics)
    
    return summary


# ============================================================================
# Time Series Functions
# ============================================================================

def generate_date_range(
    start_date: Union[str, datetime],
    end_date: Union[str, datetime],
    freq: str = 'W'
) -> pd.DatetimeIndex:
    """
    Generate date range.
    
    Args:
        start_date: Start date
        end_date: End date
        freq: Frequency string ('D', 'W', 'M', etc.)
        
    Returns:
        DatetimeIndex
    """
    return pd.date_range(start=start_date, end=end_date, freq=freq)


def calculate_rolling_statistics(
    df: pd.DataFrame,
    value_column: str,
    window: int,
    statistics: List[str] = ['mean', 'std'],
    min_periods: Optional[int] = None
) -> pd.DataFrame:
    """
    Calculate rolling statistics for time series.
    
    Args:
        df: Input dataframe (should be sorted by time)
        value_column: Column to calculate statistics for
        window: Rolling window size
        statistics: Statistics to calculate
        min_periods: Minimum number of observations
        
    Returns:
        DataFrame with rolling statistics
    """
    result = pd.DataFrame(index=df.index)
    
    for stat in statistics:
        if stat == 'mean':
            result[f'{value_column}_rolling_mean'] = df[value_column].rolling(
                window=window, min_periods=min_periods
            ).mean()
        elif stat == 'std':
            result[f'{value_column}_rolling_std'] = df[value_column].rolling(
                window=window, min_periods=min_periods
            ).std()
        elif stat == 'min':
            result[f'{value_column}_rolling_min'] = df[value_column].rolling(
                window=window, min_periods=min_periods
            ).min()
        elif stat == 'max':
            result[f'{value_column}_rolling_max'] = df[value_column].rolling(
                window=window, min_periods=min_periods
            ).max()
    
    return result


def detect_trend(
    series: pd.Series,
    method: str = "linear"
) -> Dict[str, float]:
    """
    Detect trend in time series.
    
    Args:
        series: Time series data
        method: Trend detection method
        
    Returns:
        Dictionary with trend statistics
    """
    x = np.arange(len(series))
    y = series.values
    
    if method == "linear":
        slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
        
        return {
            "slope": slope,
            "intercept": intercept,
            "r_squared": r_value ** 2,
            "p_value": p_value,
            "trend_direction": "increasing" if slope > 0 else "decreasing"
        }
    
    else:
        raise ValueError(f"Unknown method: {method}")


# ============================================================================
# Causal Inference Helper Functions
# ============================================================================

def calculate_propensity_scores(
    df: pd.DataFrame,
    treatment_column: str,
    covariate_columns: List[str],
    model_type: str = "logistic"
) -> np.ndarray:
    """
    Calculate propensity scores.
    
    Args:
        df: Input dataframe
        treatment_column: Treatment indicator column
        covariate_columns: Covariate columns
        model_type: Model type for propensity score
        
    Returns:
        Array of propensity scores
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.ensemble import RandomForestClassifier
    
    X = df[covariate_columns]
    y = df[treatment_column]
    
    if model_type == "logistic":
        model = LogisticRegression(max_iter=1000)
    elif model_type == "random_forest":
        model = RandomForestClassifier(n_estimators=100, random_state=42)
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    model.fit(X, y)
    propensity_scores = model.predict_proba(X)[:, 1]
    
    return propensity_scores


def check_covariate_balance(
    df: pd.DataFrame,
    treatment_column: str,
    covariate_columns: List[str],
    threshold: float = 0.1
) -> Dict[str, Dict[str, float]]:
    """
    Check covariate balance between treatment and control groups.
    
    Args:
        df: Input dataframe
        treatment_column: Treatment indicator column
        covariate_columns: Covariate columns to check
        threshold: Standardized difference threshold
        
    Returns:
        Dictionary with balance statistics
    """
    treated = df[df[treatment_column] == 1]
    control = df[df[treatment_column] == 0]
    
    balance_stats = {}
    
    for col in covariate_columns:
        if df[col].dtype in ['float64', 'int64']:
            # Continuous variable
            mean_treated = treated[col].mean()
            mean_control = control[col].mean()
            std_pooled = np.sqrt(
                (treated[col].var() + control[col].var()) / 2
            )
            
            std_diff = (mean_treated - mean_control) / std_pooled
            
            balance_stats[col] = {
                "mean_treated": mean_treated,
                "mean_control": mean_control,
                "standardized_difference": std_diff,
                "balanced": abs(std_diff) < threshold
            }
        else:
            # Categorical variable
            prop_treated = treated[col].value_counts(normalize=True)
            prop_control = control[col].value_counts(normalize=True)
            
            max_diff = 0
            for category in df[col].unique():
                p_t = prop_treated.get(category, 0)
                p_c = prop_control.get(category, 0)
                diff = abs(p_t - p_c)
                max_diff = max(max_diff, diff)
            
            balance_stats[col] = {
                "max_proportion_difference": max_diff,
                "balanced": max_diff < threshold
            }
    
    return balance_stats


def calculate_sample_size(
    effect_size: float,
    power: float = 0.8,
    alpha: float = 0.05,
    ratio: float = 1.0
) -> Dict[str, int]:
    """
    Calculate required sample size for causal inference.
    
    Args:
        effect_size: Expected effect size
        power: Desired statistical power
        alpha: Significance level
        ratio: Ratio of treatment to control
        
    Returns:
        Dictionary with sample sizes
    """
    from statsmodels.stats.power import tt_ind_solve_power
    
    n_control = tt_ind_solve_power(
        effect_size=effect_size,
        power=power,
        alpha=alpha,
        ratio=ratio
    )
    
    n_treatment = int(n_control * ratio)
    n_control = int(n_control)
    
    return {
        "n_treatment": n_treatment,
        "n_control": n_control,
        "n_total": n_treatment + n_control
    }


# ============================================================================
# Business Metric Functions
# ============================================================================

def calculate_roi(
    revenue: float,
    cost: float,
    time_period_days: Optional[int] = None
) -> Dict[str, float]:
    """
    Calculate return on investment metrics.
    
    Args:
        revenue: Total revenue
        cost: Total cost
        time_period_days: Time period in days
        
    Returns:
        Dictionary with ROI metrics
    """
    profit = revenue - cost
    roi_percentage = (profit / cost) * 100 if cost > 0 else 0
    
    metrics = {
        "revenue": revenue,
        "cost": cost,
        "profit": profit,
        "roi_percentage": roi_percentage,
        "roi_ratio": revenue / cost if cost > 0 else 0
    }
    
    if time_period_days:
        metrics["daily_profit"] = profit / time_period_days
        metrics["annualized_roi"] = roi_percentage * (365 / time_period_days)
    
    return metrics


def calculate_conversion_metrics(
    visitors: int,
    conversions: int,
    revenue_per_conversion: float
) -> Dict[str, float]:
    """
    Calculate conversion funnel metrics.
    
    Args:
        visitors: Number of visitors
        conversions: Number of conversions
        revenue_per_conversion: Revenue per conversion
        
    Returns:
        Dictionary with conversion metrics
    """
    conversion_rate = conversions / visitors if visitors > 0 else 0
    total_revenue = conversions * revenue_per_conversion
    revenue_per_visitor = total_revenue / visitors if visitors > 0 else 0
    
    return {
        "visitors": visitors,
        "conversions": conversions,
        "conversion_rate": conversion_rate,
        "total_revenue": total_revenue,
        "revenue_per_visitor": revenue_per_visitor,
        "revenue_per_conversion": revenue_per_conversion
    }


def calculate_ltv(
    average_order_value: float,
    purchase_frequency: float,
    customer_lifespan_months: float,
    discount_rate: float = 0.1
) -> float:
    """
    Calculate customer lifetime value.
    
    Args:
        average_order_value: Average order value
        purchase_frequency: Purchases per month
        customer_lifespan_months: Customer lifespan in months
        discount_rate: Annual discount rate
        
    Returns:
        Customer lifetime value
    """
    monthly_discount_rate = discount_rate / 12
    ltv = 0
    
    for month in range(int(customer_lifespan_months)):
        monthly_revenue = average_order_value * purchase_frequency
        discounted_revenue = monthly_revenue / ((1 + monthly_discount_rate) ** month)
        ltv += discounted_revenue
    
    return float(ltv)


# ============================================================================
# Validation Functions
# ============================================================================

def validate_treatment_data(
    df: pd.DataFrame,
    treatment_column: str,
    outcome_column: str,
    min_treated: int = 10,
    min_control: int = 10
) -> Dict[str, Any]:
    """
    Validate data for causal inference.
    
    Args:
        df: Input dataframe
        treatment_column: Treatment column name
        outcome_column: Outcome column name
        min_treated: Minimum treated units required
        min_control: Minimum control units required
        
    Returns:
        Validation results dictionary
    """
    results = {
        "valid": True,
        "errors": [],
        "warnings": [],
        "statistics": {}
    }
    
    # Check columns exist
    if treatment_column not in df.columns:
        results["valid"] = False
        results["errors"].append(f"Treatment column '{treatment_column}' not found")
    
    if outcome_column not in df.columns:
        results["valid"] = False
        results["errors"].append(f"Outcome column '{outcome_column}' not found")
    
    if not results["valid"]:
        return results
    
    # Check treatment is binary
    unique_treatments = df[treatment_column].unique()
    if len(unique_treatments) != 2 or set(unique_treatments) != {0, 1}:
        results["valid"] = False
        results["errors"].append("Treatment must be binary (0/1)")
    
    # Check sample sizes
    n_treated = df[df[treatment_column] == 1].shape[0]
    n_control = df[df[treatment_column] == 0].shape[0]
    
    results["statistics"]["n_treated"] = n_treated
    results["statistics"]["n_control"] = n_control
    results["statistics"]["n_total"] = len(df)
    
    if n_treated < min_treated:
        results["valid"] = False
        results["errors"].append(f"Insufficient treated units: {n_treated} < {min_treated}")
    
    if n_control < min_control:
        results["valid"] = False
        results["errors"].append(f"Insufficient control units: {n_control} < {min_control}")
    
    # Check for extreme imbalance
    treatment_rate = n_treated / len(df)
    results["statistics"]["treatment_rate"] = treatment_rate
    
    if treatment_rate < 0.05 or treatment_rate > 0.95:
        results["warnings"].append(
            f"Extreme treatment imbalance: {treatment_rate:.1%} treated"
        )
    
    # Check outcome variance
    outcome_variance = df[outcome_column].var()
    if outcome_variance < 1e-6:
        results["warnings"].append("Outcome has very low variance")
    
    return results


# ============================================================================
# Utility Functions
# ============================================================================

def format_number(
    value: Union[int, float],
    format_type: str = "general",
    decimals: int = 2
) -> str:
    """
    Format number for display.
    
    Args:
        value: Number to format
        format_type: Format type ('general', 'percentage', 'currency', 'scientific')
        decimals: Number of decimal places
        
    Returns:
        Formatted string
    """
    if pd.isna(value):
        return "N/A"
    
    if format_type == "general":
        if abs(value) >= 1000000:
            return f"{value/1000000:.{decimals}f}M"
        elif abs(value) >= 1000:
            return f"{value/1000:.{decimals}f}K"
        else:
            return f"{value:.{decimals}f}"
    
    elif format_type == "percentage":
        return f"{value*100:.{decimals}f}%"
    
    elif format_type == "currency":
        return f"${value:,.{decimals}f}"
    
    elif format_type == "scientific":
        return f"{value:.{decimals}e}"
    
    else:
        return str(value)


def safe_divide(
    numerator: Union[int, float],
    denominator: Union[int, float],
    default: Union[int, float] = 0
) -> float:
    """Safely divide two numbers."""
    if denominator == 0:
        return float(default)
    return float(numerator / denominator)


def chunk_list(lst: List[Any], chunk_size: int) -> List[List[Any]]:
    """Split list into chunks."""
    return [lst[i:i + chunk_size] for i in range(0, len(lst), chunk_size)]


def flatten_dict(
    d: Dict[str, Any],
    parent_key: str = '',
    sep: str = '_'
) -> Dict[str, Any]:
    """Flatten nested dictionary."""
    items = []
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.extend(flatten_dict(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)
