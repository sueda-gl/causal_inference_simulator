"""Uncertainty quantification for predictions."""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
from scipy import stats
from loguru import logger

from ..config import get_settings


class UncertaintyQuantifier:
    """
    Quantifies uncertainty in causal predictions.
    
    This includes:
    - Parameter uncertainty from causal estimation
    - Model uncertainty from specification
    - Temporal uncertainty from future predictions
    -# Continuing src/prediction/uncertainty.py

    - Aleatoric uncertainty from inherent randomness
    """
    
    def __init__(self):
        """Initialize the uncertainty quantifier."""
        self.settings = get_settings()
        logger.info("Initialized UncertaintyQuantifier")
    
    def quantify_prediction_uncertainty(
        self,
        base_prediction: float,
        causal_estimates: Dict[str, Any],
        time_horizon: int,
        brand_characteristics: Dict[str, Any],
        n_simulations: int = 1000,
    ) -> Dict[str, Any]:
        """
        Comprehensive uncertainty quantification for predictions.
        
        Args:
            base_prediction: Point estimate of effect
            causal_estimates: Causal effect estimates with uncertainty
            time_horizon: Prediction horizon in weeks
            brand_characteristics: Brand features affecting uncertainty
            n_simulations: Number of Monte Carlo simulations
            
        Returns:
            Dictionary with uncertainty metrics and intervals
        """
        logger.debug(f"Quantifying uncertainty for {time_horizon} week prediction")
        
        # Run Monte Carlo simulation
        simulated_outcomes = self._monte_carlo_simulation(
            base_prediction,
            causal_estimates,
            time_horizon,
            brand_characteristics,
            n_simulations,
        )
        
        # Calculate uncertainty metrics
        uncertainty_metrics = self._calculate_uncertainty_metrics(simulated_outcomes)
        
        # Decompose uncertainty sources
        uncertainty_decomposition = self._decompose_uncertainty(
            simulated_outcomes,
            causal_estimates,
            time_horizon,
        )
        
        # Calculate prediction intervals
        prediction_intervals = self._calculate_prediction_intervals(
            simulated_outcomes,
            confidence_levels=[0.5, 0.8, 0.95],
        )
        
        # Assess prediction reliability
        reliability_assessment = self._assess_reliability(
            uncertainty_metrics,
            time_horizon,
            brand_characteristics,
        )
        
        return {
            "point_estimate": base_prediction,
            "uncertainty_metrics": uncertainty_metrics,
            "uncertainty_decomposition": uncertainty_decomposition,
            "prediction_intervals": prediction_intervals,
            "reliability": reliability_assessment,
            "simulated_distribution": {
                "mean": float(np.mean(simulated_outcomes)),
                "median": float(np.median(simulated_outcomes)),
                "std": float(np.std(simulated_outcomes)),
                "skewness": float(stats.skew(simulated_outcomes)),
                "kurtosis": float(stats.kurtosis(simulated_outcomes)),
            },
        }
    
    def _monte_carlo_simulation(
        self,
        base_prediction: float,
        causal_estimates: Dict[str, Any],
        time_horizon: int,
        brand_characteristics: Dict[str, Any],
        n_simulations: int,
    ) -> np.ndarray:
        """Run Monte Carlo simulation to propagate uncertainty."""
        simulated_outcomes = []
        
        for _ in range(n_simulations):
            # Sample from parameter uncertainty
            param_uncertainty = self._sample_parameter_uncertainty(causal_estimates)
            
            # Add model uncertainty
            model_uncertainty = self._sample_model_uncertainty(brand_characteristics)
            
            # Add temporal uncertainty
            temporal_uncertainty = self._sample_temporal_uncertainty(time_horizon)
            
            # Add aleatoric uncertainty
            aleatoric_uncertainty = np.random.normal(0, 0.01)
            
            # Combine all sources
            total_effect = (
                base_prediction +
                param_uncertainty +
                model_uncertainty +
                temporal_uncertainty +
                aleatoric_uncertainty
            )
            
            # Ensure realistic bounds
            total_effect = np.clip(total_effect, -0.5, 0.5)
            
            simulated_outcomes.append(total_effect)
        
        return np.array(simulated_outcomes)
    
    def _sample_parameter_uncertainty(
        self,
        causal_estimates: Dict[str, Any],
    ) -> float:
        """Sample from parameter uncertainty distribution."""
        if "confidence_interval" in causal_estimates:
            ci = causal_estimates["confidence_interval"]
            # Assume normal distribution
            mean = causal_estimates.get("effect", (ci[0] + ci[1]) / 2)
            std = (ci[1] - ci[0]) / 3.92  # 95% CI
            return np.random.normal(0, std)
        else:
            # Default uncertainty
            return np.random.normal(0, 0.02)
    
    def _sample_model_uncertainty(
        self,
        brand_characteristics: Dict[str, Any],
    ) -> float:
        """Sample from model specification uncertainty."""
        # Higher uncertainty for edge cases
        base_uncertainty = 0.015
        
        # Adjust based on brand characteristics
        if brand_characteristics.get("brand_size") == "small":
            base_uncertainty *= 1.5  # Less data for small brands
        
        if brand_characteristics.get("market_segment") == "emerging":
            base_uncertainty *= 1.3  # More volatile segment
        
        return np.random.normal(0, base_uncertainty)
    
    def _sample_temporal_uncertainty(self, time_horizon: int) -> float:
        """Sample from temporal uncertainty (increases with time)."""
        # Uncertainty grows with square root of time
        temporal_std = 0.005 * np.sqrt(time_horizon)
        return np.random.normal(0, temporal_std)
    
    def _calculate_uncertainty_metrics(
        self,
        simulated_outcomes: np.ndarray,
    ) -> Dict[str, float]:
        """Calculate various uncertainty metrics."""
        return {
            "total_uncertainty": float(np.std(simulated_outcomes)),
            "coefficient_of_variation": float(
                np.std(simulated_outcomes) / (np.mean(simulated_outcomes) + 1e-6)
            ),
            "entropy": float(stats.entropy(
                np.histogram(simulated_outcomes, bins=50)[0] + 1e-6
            )),
            "iqr": float(np.percentile(simulated_outcomes, 75) - 
                        np.percentile(simulated_outcomes, 25)),
            "mad": float(np.median(
                np.abs(simulated_outcomes - np.median(simulated_outcomes))
            )),
        }
    
    def _decompose_uncertainty(
        self,
        simulated_outcomes: np.ndarray,
        causal_estimates: Dict[str, Any],
        time_horizon: int,
    ) -> Dict[str, float]:
        """Decompose uncertainty into different sources."""
        # Estimate contribution of each source
        # This is simplified - in practice would use more sophisticated methods
        
        total_variance = np.var(simulated_outcomes)
        
        # Estimate components
        param_variance = (
            ((causal_estimates.get("confidence_interval", (0, 0.1))[1] -
              causal_estimates.get("confidence_interval", (0, 0.1))[0]) / 3.92) ** 2
        )
        temporal_variance = (0.005 * np.sqrt(time_horizon)) ** 2
        model_variance = 0.015 ** 2
        aleatoric_variance = 0.01 ** 2
        
        # Normalize to percentages
        total_estimated = (
            param_variance + temporal_variance + 
            model_variance + aleatoric_variance
        )
        
        if total_estimated > 0:
            scale_factor = total_variance / total_estimated
            
            return {
                "parameter_uncertainty_pct": float(
                    100 * param_variance * scale_factor / total_variance
                ),
                "temporal_uncertainty_pct": float(
                    100 * temporal_variance * scale_factor / total_variance
                ),
                "model_uncertainty_pct": float(
                    100 * model_variance * scale_factor / total_variance
                ),
                "aleatoric_uncertainty_pct": float(
                    100 * aleatoric_variance * scale_factor / total_variance
                ),
            }
        else:
            return {
                "parameter_uncertainty_pct": 25.0,
                "temporal_uncertainty_pct": 25.0,
                "model_uncertainty_pct": 25.0,
                "aleatoric_uncertainty_pct": 25.0,
            }
    
    def _calculate_prediction_intervals(
        self,
        simulated_outcomes: np.ndarray,
        confidence_levels: List[float],
    ) -> Dict[str, Tuple[float, float]]:
        """Calculate prediction intervals at different confidence levels."""
        intervals = {}
        
        for level in confidence_levels:
            alpha = 1 - level
            lower = np.percentile(simulated_outcomes, 100 * alpha / 2)
            upper = np.percentile(simulated_outcomes, 100 * (1 - alpha / 2))
            
            intervals[f"{int(level * 100)}%"] = (float(lower), float(upper))
        
        return intervals
    
    def _assess_reliability(
        self,
        uncertainty_metrics: Dict[str, float],
        time_horizon: int,
        brand_characteristics: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Assess overall prediction reliability."""
        # Calculate reliability score (0-1)
        reliability_score = 1.0
        
        # Penalize high uncertainty
        if uncertainty_metrics["coefficient_of_variation"] > 0.5:
            reliability_score *= 0.7
        elif uncertainty_metrics["coefficient_of_variation"] > 0.3:
            reliability_score *= 0.85
        
        # Penalize long time horizons
        if time_horizon > 12:
            reliability_score *= 0.8
        elif time_horizon > 8:
            reliability_score *= 0.9
        
        # Penalize edge cases
        if brand_characteristics.get("brand_size") == "small":
            reliability_score *= 0.9
        
        # Determine reliability category
        if reliability_score >= 0.8:
            category = "high"
            interpretation = "Predictions are highly reliable"
        elif reliability_score >= 0.6:
            category = "moderate"
            interpretation = "Predictions are moderately reliable"
        else:
            category = "low"
            interpretation = "Predictions have significant uncertainty"
        
        return {
            "reliability_score": float(reliability_score),
            "reliability_category": category,
            "interpretation": interpretation,
            "confidence_in_prediction": float(reliability_score * 100),
        }
    
    def calculate_value_of_information(
        self,
        current_uncertainty: Dict[str, Any],
        potential_data_sources: List[str],
    ) -> Dict[str, float]:
        """
        Calculate the value of acquiring additional information.
        
        This helps prioritize what data to collect to reduce uncertainty.
        """
        voi_results = {}
        
        current_variance = current_uncertainty["uncertainty_metrics"]["total_uncertainty"] ** 2
        
        # Estimate variance reduction from each data source
        variance_reductions = {
            "more_historical_data": 0.3,  # 30% reduction
            "competitor_analysis": 0.2,
            "market_research": 0.25,
            "ab_test_results": 0.5,  # Most valuable
            "expert_opinions": 0.15,
        }
        
        for source in potential_data_sources:
            if source in variance_reductions:
                # Calculate expected reduction in uncertainty
                reduction_factor = variance_reductions[source]
                new_variance = current_variance * (1 - reduction_factor)
                
                # Value is proportional to uncertainty reduction
                voi = current_variance - new_variance
                
                voi_results[source] = float(voi)
        
        # Normalize to 0-1 scale
        if voi_results:
            max_voi = max(voi_results.values())
            if max_voi > 0:
                voi_results = {k: v / max_voi for k, v in voi_results.items()}
        
        return voi_results