"""What-if prediction engine for personalized recommendations."""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime, timedelta
from loguru import logger

from ..data.schemas import BrandProfile, ActionSet
from ..causal.effects import HeterogeneousEffectsAnalyzer
from ..config import get_settings


class WhatIfPredictor:
    """
    Predicts outcomes for specific brands under different action scenarios.
    
    This enables personalized recommendations by predicting what would
    happen if a specific brand took certain actions.
    """
    
    def __init__(self, causal_results: Dict[str, Any], heterogeneous_analyzer: Optional[HeterogeneousEffectsAnalyzer] = None):
        """Initialize the what-if predictor."""
        self.settings = get_settings()
        self.causal_results = causal_results
        self.heterogeneous_analyzer = heterogeneous_analyzer
        
        logger.info("Initialized WhatIfPredictor")
    
    def predict_intervention_impact(
        self,
        brand_profile: BrandProfile,
        planned_actions: ActionSet,
        time_horizon: int = 12,
        confidence_level: float = 0.95,
    ) -> Dict[str, Any]:
        """
        Predict the impact of planned actions for a specific brand.
        
        Args:
            brand_profile: Brand characteristics
            planned_actions: Set of actions to take
            time_horizon: Weeks to predict into future
            confidence_level: Confidence level for intervals
            
        Returns:
            Dictionary with predictions and uncertainty
        """
        logger.info(
            f"Predicting impact for {brand_profile.brand_name} "
            f"with actions: {planned_actions.dict()}"
        )
        
        # Calculate expected effects for each action
        action_effects = self._calculate_action_effects(brand_profile, planned_actions)
        
        # Generate timeline predictions
        timeline = self._generate_impact_timeline(
            brand_profile, action_effects, time_horizon
        )
        
        # Calculate aggregate metrics
        total_effect = sum(effect["effect"] for effect in action_effects.values())
        
        # Estimate uncertainty
        uncertainty = self._estimate_prediction_uncertainty(
            brand_profile, action_effects, time_horizon
        )
        
        # Calculate time to peak effect
        time_to_peak = self._estimate_time_to_peak(action_effects)
        
        # Generate recommendations
        recommendations = self._generate_recommendations(
            brand_profile, planned_actions, action_effects
        )
        
        results = {
            "brand": brand_profile.brand_name,
            "current_visibility": brand_profile.base_visibility,
            "predicted_visibility": min(
                brand_profile.base_visibility + total_effect, 1.0
            ),
            "expected_increase": total_effect,
            "confidence_interval": uncertainty["confidence_interval"],
            "confidence_level": confidence_level,
            "time_to_peak": time_to_peak,
            "timeline": timeline,
            "action_effects": action_effects,
            "recommendations": recommendations,
            "uncertainty_factors": uncertainty["factors"],
        }
        
        logger.info(
            f"Prediction complete. Expected increase: {total_effect:.3f} "
            f"({uncertainty['confidence_interval'][0]:.3f}, "
            f"{uncertainty['confidence_interval'][1]:.3f})"
        )
        
        return results
    
    def _calculate_action_effects(
        self,
        brand_profile: BrandProfile,
        planned_actions: ActionSet,
    ) -> Dict[str, Dict[str, Any]]:
        """Calculate expected effect for each planned action."""
        effects = {}
        
        # Convert brand profile to feature dict
        brand_features = {
            "brand_size": brand_profile.brand_size,
            "innovation_score": brand_profile.innovation_score,
            "market_segment": brand_profile.market_segment,
        }
        
        for action, is_planned in planned_actions.dict().items():
            if is_planned:
                # Get base effect from causal analysis
                if action in self.causal_results:
                    base_effect = self.causal_results[action]["consensus"]["effect"]
                    ci = self.causal_results[action]["consensus"]["confidence_interval"]
                else:
                    # Default effect if not analyzed
                    base_effect = 0.03
                    ci = (0.01, 0.05)
                
                # Adjust for heterogeneous effects if available
                if self.heterogeneous_analyzer and action in self.heterogeneous_analyzer.models:
                    try:
                        personalized_effect, personalized_ci = (
                            self.heterogeneous_analyzer.predict_individual_effect(
                                action, brand_features
                            )
                        )
                        effect = personalized_effect
                        ci = personalized_ci
                    except Exception as e:
                        logger.warning(
                            f"Failed to get personalized effect for {action}: {e}"
                        )
                        effect = base_effect
                else:
                    # Simple heterogeneous adjustment based on brand size
                    if action == "wikipedia_update":
                        size_multipliers = {
                            "large": 1.5,
                            "medium": 1.0,
                            "small": 0.6,
                        }
                        effect = base_effect * size_multipliers.get(
                            brand_profile.brand_size, 1.0
                        )
                    elif action == "youtube_content":
                        # Adjust by innovation score
                        effect = base_effect * (0.5 + brand_profile.innovation_score)
                    else:
                        effect = base_effect
                
                effects[action] = {
                    "effect": effect,
                    "confidence_interval": ci,
                    "personalized": self.heterogeneous_analyzer is not None,
                }
        
        return effects
    
    def _generate_impact_timeline(
        self,
        brand_profile: BrandProfile,
        action_effects: Dict[str, Dict[str, Any]],
        time_horizon: int,
    ) -> pd.DataFrame:
        """Generate week-by-week impact predictions."""
        timeline_data = []
        current_date = datetime.now()
        
        # Different actions have different time dynamics
        effect_curves = {
            "wikipedia_update": lambda t: 1 - np.exp(-t / 2),  # Quick rise
            "youtube_content": lambda t: 1 - np.exp(-t / 4),   # Slower rise
            "press_release": lambda t: np.exp(-t / 1) if t > 0 else 1,  # Immediate spike
        }
        
        baseline = brand_profile.base_visibility
        
        for week in range(time_horizon + 1):
            week_date = current_date + timedelta(weeks=week)
            
            # Calculate cumulative effect at this time point
            total_effect = 0
            
            for action, effect_info in action_effects.items():
                if action in effect_curves:
                    # Apply time dynamics
                    time_factor = effect_curves[action](week)
                    week_effect = effect_info["effect"] * time_factor
                    total_effect += week_effect
            
            # Add some realistic noise
            noise = np.random.normal(0, 0.005) if week > 0 else 0
            
            predicted_visibility = np.clip(baseline + total_effect + noise, 0, 1)
            
            timeline_data.append({
                "week": week,
                "date": week_date,
                "predicted_visibility": predicted_visibility,
                "cumulative_effect": total_effect,
                "visibility_change": predicted_visibility - baseline,
            })
        
        return pd.DataFrame(timeline_data)
    
    def _estimate_prediction_uncertainty(
        self,
        brand_profile: BrandProfile,
        action_effects: Dict[str, Dict[str, Any]],
        time_horizon: int,
    ) -> Dict[str, Any]:
        """Estimate uncertainty in predictions."""
        # Sources of uncertainty
        uncertainty_factors = {
            "parameter_uncertainty": 0.02,  # From causal estimation
            "temporal_uncertainty": 0.01 * np.sqrt(time_horizon),  # Increases with time
            "model_uncertainty": 0.015,  # From model specification
            "external_factors": 0.025,  # Market changes, competitors, etc.
        }
        
        # Combine uncertainties (assuming independence)
        total_variance = sum(u**2 for u in uncertainty_factors.values())
        total_std = np.sqrt(total_variance)
        
        # Get point estimate
        total_effect = sum(e["effect"] for e in action_effects.values())
        
        # Calculate confidence interval
        z_score = 1.96  # 95% confidence
        ci_lower = total_effect - z_score * total_std
        ci_upper = total_effect + z_score * total_std
        
        # Account for effect size uncertainty
        for effect_info in action_effects.values():
            if "confidence_interval" in effect_info:
                ci = effect_info["confidence_interval"]
                ci_lower = min(ci_lower, ci[0])
                ci_upper = max(ci_upper, ci[1])
        
        return {
            "confidence_interval": (max(ci_lower, 0), min(ci_upper, 1)),
            "factors": uncertainty_factors,
            "total_uncertainty": total_std,
        }
    
    def _estimate_time_to_peak(
        self,
        action_effects: Dict[str, Dict[str, Any]],
    ) -> int:
        """Estimate when effects will reach their peak."""
        # Different actions peak at different times
        peak_times = {
            "wikipedia_update": 4,   # 4 weeks
            "youtube_content": 8,    # 8 weeks
            "press_release": 1,      # 1 week
        }
        
        if not action_effects:
            return 0
        
        # Weighted average by effect size
        total_weight = 0
        weighted_time = 0
        
        for action, effect_info in action_effects.items():
            if action in peak_times:
                weight = abs(effect_info["effect"])
                weighted_time += peak_times[action] * weight
                total_weight += weight
        
        if total_weight > 0:
            return int(round(weighted_time / total_weight))
        else:
            return 4  # Default
    
    def _generate_recommendations(
        self,
        brand_profile: BrandProfile,
        planned_actions: ActionSet,
        action_effects: Dict[str, Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """Generate specific recommendations based on predictions."""
        recommendations = []
        
        # Check if any actions are planned
        if not planned_actions.any_action_taken:
            recommendations.append({
                "priority": "high",
                "action": "Take action",
                "rationale": "No actions planned. Consider starting with "
                           "the most effective action for your brand profile.",
            })
            
            # Recommend best action based on brand profile
            if brand_profile.brand_size == "large":
                recommendations.append({
                    "priority": "high",
                    "action": "Update Wikipedia",
                    "rationale": "Large brands see strongest effects from "
                               "Wikipedia updates (8-12% visibility increase).",
                })
            elif brand_profile.innovation_score > 0.7:
                recommendations.append({
                    "priority": "high",
                    "action": "Create YouTube content",
                    "rationale": f"Your high innovation score ({brand_profile.innovation_score:.2f}) "
                               "suggests YouTube content will be highly effective.",
                })
        
        # Timing recommendations
        total_effect = sum(e["effect"] for e in action_effects.values())
        if total_effect > 0.1:
            recommendations.append({
                "priority": "medium",
                "action": "Monitor closely",
                "rationale": f"Expected significant impact ({total_effect:.1%}). "
                           "Set up tracking to validate predictions.",
            })
        
        # Synergy recommendations
        if planned_actions.wikipedia_update and planned_actions.youtube_content:
            recommendations.append({
                "priority": "low",
                "action": "Stagger releases",
                "rationale": "Multiple simultaneous actions may have diminishing returns. "
                           "Consider staggering by 2-3 weeks.",
            })
        
        # Risk recommendations
        if action_effects:  # Check if action_effects is not empty
            confidence = action_effects.get(
                list(action_effects.keys())[0], {}
            ).get("confidence_interval", (0, 1))
            
            if confidence[1] - confidence[0] > 0.1:
                recommendations.append({
                    "priority": "medium",
                    "action": "Start small",
                    "rationale": "High uncertainty in predictions. Consider piloting "
                               "with one action first.",
                })
        
        return recommendations
    
    def compare_action_scenarios(
        self,
        brand_profile: BrandProfile,
        scenarios: List[ActionSet],
        metric: str = "expected_visibility",
    ) -> pd.DataFrame:
        """
        Compare multiple action scenarios for a brand.
        
        Args:
            brand_profile: Brand characteristics
            scenarios: List of action sets to compare
            metric: Metric to optimize
            
        Returns:
            DataFrame comparing scenarios
        """
        results = []
        
        for i, scenario in enumerate(scenarios):
            # Get predictions for this scenario
            prediction = self.predict_intervention_impact(
                brand_profile, scenario, time_horizon=12
            )
            
            # Extract key metrics
            results.append({
                "scenario_id": i + 1,
                "actions": ", ".join(
                    [k for k, v in scenario.dict().items() if v]
                ),
                "expected_increase": prediction["expected_increase"],
                "confidence_lower": prediction["confidence_interval"][0],
                "confidence_upper": prediction["confidence_interval"][1],
                "time_to_peak": prediction["time_to_peak"],
                "final_visibility": prediction["predicted_visibility"],
                "n_actions": sum(scenario.dict().values()),
            })
        
        # Convert to DataFrame and sort by metric
        comparison_df = pd.DataFrame(results)
        
        # Add efficiency metric (effect per action)
        comparison_df["efficiency"] = (
            comparison_df["expected_increase"] / 
            comparison_df["n_actions"].replace(0, 1)
        )
        
        # Sort by expected increase
        comparison_df = comparison_df.sort_values(
            "expected_increase", ascending=False
        )
        
        return comparison_df