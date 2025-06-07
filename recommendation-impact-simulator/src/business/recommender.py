"""Action recommendation engine based on causal effects and ROI."""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from itertools import combinations
from loguru import logger

from ..data.schemas import BrandProfile, ActionSet
from ..prediction.predictor import WhatIfPredictor
from .roi_calculator import ROICalculator
from ..config import get_settings


class ActionRecommender:
    """
    Recommends optimal actions based on causal effects and business value.
    
    This is the culmination of the analysis - providing specific,
    actionable recommendations tailored to each brand.
    """
    
    def __init__(
        self,
        causal_results: Dict[str, Any],
        heterogeneous_effects: Optional[Dict[str, Any]] = None,
    ):
        """Initialize the action recommender."""
        self.settings = get_settings()
        self.causal_results = causal_results
        self.heterogeneous_effects = heterogeneous_effects
        self.predictor = WhatIfPredictor(causal_results)
        self.roi_calculator = ROICalculator()
        
        logger.info("Initialized ActionRecommender")
    
    def recommend_actions(
        self,
        brand_profile: BrandProfile,
        constraints: Optional[Dict[str, Any]] = None,
        objective: str = "maximize_roi",
    ) -> Dict[str, Any]:
        """
        Generate optimal action recommendations for a brand.
        
        Args:
            brand_profile: Brand characteristics
            constraints: Business constraints (budget, time, etc.)
            objective: Optimization objective
            
        Returns:
            Comprehensive recommendation with rationale
        """
        logger.info(f"Generating recommendations for {brand_profile.brand_name}")
        
        # Default constraints
        if constraints is None:
            constraints = {
                "max_budget": 10000,
                "max_actions": 3,
                "time_horizon": 12,
                "min_roi": 50,
            }
        
        # Generate all possible action combinations
        action_combinations = self._generate_action_combinations(constraints)
        
        # Evaluate each combination
        evaluations = []
        for actions in action_combinations:
            evaluation = self._evaluate_action_combination(
                brand_profile, actions, constraints
            )
            evaluations.append(evaluation)
        
        # Rank combinations by objective
        ranked_combinations = self._rank_combinations(
            evaluations, objective, constraints
        )
        
        # Select optimal combination
        optimal = ranked_combinations[0] if ranked_combinations else None
        
        # Generate detailed recommendation
        recommendation = self._generate_detailed_recommendation(
            brand_profile, optimal, ranked_combinations[:3], constraints
        )
        
        # Add implementation roadmap
        recommendation["implementation_roadmap"] = self._create_implementation_roadmap(
            optimal, brand_profile
        )
        
        # Add monitoring plan
        recommendation["monitoring_plan"] = self._create_monitoring_plan(
            optimal, brand_profile
        )
        
        logger.info(
            f"Recommended {sum(optimal['actions'].dict().values())} actions with "
            f"expected ROI of {optimal['roi']:.0f}%"
        )
        
        return recommendation
    
    def _generate_action_combinations(
        self,
        constraints: Dict[str, Any],
    ) -> List[ActionSet]:
        """Generate all feasible action combinations."""
        available_actions = ["wikipedia_update", "youtube_content", "press_release"]
        
        action_combinations = []
        max_actions = min(constraints.get("max_actions", 3), len(available_actions))
        
        # Generate combinations of different sizes
        for r in range(0, max_actions + 1):
            for combo in combinations(available_actions, r):
                action_set = ActionSet(
                    wikipedia_update="wikipedia_update" in combo,
                    youtube_content="youtube_content" in combo,
                    press_release="press_release" in combo,
                )
                action_combinations.append(action_set)
        
        return action_combinations
    
    def _evaluate_action_combination(
        self,
        brand_profile: BrandProfile,
        actions: ActionSet,
        constraints: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Evaluate a specific combination of actions."""
        # Get cost estimates
        action_costs = self._estimate_action_costs(actions, brand_profile)
        total_cost = sum(action_costs.values())
        
        # Check budget constraint
        if total_cost > constraints.get("max_budget", float("inf")):
            return {
                "actions": actions,
                "feasible": False,
                "reason": "Exceeds budget",
            }
        
        # Predict impact
        prediction = self.predictor.predict_intervention_impact(
            brand_profile,
            actions,
            time_horizon=constraints.get("time_horizon", 12),
        )
        
        # Calculate ROI
        roi_analysis = self.roi_calculator.calculate_roi(
            prediction["expected_increase"],
            prediction["confidence_interval"],
            action_costs,
        )
        
        return {
            "actions": actions,
            "feasible": True,
            "total_cost": total_cost,
            "action_costs": action_costs,
            "expected_visibility_increase": prediction["expected_increase"],
            "confidence_interval": prediction["confidence_interval"],
            "time_to_peak": prediction["time_to_peak"],
            "roi": roi_analysis["base_case"].roi_percentage,
            "npv": roi_analysis["base_case"].net_present_value,
            "payback_days": roi_analysis["base_case"].payback_period_days,
            "risk_adjusted_roi": roi_analysis["risk_adjusted"]["expected_roi"],
            "break_even_probability": roi_analysis["base_case"].break_even_probability,
            "prediction_details": prediction,
            "roi_details": roi_analysis,
        }
    
    def _estimate_action_costs(
        self,
        actions: ActionSet,
        brand_profile: BrandProfile,
    ) -> Dict[str, float]:
        """Estimate costs for each action based on brand profile."""
        base_costs = {
            "wikipedia_update": 500,
            "youtube_content": 2000,
            "press_release": 1000,
        }
        
        # Adjust costs based on brand characteristics
        cost_multipliers = {
            "large": 1.5,
            "medium": 1.0,
            "small": 0.7,
        }
        
        multiplier = cost_multipliers.get(brand_profile.brand_size, 1.0)
        
        action_costs = {}
        for action, is_selected in actions.dict().items():
            if is_selected:
                action_costs[action] = base_costs[action] * multiplier
        
        return action_costs
    
    def _rank_combinations(
        self,
        evaluations: List[Dict[str, Any]],
        objective: str,
        constraints: Dict[str, Any],
    ) -> List[Dict[str, Any]]:
        """Rank action combinations by objective."""
        # Filter feasible combinations
        feasible = [e for e in evaluations if e.get("feasible", False)]
        
        # Apply additional constraints
        filtered = []
        for evaluation in feasible:
            if evaluation["roi"] >= constraints.get("min_roi", 0):
                filtered.append(evaluation)
        
        # Sort by objective
        if objective == "maximize_roi":
            sorted_combos = sorted(filtered, key=lambda x: x["roi"], reverse=True)
        elif objective == "maximize_visibility":
            sorted_combos = sorted(
                filtered,
                key=lambda x: x["expected_visibility_increase"],
                reverse=True,
            )
        elif objective == "minimize_risk":
            sorted_combos = sorted(
                filtered,
                key=lambda x: x["break_even_probability"],
                reverse=True,
            )
        elif objective == "quick_wins":
            sorted_combos = sorted(filtered, key=lambda x: x["payback_days"])
        else:
            sorted_combos = filtered
        
        return sorted_combos
    
    def _generate_detailed_recommendation(
        self,
        brand_profile: BrandProfile,
        optimal: Dict[str, Any],
        top_alternatives: List[Dict[str, Any]],
        constraints: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Generate comprehensive recommendation with rationale."""
        if not optimal:
            return {
                "status": "no_recommendation",
                "reason": "No feasible actions found within constraints",
                "suggestions": ["Increase budget", "Relax ROI requirements"],
            }
        
        # Extract recommended actions
        recommended_actions = [
            action for action, selected in optimal["actions"].dict().items()
            if selected
        ]
        
        # Generate rationale
        rationale = self._generate_rationale(
            brand_profile, optimal, recommended_actions
        )
        
        # Compare with alternatives
        comparison = self._compare_alternatives(optimal, top_alternatives)
        
        return {
            "status": "success",
            "recommended_actions": recommended_actions,
            "investment_required": optimal["total_cost"],
            "expected_outcomes": {
                "visibility_increase": f"{optimal['expected_visibility_increase']:.1%}",
                "roi": f"{optimal['roi']:.0f}%",
                "payback_period": f"{optimal['payback_days']} days",
                "confidence": f"{optimal['break_even_probability']:.0%}",
            },
            "rationale": rationale,
            "alternatives": comparison,
            "detailed_metrics": optimal,
            "constraints_applied": constraints,
        }
    
    def _generate_rationale(
        self,
        brand_profile: BrandProfile,
        optimal: Dict[str, Any],
        recommended_actions: List[str],
    ) -> List[str]:
        """Generate rationale for recommendations."""
        rationale = []
        
        # Overall rationale
        rationale.append(
            f"This combination maximizes ROI ({optimal['roi']:.0f}%) while "
            f"maintaining acceptable risk levels."
        )
        
        # Action-specific rationale
        if "wikipedia_update" in recommended_actions:
            if brand_profile.brand_size == "large":
                rationale.append(
                    "Wikipedia updates are particularly effective for large brands "
                    "like yours, with expected 8-12% visibility increase."
                )
            else:
                rationale.append(
                    "Wikipedia updates provide good baseline visibility improvement "
                    "at relatively low cost."
                )
        
        if "youtube_content" in recommended_actions:
            if brand_profile.innovation_score > 0.7:
                rationale.append(
                    f"Your high innovation score ({brand_profile.innovation_score:.2f}) "
                    "makes YouTube content especially effective for your brand."
                )
            else:
                rationale.append(
                    "YouTube content builds long-term visibility and engages "
                    "younger demographics."
                )
        
        if "press_release" in recommended_actions:
            rationale.append(
                "Press releases provide immediate visibility boost and can "
                "amplify other marketing efforts."
            )
        
        # Timing rationale
        if optimal["time_to_peak"] <= 4:
            rationale.append(
                f"Effects will materialize quickly (peak in {optimal['time_to_peak']} weeks), "
                "allowing for rapid validation and adjustment."
            )
        
        return rationale
    
    def _compare_alternatives(
        self,
        optimal: Dict[str, Any],
        alternatives: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """Compare optimal choice with alternatives."""
        comparisons = []
        
        for i, alt in enumerate(alternatives[1:], 1):  # Skip first (optimal)
            if not alt.get("feasible", False):
                continue
                
            comparison = {
                "rank": i + 1,
                "actions": [
                    a for a, s in alt["actions"].dict().items() if s
                ],
                "roi_difference": alt["roi"] - optimal["roi"],
                "cost_difference": alt["total_cost"] - optimal["total_cost"],
                "visibility_difference": (
                    alt["expected_visibility_increase"] - 
                    optimal["expected_visibility_increase"]
                ),
                "key_tradeoff": self._identify_key_tradeoff(optimal, alt),
            }
            comparisons.append(comparison)
        
        return comparisons
    
    def _identify_key_tradeoff(
        self,
        optimal: Dict[str, Any],
        alternative: Dict[str, Any],
    ) -> str:
        """Identify the key tradeoff between two options."""
        if alternative["roi"] > optimal["roi"]:
            return f"Higher ROI (+{alternative['roi'] - optimal['roi']:.0f}%) but higher cost"
        elif alternative["total_cost"] < optimal["total_cost"]:
            return f"Lower cost (-${optimal['total_cost'] - alternative['total_cost']:.0f}) but lower returns"
        elif alternative["payback_days"] < optimal["payback_days"]:
            return f"Faster payback ({alternative['payback_days']} days) but lower total ROI"
        else:
            return "Similar performance with different action mix"
    
    def _create_implementation_roadmap(
        self,
        optimal: Dict[str, Any],
        brand_profile: BrandProfile,
    ) -> List[Dict[str, Any]]:
        """Create step-by-step implementation roadmap."""
        roadmap = []
        week = 0
        
        # Determine action order based on dependencies and impact
        actions = [a for a, s in optimal["actions"].dict().items() if s]
        
        # Press release usually goes last
        if "press_release" in actions:
            actions.remove("press_release")
            actions.append("press_release")
        
        for action in actions:
            if action == "wikipedia_update":
                roadmap.append({
                    "week": week,
                    "action": "Wikipedia Update",
                    "tasks": [
                        "Audit current Wikipedia presence",
                        "Gather reliable sources and citations",
                        "Draft neutral, factual content updates",
                        "Submit changes through proper channels",
                        "Monitor for approval and edits",
                    ],
                    "duration_weeks": 2,
                    "responsible_team": "Content Marketing",
                    "success_metrics": ["Page views", "Edit persistence", "Link clicks"],
                })
                week += 2
                
            elif action == "youtube_content":
                roadmap.append({
                    "week": week,
                    "action": "YouTube Content Creation",
                    "tasks": [
                        "Develop content strategy aligned with brand",
                        "Create production schedule",
                        "Film initial video series",
                        "Optimize for search and discovery",
                        "Promote across other channels",
                    ],
                    "duration_weeks": 4,
                    "responsible_team": "Video Marketing",
                    "success_metrics": ["Views", "Engagement rate", "Subscriber growth"],
                })
                week += 3
                
            elif action == "press_release":
                roadmap.append({
                    "week": week,
                    "action": "Press Release",
                    "tasks": [
                        "Identify newsworthy angle",
                        "Draft compelling press release",
                        "Build media list for distribution",
                        "Distribute through PR channels",
                        "Follow up with key journalists",
                    ],
                    "duration_weeks": 1,
                    "responsible_team": "PR Team",
                    "success_metrics": ["Media pickups", "Reach", "Backlinks"],
                })
                week += 1
        
        return roadmap
    
    def _create_monitoring_plan(
        self,
        optimal: Dict[str, Any],
        brand_profile: BrandProfile,
    ) -> Dict[str, Any]:
        """Create monitoring plan to track results."""
        return {
            "kpis": [
                {
                    "metric": "AI Visibility Score",
                    "baseline": brand_profile.base_visibility,
                    "target": brand_profile.base_visibility + optimal["expected_visibility_increase"],
                    "measurement_frequency": "Weekly",
                    "data_source": "Platform Analytics",
                },
                {
                    "metric": "Organic Traffic",
                    "baseline": "Current",
                    "target": f"+{optimal['expected_visibility_increase'] * 100:.0f}%",
                    "measurement_frequency": "Weekly",
                    "data_source": "Google Analytics",
                },
                {
                    "metric": "Conversion Rate",
                    "baseline": "Current",
                    "target": "Maintain or improve",
                    "measurement_frequency": "Weekly",
                    "data_source": "CRM",
                },
                {
                    "metric": "ROI",
                    "baseline": 0,
                    "target": f"{optimal['roi']:.0f}%",
                    "measurement_frequency": "Monthly",
                    "data_source": "Financial Reports",
                },
            ],
            "checkpoints": [
                {
                    "week": 2,
                    "milestone": "Initial implementation complete",
                    "success_criteria": "All actions initiated",
                    "decision_point": "Continue or adjust tactics",
                },
                {
                    "week": optimal["time_to_peak"],
                    "milestone": "Peak effect expected",
                    "success_criteria": f">{optimal['expected_visibility_increase']/2:.1%} visibility increase",
                    "decision_point": "Scale up or maintain",
                },
                {
                    "week": 12,
                    "milestone": "Quarterly review",
                    "success_criteria": f"ROI >{optimal['roi']/2:.0f}%",
                    "decision_point": "Continue, expand, or pivot strategy",
                },
            ],
            "alerts": [
                {
                    "condition": "Visibility decrease >5%",
                    "action": "Investigate competitor actions",
                },
                {
                    "condition": "ROI below target by >20%",
                    "action": "Review and adjust strategy",
                },
                {
                    "condition": "Unexpected positive results >150% of target",
                    "action": "Analyze for scaling opportunities",
                },
            ],
        }
    
    def generate_executive_summary(
        self,
        recommendation: Dict[str, Any],
        brand_profile: BrandProfile,
    ) -> str:
        """Generate executive summary of recommendations."""
        if recommendation["status"] != "success":
            return f"No viable recommendations found. {recommendation.get('reason', '')}"
        
        actions = recommendation["recommended_actions"]
        outcomes = recommendation["expected_outcomes"]
        
        summary = f"""
## Executive Summary: AI Visibility Recommendations for {brand_profile.brand_name}

### Recommended Actions
{', '.join([a.replace('_', ' ').title() for a in actions])}

### Investment & Returns
- **Total Investment**: ${recommendation['investment_required']:,.0f}
- **Expected ROI**: {outcomes['roi']}
- **Payback Period**: {outcomes['payback_period']}
- **Visibility Increase**: {outcomes['visibility_increase']}

### Key Rationale
{'. '.join(recommendation['rationale'][:2])}

### Next Steps
1. Approve budget allocation of ${recommendation['investment_required']:,.0f}
2. Assign teams according to implementation roadmap
3. Begin execution in priority order
4. Monitor KPIs weekly and adjust as needed

**Confidence Level**: {outcomes['confidence']} probability of achieving targets
"""
        
        return summary