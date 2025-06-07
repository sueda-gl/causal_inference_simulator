"""ROI calculation and business metrics translation."""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from loguru import logger

from ..config import get_settings


@dataclass
class BusinessMetrics:
    """Container for business metrics."""
    
    investment_cost: float
    expected_revenue: float
    roi_percentage: float
    payback_period_days: int
    break_even_probability: float
    net_present_value: float


class ROICalculator:
    """
    Translates visibility improvements to business value.
    
    This is crucial for demonstrating the practical value of
    recommendations to business stakeholders.
    """
    
    def __init__(self):
        """Initialize the ROI calculator."""
        self.settings = get_settings()
        logger.info("Initialized ROICalculator")
    
    def calculate_roi(
        self,
        visibility_increase: float,
        visibility_confidence_interval: Tuple[float, float],
        action_costs: Dict[str, float],
        business_assumptions: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Calculate comprehensive ROI metrics.
        
        Args:
            visibility_increase: Expected visibility increase (0-1)
            visibility_confidence_interval: CI for visibility increase
            action_costs: Cost of each action
            business_assumptions: Business model parameters
            
        Returns:
            Dictionary with ROI metrics and scenarios
        """
        logger.info(f"Calculating ROI for {visibility_increase:.1%} visibility increase")
        
        # Use defaults if not provided
        if business_assumptions is None:
            business_assumptions = self._get_default_assumptions()
        else:
            # Merge provided assumptions with defaults to ensure all required fields exist
            default_assumptions = self._get_default_assumptions()
            business_assumptions = {**default_assumptions, **business_assumptions}
        
        # Calculate base case
        base_metrics = self._calculate_base_case(
            visibility_increase,
            action_costs,
            business_assumptions,
        )
        
        # Calculate scenarios (optimistic, pessimistic)
        scenarios = self._calculate_scenarios(
            visibility_confidence_interval,
            action_costs,
            business_assumptions,
        )
        
        # Calculate sensitivity analysis
        sensitivity = self._sensitivity_analysis(
            visibility_increase,
            action_costs,
            business_assumptions,
        )
        
        # Calculate time-based metrics
        time_metrics = self._calculate_time_metrics(
            base_metrics,
            business_assumptions,
        )
        
        # Risk-adjusted metrics
        risk_adjusted = self._calculate_risk_adjusted_metrics(
            base_metrics,
            scenarios,
            business_assumptions,
        )
        
        return {
            "base_case": base_metrics,
            "scenarios": scenarios,
            "sensitivity": sensitivity,
            "time_metrics": time_metrics,
            "risk_adjusted": risk_adjusted,
            "recommendations": self._generate_roi_recommendations(
                base_metrics, scenarios, sensitivity
            ),
        }
    
    def _get_default_assumptions(self) -> Dict[str, Any]:
        """Get default business assumptions."""
        return {
            "monthly_searches": self.settings.default_monthly_searches,
            "conversion_rate": self.settings.default_conversion_rate,
            "revenue_per_conversion": self.settings.default_revenue_per_conversion,
            "discount_rate": 0.10,  # 10% annual
            "time_horizon_months": 12,
            "market_growth_rate": 0.05,  # 5% annual
            "competitor_response_probability": 0.3,
            "competitor_response_impact": -0.2,  # 20% reduction if competitors respond
        }
    
    def _calculate_base_case(
        self,
        visibility_increase: float,
        action_costs: Dict[str, float],
        assumptions: Dict[str, Any],
    ) -> BusinessMetrics:
        """Calculate base case business metrics."""
        # Total investment
        total_cost = sum(action_costs.values())
        
        # Additional monthly traffic from visibility increase
        additional_traffic = assumptions["monthly_searches"] * visibility_increase
        
        # Monthly conversions and revenue
        monthly_conversions = additional_traffic * assumptions["conversion_rate"]
        monthly_revenue = monthly_conversions * assumptions["revenue_per_conversion"]
        
        # Total revenue over time horizon
        total_revenue = monthly_revenue * assumptions["time_horizon_months"]
        
        # Adjust for market growth
        growth_factor = (1 + assumptions["market_growth_rate"]) ** (
            assumptions["time_horizon_months"] / 12
        )
        total_revenue *= growth_factor
        
        # Calculate ROI
        roi = (total_revenue - total_cost) / total_cost * 100 if total_cost > 0 else 0
        
        # Payback period
        if monthly_revenue > 0:
            payback_months = total_cost / monthly_revenue
            payback_days = int(payback_months * 30)
        else:
            payback_days = 999999  # Never
        
        # Break-even probability (simplified)
        break_even_prob = self._calculate_break_even_probability(
            total_revenue, total_cost, visibility_increase
        )
        
        # NPV calculation
        npv = self._calculate_npv(
            initial_investment=total_cost,
            monthly_cash_flows=[monthly_revenue] * assumptions["time_horizon_months"],
            discount_rate=assumptions["discount_rate"] / 12,  # Monthly rate
        )
        
        return BusinessMetrics(
            investment_cost=total_cost,
            expected_revenue=total_revenue,
            roi_percentage=roi,
            payback_period_days=payback_days,
            break_even_probability=break_even_prob,
            net_present_value=npv,
        )
    
    def _calculate_scenarios(
        self,
        visibility_ci: Tuple[float, float],
        action_costs: Dict[str, float],
        assumptions: Dict[str, Any],
    ) -> Dict[str, BusinessMetrics]:
        """Calculate optimistic and pessimistic scenarios."""
        scenarios = {}
        
        # Pessimistic scenario (lower bound of CI)
        scenarios["pessimistic"] = self._calculate_base_case(
            visibility_ci[0],
            action_costs,
            assumptions,
        )
        
        # Optimistic scenario (upper bound of CI)
        scenarios["optimistic"] = self._calculate_base_case(
            visibility_ci[1],
            action_costs,
            assumptions,
        )
        
        # Competitive response scenario
        competitive_assumptions = assumptions.copy()
        competitive_assumptions["monthly_searches"] *= (
            1 + assumptions["competitor_response_impact"]
        )
        
        scenarios["with_competition"] = self._calculate_base_case(
            (visibility_ci[0] + visibility_ci[1]) / 2,  # Use mean
            action_costs,
            competitive_assumptions,
        )
        
        return scenarios
    
    def _sensitivity_analysis(
        self,
        visibility_increase: float,
        action_costs: Dict[str, float],
        assumptions: Dict[str, Any],
    ) -> Dict[str, Dict[str, float]]:
        """Perform sensitivity analysis on key parameters."""
        sensitivity_results = {}
        
        # Parameters to test
        parameters = {
            "conversion_rate": np.linspace(0.005, 0.05, 5),
            "revenue_per_conversion": np.linspace(25, 100, 5),
            "monthly_searches": np.linspace(50000, 200000, 5),
        }
        
        base_roi = self._calculate_base_case(
            visibility_increase, action_costs, assumptions
        ).roi_percentage
        
        for param, values in parameters.items():
            roi_values = []
            
            for value in values:
                # Create modified assumptions
                test_assumptions = assumptions.copy()
                test_assumptions[param] = value
                
                # Calculate ROI
                metrics = self._calculate_base_case(
                    visibility_increase, action_costs, test_assumptions
                )
                roi_values.append(metrics.roi_percentage)
            
            # Calculate elasticity
            elasticity = np.polyfit(values, roi_values, 1)[0]
            
            sensitivity_results[param] = {
                "values": values.tolist(),
                "roi_values": roi_values,
                "elasticity": float(elasticity),
                "base_value": assumptions[param],
                "base_roi": float(base_roi),
            }
        
        return sensitivity_results
    
    def _calculate_time_metrics(
        self,
        base_metrics: BusinessMetrics,
        assumptions: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Calculate time-based financial metrics."""
        # Monthly cash flows
        monthly_revenue = (
            base_metrics.expected_revenue / assumptions["time_horizon_months"]
        )
        
        # Cumulative cash flow over time
        months = range(1, assumptions["time_horizon_months"] + 1)
        cumulative_revenue = [monthly_revenue * m for m in months]
        cumulative_profit = [
            rev - base_metrics.investment_cost for rev in cumulative_revenue
        ]
        
        # Find break-even month
        break_even_month = None
        for i, profit in enumerate(cumulative_profit):
            if profit > 0:
                break_even_month = i + 1
                break
        
        # IRR calculation (simplified)
        cash_flows = [-base_metrics.investment_cost] + [monthly_revenue] * assumptions["time_horizon_months"]
        irr = self._calculate_irr(cash_flows)
        
        return {
            "monthly_revenue": monthly_revenue,
            "break_even_month": break_even_month,
            "internal_rate_of_return": irr,
            "cumulative_timeline": {
                "months": list(months),
                "revenue": cumulative_revenue,
                "profit": cumulative_profit,
            },
        }
    
    def _calculate_risk_adjusted_metrics(
        self,
        base_metrics: BusinessMetrics,
        scenarios: Dict[str, BusinessMetrics],
        assumptions: Dict[str, Any],
    ) -> Dict[str, float]:
        """Calculate risk-adjusted financial metrics."""
        # Expected value considering probabilities
        prob_optimistic = 0.2
        prob_base = 0.5
        prob_pessimistic = 0.2
        prob_competition = assumptions["competitor_response_probability"]
        
        # Adjust probabilities to sum to 1
        total_prob = prob_optimistic + prob_base + prob_pessimistic
        prob_optimistic /= total_prob * (1 - prob_competition)
        prob_base /= total_prob * (1 - prob_competition)
        prob_pessimistic /= total_prob * (1 - prob_competition)
        
        # Calculate expected ROI
        expected_roi = (
            scenarios["optimistic"].roi_percentage * prob_optimistic +
            base_metrics.roi_percentage * prob_base +
            scenarios["pessimistic"].roi_percentage * prob_pessimistic +
            scenarios["with_competition"].roi_percentage * prob_competition
        )
        
        # Calculate Value at Risk (VaR) - 95% confidence
        roi_values = [
            scenarios["pessimistic"].roi_percentage,
            scenarios["with_competition"].roi_percentage,
            base_metrics.roi_percentage,
            scenarios["optimistic"].roi_percentage,
        ]
        var_95 = np.percentile(roi_values, 5)
        
        # Sharpe ratio (simplified)
        roi_std = np.std(roi_values)
        risk_free_rate = 2  # 2% annual
        sharpe_ratio = (expected_roi - risk_free_rate) / roi_std if roi_std > 0 else 0
        
        return {
            "expected_roi": float(expected_roi),
            "roi_standard_deviation": float(roi_std),
            "value_at_risk_95": float(var_95),
            "sharpe_ratio": float(sharpe_ratio),
            "risk_category": self._categorize_risk(roi_std, var_95),
        }
    
    def _calculate_break_even_probability(
        self,
        expected_revenue: float,
        total_cost: float,
        visibility_increase: float,
    ) -> float:
        """Estimate probability of breaking even."""
        # Handle edge case where expected revenue is zero
        if expected_revenue <= 0:
            return 0.0  # No chance of breaking even with zero revenue
        
        # Simplified calculation based on margin of safety
        margin_of_safety = (expected_revenue - total_cost) / expected_revenue
        
        # Higher margin = higher probability
        if margin_of_safety > 0.5:
            return 0.95
        elif margin_of_safety > 0.3:
            return 0.85
        elif margin_of_safety > 0.1:
            return 0.70
        elif margin_of_safety > 0:
            return 0.55
        else:
            return 0.30
    
    def _calculate_npv(
        self,
        initial_investment: float,
        monthly_cash_flows: List[float],
        discount_rate: float,
    ) -> float:
        """Calculate Net Present Value."""
        npv = -initial_investment
        
        for t, cash_flow in enumerate(monthly_cash_flows, 1):
            npv += cash_flow / (1 + discount_rate) ** t
        
        return npv
    
    def _calculate_irr(self, cash_flows: List[float]) -> float:
        """Calculate Internal Rate of Return."""
        # Use numpy's IRR function if available, otherwise approximate
        try:
            # Simple approximation using NPV = 0
            def npv_at_rate(rate):
                return sum(cf / (1 + rate) ** i for i, cf in enumerate(cash_flows))
            
            # Binary search for IRR
            low, high = -0.5, 0.5
            for _ in range(50):
                mid = (low + high) / 2
                npv = npv_at_rate(mid)
                
                if abs(npv) < 0.01:
                    return mid * 12  # Convert to annual
                elif npv > 0:
                    low = mid
                else:
                    high = mid
            
            return mid * 12
        except:
            return 0.0
    
    def _categorize_risk(
        self,
        roi_std: float,
        var_95: float,
    ) -> str:
        """Categorize investment risk level."""
        if var_95 < -10:
            return "high"
        elif var_95 < 0:
            return "moderate"
        elif roi_std > 50:
            return "moderate"
        else:
            return "low"
    
    def _generate_roi_recommendations(
        self,
        base_metrics: BusinessMetrics,
        scenarios: Dict[str, BusinessMetrics],
        sensitivity: Dict[str, Dict[str, float]],
    ) -> List[Dict[str, Any]]:
        """Generate ROI-based recommendations."""
        recommendations = []
        
        # ROI assessment
        if base_metrics.roi_percentage > 200:
            recommendations.append({
                "type": "strong_positive",
                "message": f"Excellent ROI of {base_metrics.roi_percentage:.0f}%. "
                          "Strong recommendation to proceed.",
                "confidence": "high",
            })
        elif base_metrics.roi_percentage > 50:
            recommendations.append({
                "type": "positive",
                "message": f"Good ROI of {base_metrics.roi_percentage:.0f}%. "
                          "Recommended to proceed with monitoring.",
                "confidence": "moderate",
            })
        else:
            recommendations.append({
                "type": "cautious",
                "message": f"Modest ROI of {base_metrics.roi_percentage:.0f}%. "
                          "Consider starting with lower-cost actions.",
                "confidence": "low",
            })
        
        # Risk assessment
        if scenarios["pessimistic"].roi_percentage < 0:
            recommendations.append({
                "type": "risk_warning",
                "message": "Potential for negative returns in pessimistic scenario. "
                          "Consider risk mitigation strategies.",
                "confidence": "high",
            })
        
        # Sensitivity insights
        most_sensitive = max(
            sensitivity.items(),
            key=lambda x: abs(x[1]["elasticity"])
        )
        
        recommendations.append({
            "type": "sensitivity",
            "message": f"ROI is most sensitive to {most_sensitive[0].replace('_', ' ')}. "
                      f"Focus on optimizing this metric.",
            "confidence": "high",
        })
        
        # Payback period
        if base_metrics.payback_period_days < 90:
            recommendations.append({
                "type": "quick_return",
                "message": f"Quick payback period of {base_metrics.payback_period_days} days. "
                          "Low financial risk.",
                "confidence": "high",
            })
        
        return recommendations
    
    def generate_roi_report(
        self,
        roi_analysis: Dict[str, Any],
        format: str = "summary",
    ) -> str:
        """Generate a formatted ROI report."""
        base = roi_analysis["base_case"]
        scenarios = roi_analysis["scenarios"]
        
        if format == "summary":
            report = f"""
## ROI Analysis Summary

### Base Case Metrics
- **Investment Required**: ${base.investment_cost:,.0f}
- **Expected Revenue**: ${base.expected_revenue:,.0f}
- **ROI**: {base.roi_percentage:.0f}%
- **Payback Period**: {base.payback_period_days} days
- **NPV**: ${base.net_present_value:,.0f}

### Scenario Analysis
- **Optimistic ROI**: {scenarios['optimistic'].roi_percentage:.0f}%
- **Pessimistic ROI**: {scenarios['pessimistic'].roi_percentage:.0f}%
- **With Competition**: {scenarios['with_competition'].roi_percentage:.0f}%

### Risk Assessment
- **Break-even Probability**: {base.break_even_probability:.0%}
- **Risk-adjusted ROI**: {roi_analysis['risk_adjusted']['expected_roi']:.0f}%
- **Risk Level**: {roi_analysis['risk_adjusted']['risk_category'].title()}
"""
        else:
            # Detailed report would include more analysis
            report = "Detailed report format not implemented"
        
        return report