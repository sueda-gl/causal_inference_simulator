"""Validators for causal assumptions and model diagnostics."""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from scipy import stats
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from loguru import logger

from ..config import get_settings


class CausalAssumptionValidator:
    """
    Validates key assumptions required for causal inference.
    
    This includes checking for:
    - Positivity (overlap)
    - Unconfoundedness (via balance checks)
    - SUTVA (no interference)
    - Model specification
    """
    
    def __init__(self):
        """Initialize the assumption validator."""
        self.settings = get_settings()
        logger.info("Initialized CausalAssumptionValidator")
    
    def validate_all_assumptions(
        self,
        data: pd.DataFrame,
        treatment: str,
        outcome: str,
        confounders: List[str],
    ) -> Dict[str, Any]:
        """Run all assumption checks and return comprehensive report."""
        logger.info(f"Validating causal assumptions for {treatment} → {outcome}")
        
        results = {
            "treatment": treatment,
            "outcome": outcome,
            "assumptions": {
                "positivity": self._check_positivity(data, treatment, confounders),
                "balance": self._check_covariate_balance(data, treatment, confounders),
                "common_support": self._check_common_support(data, treatment, confounders),
                "no_interference": self._check_no_interference(data, treatment, outcome),
            },
            "overall_validity": None,
            "warnings": [],
            "recommendations": [],
        }
        
        # Assess overall validity
        results["overall_validity"] = self._assess_overall_validity(results)
        
        # Generate recommendations
        results["recommendations"] = self._generate_recommendations(results)
        
        logger.info(f"Assumption validation complete. Overall: {results['overall_validity']}")
        
        return results
    
    def _check_positivity(
        self,
        data: pd.DataFrame,
        treatment: str,
        confounders: List[str],
    ) -> Dict[str, Any]:
        """
        Check positivity assumption (overlap in propensity scores).
        
        Both treated and untreated units should exist across the
        covariate space.
        """
        logger.debug("Checking positivity assumption...")
        
        # Fit propensity score model
        from sklearn.linear_model import LogisticRegression
        from sklearn.preprocessing import StandardScaler
        
        # Prepare data
        X = pd.get_dummies(data[confounders], drop_first=True)
        y = data[treatment].astype(int)
        
        # Scale features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Fit model
        ps_model = LogisticRegression(max_iter=1000, random_state=self.settings.random_seed)
        ps_model.fit(X_scaled, y)
        
        # Get propensity scores
        propensity_scores = ps_model.predict_proba(X_scaled)[:, 1]
        
        # Check overlap
        treated_ps = propensity_scores[y == 1]
        control_ps = propensity_scores[y == 0]
        
        # Calculate overlap metrics
        overlap_metrics = {
            "min_treated_ps": float(treated_ps.min()),
            "max_control_ps": float(control_ps.max()),
            "common_support_region": (
                float(max(treated_ps.min(), control_ps.min())),
                float(min(treated_ps.max(), control_ps.max())),
            ),
            "percent_off_support": float(
                np.mean((propensity_scores < 0.1) | (propensity_scores > 0.9)) * 100
            ),
        }
        
        # Assess positivity
        positivity_violated = (
            overlap_metrics["min_treated_ps"] > 0.9 or
            overlap_metrics["max_control_ps"] < 0.1 or
            overlap_metrics["percent_off_support"] > 20
        )
        
        return {
            "satisfied": not positivity_violated,
            "metrics": overlap_metrics,
            "propensity_score_distribution": {
                "treated": {
                    "mean": float(treated_ps.mean()),
                    "std": float(treated_ps.std()),
                    "min": float(treated_ps.min()),
                    "max": float(treated_ps.max()),
                },
                "control": {
                    "mean": float(control_ps.mean()),
                    "std": float(control_ps.std()),
                    "min": float(control_ps.min()),
                    "max": float(control_ps.max()),
                },
            },
        }
    
    def _check_covariate_balance(
        self,
        data: pd.DataFrame,
        treatment: str,
        confounders: List[str],
    ) -> Dict[str, Any]:
        """
        Check covariate balance between treated and control groups.
        
        Large imbalances suggest potential confounding.
        """
        logger.debug("Checking covariate balance...")
        
        treated = data[data[treatment] == 1]
        control = data[data[treatment] == 0]
        
        balance_results = {}
        max_smd = 0
        
        for confounder in confounders:
            if confounder in ["brand_size", "market_segment"]:
                # Categorical variable - check proportions
                treated_props = treated[confounder].value_counts(normalize=True)
                control_props = control[confounder].value_counts(normalize=True)
                
                # Calculate standardized difference for each category
                for category in set(treated_props.index) | set(control_props.index):
                    p_t = treated_props.get(category, 0)
                    p_c = control_props.get(category, 0)
                    
                    # Standardized mean difference
                    smd = abs(p_t - p_c) / np.sqrt((p_t * (1 - p_t) + p_c * (1 - p_c)) / 2)
                    
                    balance_results[f"{confounder}_{category}"] = {
                        "treated_prop": float(p_t),
                        "control_prop": float(p_c),
                        "standardized_diff": float(smd),
                        "balanced": smd < 0.1,  # Common threshold
                    }
                    
                    max_smd = max(max_smd, smd)
            else:
                # Continuous variable
                treated_mean = treated[confounder].mean()
                control_mean = control[confounder].mean()
                
                treated_std = treated[confounder].std()
                control_std = control[confounder].std()
                
                # Standardized mean difference
                pooled_std = np.sqrt((treated_std**2 + control_std**2) / 2)
                smd = abs(treated_mean - control_mean) / pooled_std if pooled_std > 0 else 0
                
                balance_results[confounder] = {
                    "treated_mean": float(treated_mean),
                    "control_mean": float(control_mean),
                    "standardized_diff": float(smd),
                    "balanced": smd < 0.1,
                }
                
                max_smd = max(max_smd, smd)
        
        return {
            "satisfied": max_smd < 0.25,  # Less stringent overall threshold
            "max_standardized_diff": float(max_smd),
            "covariate_balance": balance_results,
            "n_imbalanced": sum(
                1 for r in balance_results.values() if not r["balanced"]
            ),
        }
    
    def _check_common_support(
        self,
        data: pd.DataFrame,
        treatment: str,
        confounders: List[str],
    ) -> Dict[str, Any]:
        """Check for common support in covariate distributions."""
        logger.debug("Checking common support...")
        
        # For each continuous confounder, check overlap
        support_results = {}
        
        for confounder in confounders:
            if confounder not in ["brand_size", "market_segment"]:
                treated_values = data[data[treatment] == 1][confounder]
                control_values = data[data[treatment] == 0][confounder]
                
                # Calculate overlap
                treated_range = (treated_values.min(), treated_values.max())
                control_range = (control_values.min(), control_values.max())
                
                overlap_start = max(treated_range[0], control_range[0])
                overlap_end = min(treated_range[1], control_range[1])
                
                # Check if there's meaningful overlap
                has_overlap = overlap_start < overlap_end
                
                # Calculate overlap percentage
                if has_overlap:
                    total_range = max(treated_range[1], control_range[1]) - min(
                        treated_range[0], control_range[0]
                    )
                    overlap_range = overlap_end - overlap_start
                    overlap_pct = overlap_range / total_range if total_range > 0 else 0
                else:
                    overlap_pct = 0
                
                support_results[confounder] = {
                    "has_overlap": has_overlap,
                    "overlap_percentage": float(overlap_pct),
                    "treated_range": (float(treated_range[0]), float(treated_range[1])),
                    "control_range": (float(control_range[0]), float(control_range[1])),
                }
        
        # Overall assessment
        min_overlap = min(
            (r["overlap_percentage"] for r in support_results.values()),
            default=1.0
        )
        
        return {
            "satisfied": min_overlap > 0.5,  # At least 50% overlap
            "min_overlap_percentage": float(min_overlap),
            "covariate_support": support_results,
        }
    
    def _check_no_interference(
        self,
        data: pd.DataFrame,
        treatment: str,
        outcome: str,
    ) -> Dict[str, Any]:
        """
        Check SUTVA (Stable Unit Treatment Value Assumption).
        
        This checks whether units interfere with each other.
        In our context, this might happen if brands react to
        competitors' actions.
        """
        logger.debug("Checking no-interference assumption...")
        
        # Group by time period
        time_groups = data.groupby("week")
        
        # Check if treatment assignment in one unit affects outcomes in others
        interference_tests = []
        
        for week, group in time_groups:
            if len(group) > 1:
                # Check correlation between treatment rate and individual outcomes
                treatment_rate = group[treatment].mean()
                
                # For each unit, check if its outcome correlates with others' treatment
                for idx, row in group.iterrows():
                    others_treatment_rate = (
                        group[group.index != idx][treatment].mean()
                    )
                    
                    interference_tests.append({
                        "unit_outcome": row[outcome],
                        "others_treatment_rate": others_treatment_rate,
                        "week": week,
                    })
        
        if interference_tests:
            # Convert to DataFrame for analysis
            interference_df = pd.DataFrame(interference_tests)
            
            # Calculate correlation
            correlation = interference_df["unit_outcome"].corr(
                interference_df["others_treatment_rate"]
            )
            
            # Test significance
            n = len(interference_df)
            t_stat = correlation * np.sqrt(n - 2) / np.sqrt(1 - correlation**2)
            p_value = 2 * (1 - stats.t.cdf(abs(t_stat), n - 2))
            
            interference_detected = abs(correlation) > 0.1 and p_value < 0.05
        else:
            correlation = 0
            p_value = 1
            interference_detected = False
        
        return {
            "satisfied": not interference_detected,
            "correlation_with_others_treatment": float(correlation),
            "p_value": float(p_value),
            "interpretation": (
                "No significant interference detected"
                if not interference_detected
                else "Potential interference between units detected"
            ),
        }
    
    def _assess_overall_validity(self, results: Dict[str, Any]) -> str:
        """Assess overall validity based on all assumption checks."""
        assumptions = results["assumptions"]
        
        # Count satisfied assumptions
        n_satisfied = sum(
            1 for check in assumptions.values()
            if check.get("satisfied", False)
        )
        
        n_total = len(assumptions)
        
        # Determine overall validity
        if n_satisfied == n_total:
            return "strong"
        elif n_satisfied >= n_total - 1:
            return "moderate"
        elif n_satisfied >= n_total - 2:
            return "weak"
        else:
            return "violated"
    
    def _generate_recommendations(self, results: Dict[str, Any]) -> List[str]:
        """Generate actionable recommendations based on validation results."""
        recommendations = []
        assumptions = results["assumptions"]
        
        # Positivity violations
        if not assumptions["positivity"]["satisfied"]:
            recommendations.append(
                "Consider trimming extreme propensity scores or "
                "restricting analysis to common support region"
            )
        
        # Balance violations
        if not assumptions["balance"]["satisfied"]:
            recommendations.append(
                "Use propensity score matching or weighting to improve balance"
            )
            
            # Identify worst balanced covariates
            worst_balanced = [
                name for name, stats in assumptions["balance"]["covariate_balance"].items()
                if not stats["balanced"]
            ]
            if worst_balanced:
                recommendations.append(
                    f"Pay special attention to: {', '.join(worst_balanced[:3])}"
                )
        
        # Common support violations
        if not assumptions["common_support"]["satisfied"]:
            recommendations.append(
                "Consider stratified analysis within regions of common support"
            )
        
        # Interference violations
        if not assumptions["no_interference"]["satisfied"]:
            recommendations.append(
                "Consider using methods that account for interference, "
                "such as spatial models or network analysis"
            )
        
        # General recommendations
        if results["overall_validity"] in ["weak", "violated"]:
            recommendations.append(
                "Interpret causal estimates with caution and consider "
                "sensitivity analysis for unmeasured confounding"
            )
        
        return recommendations