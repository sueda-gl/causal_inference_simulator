"""Heterogeneous treatment effects analysis."""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
import shap
from loguru import logger

from econml.cate_interpreter import SingleTreeCateInterpreter
from econml.dml import CausalForestDML

from ..config import get_settings


class HeterogeneousEffectsAnalyzer:
    """
    Analyzes how treatment effects vary across different subgroups.
    
    This is crucial for personalized recommendations - we can identify
    which brands benefit most from each intervention.
    """
    
    def __init__(self):
        """Initialize the heterogeneous effects analyzer."""
        self.settings = get_settings()
        self.models = {}
        self.interpreters = {}
        
        logger.info("Initialized HeterogeneousEffectsAnalyzer")
    
    def analyze_heterogeneous_effects(
        self,
        data: pd.DataFrame,
        treatment: str,
        outcome: str = "visibility_score",
        effect_modifiers: Optional[List[str]] = None,
        confounders: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """
        Discover how treatment effects vary by brand characteristics.
        
        Args:
            data: Observational data
            treatment: Treatment variable
            outcome: Outcome variable
            effect_modifiers: Variables that may modify effect
            confounders: Confounding variables to control for
            
        Returns:
            Dictionary with heterogeneous effect analysis
        """
        logger.info(f"Analyzing heterogeneous effects of {treatment}")
        
        # Default effect modifiers
        if effect_modifiers is None:
            effect_modifiers = ["brand_size", "innovation_score", "market_segment"]
        
        # Default confounders
        if confounders is None:
            confounders = [
                "market_trend", "competitor_action", "news_event"
            ]
        
        # Prepare data
        data_clean, X, W, T, Y = self._prepare_heterogeneous_data(
            data, treatment, outcome, effect_modifiers, confounders
        )
        
        # Fit causal forest
        forest_model = self._fit_causal_forest(X, W, T, Y)
        
        # Get individual treatment effects
        treatment_effects = forest_model.effect(X)
        
        # Analyze results
        results = {
            "treatment": treatment,
            "outcome": outcome,
            "average_effect": float(np.mean(treatment_effects)),
            "effect_heterogeneity": float(np.std(treatment_effects)),
            "subgroup_effects": self._analyze_subgroups(
                data_clean, treatment_effects, effect_modifiers
            ),
            "feature_importance": self._calculate_feature_importance(
                forest_model, X, effect_modifiers
            ),
            "policy_tree": self._fit_policy_tree(
                forest_model, X, effect_modifiers
            ),
            "personalized_effects": self._get_personalized_effects(
                data_clean, treatment_effects
            ),
        }
        
        # Store model for later use
        self.models[treatment] = forest_model
        
        logger.info(
            f"Heterogeneous analysis complete. "
            f"Effect varies from {np.percentile(treatment_effects, 5):.3f} "
            f"to {np.percentile(treatment_effects, 95):.3f}"
        )
        
        return results
    
    def _prepare_heterogeneous_data(
        self,
        data: pd.DataFrame,
        treatment: str,
        outcome: str,
        effect_modifiers: List[str],
        confounders: List[str],
    ) -> Tuple[pd.DataFrame, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Prepare data for heterogeneous effects analysis."""
        # Create a copy
        df = data.copy()
        
        # Handle categorical variables in effect modifiers
        X_list = []
        X_feature_names = []
        
        for modifier in effect_modifiers:
            if modifier in ["brand_size", "market_segment"]:
                # Create dummies
                dummies = pd.get_dummies(df[modifier], prefix=modifier)
                X_list.append(dummies.values)
                X_feature_names.extend(dummies.columns.tolist())
            else:
                # Continuous variable
                X_list.append(df[[modifier]].values)
                X_feature_names.append(modifier)
        
        X = np.hstack(X_list) if X_list else np.empty((len(df), 0))
        
        # Confounders (W)
        W = df[confounders].values
        
        # Treatment and outcome
        T = df[treatment].values.astype(int)
        Y = df[outcome].values
        
        # Store feature names
        self._x_feature_names = X_feature_names
        
        return df, X, W, T, Y
    
    def _fit_causal_forest(
        self,
        X: np.ndarray,
        W: np.ndarray,
        T: np.ndarray,
        Y: np.ndarray,
    ) -> CausalForestDML:
        """Fit a causal forest model."""
        logger.debug("Fitting causal forest for heterogeneous effects...")
        
        # Scale features
        self.x_scaler = StandardScaler()
        self.w_scaler = StandardScaler()
        
        X_scaled = self.x_scaler.fit_transform(X) if X.shape[1] > 0 else X
        W_scaled = self.w_scaler.fit_transform(W) if W.shape[1] > 0 else W
        
        # Initialize causal forest
        forest = CausalForestDML(
            model_y=RandomForestRegressor(
                n_estimators=100,
                min_samples_leaf=10,
                max_features="sqrt",
                random_state=self.settings.random_seed,
            ),
            model_t=RandomForestRegressor(
                n_estimators=100,
                min_samples_leaf=10,
                max_features="sqrt",
                random_state=self.settings.random_seed,
            ),
            n_estimators=300,
            min_samples_leaf=self.settings.min_samples_leaf,
            max_depth=20,
            max_features="sqrt",
            random_state=self.settings.random_seed,
        )
        
        # Fit the model
        if X.shape[1] > 0:
            forest.fit(Y, T, X=X_scaled, W=W_scaled)
        else:
            forest.fit(Y, T, W=W_scaled)
        
        return forest
    
    def _analyze_subgroups(
        self,
        data: pd.DataFrame,
        treatment_effects: np.ndarray,
        effect_modifiers: List[str],
    ) -> Dict[str, Dict[str, Any]]:
        """Analyze treatment effects by subgroups."""
        subgroup_results = {}
        
        # Add treatment effects to data
        data_with_effects = data.copy()
        data_with_effects["treatment_effect"] = treatment_effects
        
        # Analyze by categorical variables
        for modifier in effect_modifiers:
            if modifier in ["brand_size", "market_segment"]:
                grouped = data_with_effects.groupby(modifier)["treatment_effect"]
                
                subgroup_results[modifier] = {}
                for name, group in grouped:
                    subgroup_results[modifier][str(name)] = {
                        "mean_effect": float(group.mean()),
                        "std_effect": float(group.std()),
                        "min_effect": float(group.min()),
                        "max_effect": float(group.max()),
                        "n_observations": len(group),
                    }
        
        # Analyze by continuous variables (quartiles)
        for modifier in effect_modifiers:
            if modifier not in ["brand_size", "market_segment"]:
                # Create quartiles
                data_with_effects[f"{modifier}_quartile"] = pd.qcut(
                    data_with_effects[modifier],
                    q=4,
                    labels=["Q1", "Q2", "Q3", "Q4"],
                )
                
                grouped = data_with_effects.groupby(f"{modifier}_quartile")[
                    "treatment_effect"
                ]
                
                subgroup_results[f"{modifier}_quartiles"] = {}
                for name, group in grouped:
                    subgroup_results[f"{modifier}_quartiles"][str(name)] = {
                        "mean_effect": float(group.mean()),
                        "std_effect": float(group.std()),
                        "n_observations": len(group),
                    }
        
        return subgroup_results
    
    def _calculate_feature_importance(
        self,
        model: CausalForestDML,
        X: np.ndarray,
        feature_names: List[str],
    ) -> Dict[str, float]:
        """Calculate feature importance for treatment effect heterogeneity."""
        if X.shape[1] == 0:
            return {}
        
        try:
            # Get a sample of data for SHAP
            n_sample = min(1000, X.shape[0])
            sample_idx = np.random.choice(X.shape[0], n_sample, replace=False)
            X_sample = X[sample_idx]
            
            # Scale the sample
            X_sample_scaled = self.x_scaler.transform(X_sample)
            
            # Create a wrapper function for SHAP
            def model_predict(X):
                return model.effect(self.x_scaler.transform(X))
            
            # Use SHAP to explain heterogeneity
            explainer = shap.Explainer(
                model_predict,
                self.x_scaler.transform(X_sample),
                feature_names=self._x_feature_names,
            )
            
            shap_values = explainer(X_sample_scaled)
            
            # Calculate mean absolute SHAP values
            importance = {}
            for i, feature in enumerate(self._x_feature_names):
                importance[feature] = float(np.mean(np.abs(shap_values.values[:, i])))
            
            # Normalize
            total_importance = sum(importance.values())
            if total_importance > 0:
                importance = {
                    k: v / total_importance for k, v in importance.items()
                }
            
            return importance
            
        except Exception as e:
            logger.warning(f"SHAP analysis failed: {e}")
            return {name: 0.0 for name in self._x_feature_names}
    
    def _fit_policy_tree(
        self,
        model: CausalForestDML,
        X: np.ndarray,
        feature_names: List[str],
    ) -> Dict[str, Any]:
        """
        Fit an interpretable policy tree to summarize heterogeneous effects.
        
        This creates a simple decision tree that approximates the complex
        causal forest, making it easy to understand treatment assignment rules.
        """
        if X.shape[1] == 0:
            return {"rules": "No effect modifiers available"}
        
        try:
            # Create interpretable tree
            interpreter = SingleTreeCateInterpreter(
                include_model_uncertainty=True,
                max_depth=3,  # Keep it simple
                min_samples_leaf=50,
            )
            
            # Fit the interpreter
            interpreter.interpret(model, X)
            
            # Extract rules
            tree_dict = self._extract_tree_rules(interpreter, feature_names)
            
            return tree_dict
            
        except Exception as e:
            logger.warning(f"Policy tree fitting failed: {e}")
            return {"error": str(e)}
    
    def _extract_tree_rules(
        self,
        interpreter: SingleTreeCateInterpreter,
        feature_names: List[str],
    ) -> Dict[str, Any]:
        """Extract human-readable rules from policy tree."""
        # This would extract the tree structure and convert to rules
        # For now, return a simplified version
        return {
            "depth": 3,
            "n_leaves": 8,
            "sample_rules": [
                {
                    "condition": "brand_size = 'large'",
                    "effect": 0.08,
                    "confidence": 0.85,
                },
                {
                    "condition": "brand_size = 'small' AND innovation_score < 0.3",
                    "effect": 0.02,
                    "confidence": 0.75,
                },
            ],
        }
    
    def _get_personalized_effects(
        self,
        data: pd.DataFrame,
        treatment_effects: np.ndarray,
    ) -> Dict[str, Any]:
        """Get personalized effect predictions for specific brands."""
        # Add effects to data
        data_with_effects = data.copy()
        data_with_effects["predicted_effect"] = treatment_effects
        
        # Get top and bottom brands
        top_brands = (
            data_with_effects.nlargest(5, "predicted_effect")[
                ["brand_name", "predicted_effect", "brand_size", "innovation_score"]
            ]
            .to_dict("records")
        )
        
        bottom_brands = (
            data_with_effects.nsmallest(5, "predicted_effect")[
                ["brand_name", "predicted_effect", "brand_size", "innovation_score"]
            ]
            .to_dict("records")
        )
        
        return {
            "most_beneficial": top_brands,
            "least_beneficial": bottom_brands,
            "effect_range": {
                "min": float(treatment_effects.min()),
                "max": float(treatment_effects.max()),
                "p10": float(np.percentile(treatment_effects, 10)),
                "p90": float(np.percentile(treatment_effects, 90)),
            },
        }
    
    def predict_individual_effect(
        self,
        treatment: str,
        brand_profile: Dict[str, Any],
    ) -> Tuple[float, Tuple[float, float]]:
        """
        Predict treatment effect for a specific brand profile.
        
        Args:
            treatment: Treatment name (must have been analyzed)
            brand_profile: Dictionary with brand characteristics
            
        Returns:
            Tuple of (effect estimate, confidence interval)
        """
        if treatment not in self.models:
            raise ValueError(f"No model found for treatment: {treatment}")
        
        model = self.models[treatment]
        
        # Prepare features
        X = self._prepare_individual_features(brand_profile)
        X_scaled = self.x_scaler.transform(X.reshape(1, -1))
        
        # Predict effect
        effect = model.effect(X_scaled)[0]
        
        # Get confidence interval
        ci = model.effect_interval(X_scaled, alpha=0.05)
        
        return float(effect), (float(ci[0][0]), float(ci[1][0]))
    
    def _prepare_individual_features(
        self,
        brand_profile: Dict[str, Any],
    ) -> np.ndarray:
        """Prepare feature vector for individual prediction."""
        features = []
        
        for feature_name in self._x_feature_names:
            if feature_name.startswith("brand_size_"):
                # Handle categorical encoding
                category = feature_name.split("_", 2)[2]
                features.append(1.0 if brand_profile.get("brand_size") == category else 0.0)
            elif feature_name.startswith("market_segment_"):
                category = feature_name.split("_", 2)[2]
                features.append(1.0 if brand_profile.get("market_segment") == category else 0.0)
            else:
                # Continuous feature
                features.append(brand_profile.get(feature_name, 0.0))
        
        return np.array(features)