"""Synthetic data generator with embedded causal relationships."""

import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Optional
from datetime import datetime, timedelta
from loguru import logger

from .schemas import BrandProfile, ObservationRecord
from ..config import get_settings


class CausalDataGenerator:
    """
    Generates synthetic brand visibility data with known causal relationships.
    
    This generator creates realistic data where:
    - Brand characteristics influence propensity to take actions
    - Actions have heterogeneous causal effects on visibility
    - Confounders affect both actions and outcomes
    - Temporal dynamics are modeled realistically
    """
    
    def __init__(
        self,
        n_brands: Optional[int] = None,
        n_time_periods: Optional[int] = None,
        random_seed: Optional[int] = None,
    ):
        """Initialize the data generator."""
        settings = get_settings()
        
        self.n_brands = n_brands or settings.n_brands
        self.n_time_periods = n_time_periods or settings.n_time_periods
        self.random_seed = random_seed or settings.random_seed
        
        # Set random seed for reproducibility
        np.random.seed(self.random_seed)
        
        # Define true causal effects (ground truth)
        self.causal_effects = {
            "wikipedia_update": {
                "large": 0.08,
                "medium": 0.05,
                "small": 0.03,
                "noise": 0.01,
            },
            "youtube_content": {
                "base": 0.02,
                "innovation_multiplier": 0.06,
                "noise": 0.01,
            },
            "press_release": {
                "base": 0.03,
                "noise": 0.01,
            },
        }
        
        logger.info(
            f"Initialized CausalDataGenerator with {self.n_brands} brands "
            f"and {self.n_time_periods} time periods"
        )
    
    def generate_brand_profiles(self) -> pd.DataFrame:
        """Generate static brand characteristics."""
        logger.info("Generating brand profiles...")
        
        # Real brand names for realism
        real_brands = [
            "Tesla", "BMW", "Mercedes-Benz", "Audi", "Toyota", "Honda",
            "Ford", "General Motors", "Rivian", "Lucid Motors", "Nio",
            "BYD", "Volkswagen", "Hyundai", "Kia", "Mazda", "Subaru",
            "Volvo", "Porsche", "Ferrari", "Lamborghini", "Jaguar",
        ]
        
        # Extend with synthetic names if needed
        brand_names = real_brands[:self.n_brands]
        for i in range(len(brand_names), self.n_brands):
            brand_names.append(f"AutoBrand_{i}")
        
        profiles = []
        for i, name in enumerate(brand_names):
            # Create correlated characteristics
            is_luxury = name in ["Mercedes-Benz", "BMW", "Audi", "Porsche", 
                               "Ferrari", "Lamborghini", "Jaguar"]
            is_innovative = name in ["Tesla", "Rivian", "Lucid Motors", "Nio"]
            
            # Brand size distribution (correlated with segment)
            if is_luxury:
                size_probs = [0.6, 0.3, 0.1]  # More likely to be large
            elif is_innovative:
                size_probs = [0.2, 0.3, 0.5]  # More likely to be small/emerging
            else:
                size_probs = [0.3, 0.5, 0.2]  # Standard distribution
            
            brand_size = np.random.choice(["large", "medium", "small"], p=size_probs)
            
            # Innovation score (beta distribution for realistic scores)
            if is_innovative:
                innovation_score = np.random.beta(5, 2)  # Skewed high
            elif is_luxury:
                innovation_score = np.random.beta(3, 3)  # Centered
            else:
                innovation_score = np.random.beta(2, 5)  # Skewed low
            
            # Market segment
            if is_luxury:
                market_segment = "luxury"
            elif is_innovative or brand_size == "small":
                market_segment = "emerging"
            else:
                market_segment = "mass"
            
            # Base visibility (correlated with size and segment)
            base_visibility = {
                "large": np.random.uniform(0.3, 0.5),
                "medium": np.random.uniform(0.15, 0.35),
                "small": np.random.uniform(0.05, 0.2),
            }[brand_size]
            
            profile = BrandProfile(
                brand_id=i,
                brand_name=name,
                brand_size=brand_size,
                innovation_score=innovation_score,
                market_segment=market_segment,
                base_visibility=base_visibility,
            )
            
            profiles.append(profile.dict())
        
        df = pd.DataFrame(profiles)
        logger.info(f"Generated {len(df)} brand profiles")
        return df
    
    def generate_time_series_data(
        self, brand_profiles: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Generate time-series observations with embedded causal relationships.
        
        This is the core of the simulation where we create realistic patterns
        with known ground truth causal effects.
        """
        logger.info("Generating time-series data with causal relationships...")
        
        observations = []
        start_date = datetime(2024, 1, 1)
        
        for _, brand in brand_profiles.iterrows():
            # Track brand state over time
            momentum = 0  # Visibility momentum
            recent_actions = []  # Track recent actions for delayed effects
            
            for week in range(self.n_time_periods):
                # Generate time-varying confounders
                market_trend = self._generate_market_trend(week)
                competitor_action = self._generate_competitor_action(brand, week)
                news_event = self._generate_news_event(brand, week)
                
                # Calculate propensity to take actions (selection bias)
                action_propensities = self._calculate_action_propensities(
                    brand, week, market_trend, news_event, momentum
                )
                
                # Stochastically determine actions based on propensities
                actions = {
                    action: np.random.binomial(1, min(prop, 1.0))
                    for action, prop in action_propensities.items()
                }
                
                # Calculate visibility with true causal effects
                visibility = self._calculate_visibility(
                    brand, actions, market_trend, competitor_action, 
                    news_event, momentum, recent_actions
                )
                
                # Update momentum (persistence in visibility changes)
                momentum = 0.7 * momentum + 0.3 * (visibility - brand["base_visibility"])
                
                # Track recent actions for delayed effects
                recent_actions.append(actions)
                if len(recent_actions) > 4:  # Keep last 4 weeks
                    recent_actions.pop(0)
                
                # Create observation record
                observation = ObservationRecord(
                    brand_id=int(brand["brand_id"]),
                    brand_name=brand["brand_name"],
                    week=week,
                    date=start_date + timedelta(weeks=week),
                    visibility_score=visibility,
                    wikipedia_update=bool(actions["wikipedia_update"]),
                    youtube_content=bool(actions["youtube_content"]),
                    press_release=bool(actions["press_release"]),
                    market_trend=market_trend,
                    competitor_action=bool(competitor_action),
                    news_event=bool(news_event),
                    brand_size=brand["brand_size"],
                    innovation_score=brand["innovation_score"],
                    market_segment=brand["market_segment"],
                )
                
                observations.append(observation.dict())
        
        df = pd.DataFrame(observations)
        logger.info(
            f"Generated {len(df)} observations for {self.n_brands} brands "
            f"over {self.n_time_periods} weeks"
        )
        return df
    
    def _generate_market_trend(self, week: int) -> float:
        """Generate market-wide trends with seasonality."""
        # Annual cycle with some noise
        annual_cycle = np.sin(2 * np.pi * week / 52) * 0.05
        
        # Quarterly business cycle
        quarterly_cycle = np.sin(2 * np.pi * week / 13) * 0.02
        
        # Random walk component
        noise = np.random.normal(0, 0.01)
        
        return annual_cycle + quarterly_cycle + noise
    
    def _generate_competitor_action(
        self, brand: pd.Series, week: int
    ) -> bool:
        """Generate competitor actions (more likely in competitive segments)."""
        base_prob = 0.05
        
        # Higher probability in luxury segment
        if brand["market_segment"] == "luxury":
            base_prob += 0.05
        
        # Seasonal competition (higher in Q4)
        if week % 52 >= 39:  # Q4
            base_prob += 0.05
        
        return np.random.binomial(1, base_prob)
    
    def _generate_news_event(self, brand: pd.Series, week: int) -> bool:
        """Generate news events (more likely for large/innovative brands)."""
        base_prob = 0.02
        
        # Large brands get more news coverage
        if brand["brand_size"] == "large":
            base_prob += 0.03
        
        # Innovative brands generate more news
        base_prob += brand["innovation_score"] * 0.05
        
        return np.random.binomial(1, min(base_prob, 0.15))
    
    def _calculate_action_propensities(
        self,
        brand: pd.Series,
        week: int,
        market_trend: float,
        news_event: bool,
        momentum: float,
    ) -> Dict[str, float]:
        """
        Calculate propensity to take each action.
        This creates realistic selection bias in the data.
        """
        propensities = {}
        
        # Wikipedia update propensity
        wiki_base = 0.02
        
        # Large brands update more frequently
        if brand["brand_size"] == "large":
            wiki_base += 0.05
        elif brand["brand_size"] == "medium":
            wiki_base += 0.02
        
        # News events trigger updates
        if news_event:
            wiki_base += 0.3
        
        # Update more when visibility is declining
        if momentum < -0.05:
            wiki_base += 0.1
        
        propensities["wikipedia_update"] = wiki_base
        
        # YouTube content propensity
        youtube_base = 0.01
        
        # Innovative brands create more content
        youtube_base += brand["innovation_score"] * 0.05
        
        # Emerging brands rely more on content marketing
        if brand["market_segment"] == "emerging":
            youtube_base += 0.03
        
        # Positive market trends encourage content creation
        if market_trend > 0.02:
            youtube_base += 0.02
        
        propensities["youtube_content"] = youtube_base
        
        # Press release propensity
        pr_base = 0.01
        
        # News events often accompany press releases
        if news_event:
            pr_base += 0.15
        
        # Large brands issue more press releases
        if brand["brand_size"] == "large":
            pr_base += 0.02
        
        propensities["press_release"] = pr_base
        
        return propensities
    
    def _calculate_visibility(
        self,
        brand: pd.Series,
        actions: Dict[str, int],
        market_trend: float,
        competitor_action: bool,
        news_event: bool,
        momentum: float,
        recent_actions: List[Dict[str, int]],
    ) -> float:
        """
        Calculate visibility score with true causal effects.
        This is where we embed the ground truth that our causal
        inference engine will need to discover.
        """
        # Start with base visibility
        visibility = brand["base_visibility"]
        
        # Add momentum (persistence)
        visibility += momentum * 0.3
        
        # Market trend effect
        visibility += market_trend
        
        # Competitor action effect (usually negative)
        if competitor_action:
            visibility += np.random.normal(-0.03, 0.01)
        
        # News event effect (usually positive)
        if news_event:
            visibility += np.random.normal(0.05, 0.02)
        
        # TRUE CAUSAL EFFECTS OF ACTIONS
        
        # Wikipedia update effect (heterogeneous by brand size)
        if actions["wikipedia_update"]:
            effect = self.causal_effects["wikipedia_update"][brand["brand_size"]]
            noise = self.causal_effects["wikipedia_update"]["noise"]
            visibility += np.random.normal(effect, noise)
        
        # YouTube content effect (heterogeneous by innovation score)
        if actions["youtube_content"]:
            base_effect = self.causal_effects["youtube_content"]["base"]
            innovation_bonus = (
                self.causal_effects["youtube_content"]["innovation_multiplier"] 
                * brand["innovation_score"]
            )
            noise = self.causal_effects["youtube_content"]["noise"]
            visibility += np.random.normal(base_effect + innovation_bonus, noise)
        
        # Press release effect (homogeneous)
        if actions["press_release"]:
            effect = self.causal_effects["press_release"]["base"]
            noise = self.causal_effects["press_release"]["noise"]
            visibility += np.random.normal(effect, noise)
        
        # Delayed effects from recent actions (content takes time to impact)
        for i, past_actions in enumerate(recent_actions):
            decay = 0.5 ** (len(recent_actions) - i)  # Exponential decay
            
            if past_actions.get("youtube_content", 0):
                visibility += decay * 0.02  # Lingering effect
        
        # Interaction effects (multiple simultaneous actions)
        simultaneous_actions = sum(actions.values())
        if simultaneous_actions > 1:
            # Diminishing returns
            visibility -= (simultaneous_actions - 1) * 0.01
        
        # Add general noise
        visibility += np.random.normal(0, 0.01)
        
        # Ensure visibility stays in valid range
        return np.clip(visibility, 0, 1)
    
    def get_ground_truth_effects(self) -> Dict[str, Dict[str, float]]:
        """Return the true causal effects for validation."""
        return self.causal_effects.copy()
    
    def generate_complete_dataset(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Generate both brand profiles and time-series data."""
        brand_profiles = self.generate_brand_profiles()
        time_series = self.generate_time_series_data(brand_profiles)
        
        logger.info("Complete dataset generation finished")
        return brand_profiles, time_series