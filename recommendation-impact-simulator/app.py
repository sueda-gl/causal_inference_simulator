"""
Recommendation Impact Simulator - Main Streamlit Application
==========================================================

A comprehensive causal inference engine for analyzing the impact of 
marketing actions on AI visibility and brand recommendations.
"""

import streamlit as st
import pandas as pd
import numpy as np
from typing import Dict, Any, Optional
import time
from datetime import datetime, timedelta

# Configure Streamlit page
st.set_page_config(
    page_title="Recommendation Impact Simulator",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Import our modules
import sys
import os
from pathlib import Path

# Add the current directory to the path to ensure imports work properly
current_dir = Path(__file__).parent.absolute()
sys.path.insert(0, str(current_dir))

# Now import the modules
from src.data.generator import CausalDataGenerator
from src.causal.engine import CausalInferenceEngine
from src.causal.effects import HeterogeneousEffectsAnalyzer
from src.business.roi_calculator import ROICalculator
from src.business.recommender import ActionRecommender
from src.visualization.dashboards import DashboardComponents
from src.visualization.plots import CausalVisualizer
from src.utils.logger import setup_logging, log_duration, log_business_metric
from src.config.settings import get_settings

# Initialize components
@st.cache_resource
def initialize_components():
    """Initialize all application components."""
    setup_logging(enable_file_logging=True, enable_json_logging=False)
    
    components = {
        'data_generator': CausalDataGenerator(),
        'causal_engine': CausalInferenceEngine(),
        'effects_analyzer': HeterogeneousEffectsAnalyzer(),
        'roi_calculator': ROICalculator(),
        # 'recommender': ActionRecommender(),  # This needs causal_results, will create later
        'dashboard': DashboardComponents(),
        'visualizer': CausalVisualizer(),
        # 'predictor': WhatIfPredictor(),  # This needs causal_results, will create later
        'settings': get_settings()
    }
    
    return components

# Generate or load data
@st.cache_data
def generate_synthetic_data(n_brands: int = 50, n_time_periods: int = 52, random_seed: int = 42):
    """Generate synthetic dataset with embedded causal relationships."""
    generator = CausalDataGenerator(
        n_brands=n_brands,
        n_time_periods=n_time_periods,
        random_seed=random_seed
    )
    
    with st.spinner("Generating synthetic dataset with embedded causal relationships..."):
        brand_profiles, time_series_data = generator.generate_complete_dataset()
        ground_truth = generator.get_ground_truth_effects()
    
    return brand_profiles, time_series_data, ground_truth

def main():
    """Main application logic."""
    
    # Initialize components
    components = initialize_components()
    settings = components['settings']
    
    # App header
    st.title("🎯 Recommendation Impact Simulator")
    st.markdown("""
    **Causal Inference Engine for AI Visibility Analysis**
    
    Discover the true causal impact of marketing actions on brand visibility in AI recommendations.
    This simulator uses advanced causal inference techniques to separate correlation from causation.
    """)
    
    # Sidebar configuration
    st.sidebar.title("🔧 Configuration")
    
    # Data generation settings
    st.sidebar.subheader("📊 Data Settings")
    
    n_brands = st.sidebar.slider(
        "Number of Brands",
        min_value=10,
        max_value=100,
        value=settings.n_brands,
        help="More brands = more statistical power but slower analysis"
    )
    
    n_weeks = st.sidebar.slider(
        "Time Period (Weeks)",
        min_value=20,
        max_value=104,
        value=settings.n_time_periods,
        help="Longer periods provide more reliable causal estimates"
    )
    
    random_seed = st.sidebar.number_input(
        "Random Seed",
        value=settings.random_seed,
        help="For reproducible results"
    )
    
    # Analysis settings
    st.sidebar.subheader("⚙️ Analysis Settings")
    
    confidence_level = st.sidebar.slider(
        "Confidence Level",
        min_value=0.80,
        max_value=0.99,
        value=settings.confidence_level,
        step=0.01,
        format="%.2f"
    )
    
    bootstrap_iterations = st.sidebar.slider(
        "Bootstrap Iterations",
        min_value=50,
        max_value=500,
        value=settings.bootstrap_iterations,
        help="More iterations = more accurate confidence intervals"
    )
    
    # Generate data
    brand_profiles, time_series_data, ground_truth = generate_synthetic_data(
        n_brands=n_brands,
        n_time_periods=n_weeks,
        random_seed=random_seed
    )
    
    # Display data quality metrics
    components['dashboard'].render_data_quality_report({
        'n_observations': len(time_series_data),
        'n_brands': len(brand_profiles),
        'time_period': f"{n_weeks} weeks",
        'treatment_balance': {
            'wikipedia_update': {
                'treated_pct': time_series_data['wikipedia_update'].mean()
            },
            'youtube_content': {
                'treated_pct': time_series_data['youtube_content'].mean()
            },
            'press_release': {
                'treated_pct': time_series_data['press_release'].mean()
            }
        }
    })
    
    # Quick stats
    components['dashboard'].render_quick_stats(time_series_data)
    
    # Main tabs
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "🔍 What-If Analysis",
        "📈 Causal Discovery", 
        "🎯 Heterogeneous Effects",
        "💰 Business Impact",
        "📋 Action Recommendations"
    ])
    
    with tab1:
        st.header("🔍 What-If Analysis")
        st.markdown("Predict the impact of marketing actions for specific brands")
        
        # Brand selection
        selected_brand = components['dashboard'].render_brand_selector(
            brand_profiles, 
            key="whatif_brand"
        )
        
        # Action selection
        selected_actions = components['dashboard'].render_action_selector(
            key="whatif_actions"
        )
        
        # Business assumptions
        business_params = components['dashboard'].render_business_assumptions(
            key="whatif_business"
        )
        
        # Action costs
        st.subheader("💰 Action Costs")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            wikipedia_cost = st.number_input(
                "Wikipedia Update Cost ($)",
                min_value=100,
                max_value=5000,
                value=750,
                step=50,
                key="whatif_wiki_cost"
            )
        
        with col2:
            youtube_cost = st.number_input(
                "YouTube Content Cost ($)",
                min_value=500,
                max_value=10000,
                value=2500,
                step=100,
                key="whatif_youtube_cost"
            )
        
        with col3:
            press_release_cost = st.number_input(
                "Press Release Cost ($)",
                min_value=200,
                max_value=3000,
                value=1000,
                step=50,
                key="whatif_pr_cost"
            )
        
        if st.button("🚀 Run What-If Analysis", type="primary"):
            with st.spinner("Running causal analysis..."):
                
                # Filter data for selected brand
                brand_data = time_series_data[
                    time_series_data['brand_name'] == selected_brand['brand_name']
                ].copy()
                
                # Run causal inference for each action
                causal_results = {}
                for action in ['wikipedia_update', 'youtube_content', 'press_release']:
                    if selected_actions[action]:
                        with log_duration(f"causal_analysis_{action}"):
                            result = components['causal_engine'].estimate_causal_effect(
                                data=time_series_data,
                                treatment=action,
                                outcome='visibility_score'
                            )
                            causal_results[action] = result
                
                # Predict what-if scenario
                if causal_results:
                    # Create predictor with causal results
                    from src.prediction.predictor import WhatIfPredictor
                    from src.data.schemas import BrandProfile, ActionSet
                    
                    predictor = WhatIfPredictor(causal_results)
                    
                    # Convert selected_brand dict to BrandProfile
                    brand_profile = BrandProfile(**selected_brand)
                    
                    # Convert selected_actions dict to ActionSet
                    action_set = ActionSet(**selected_actions)
                    
                    prediction = predictor.predict_intervention_impact(
                        brand_profile=brand_profile,
                        planned_actions=action_set,
                        time_horizon=12
                    )
                    
                    # Calculate ROI
                    # Extract visibility increase and confidence interval from prediction
                    visibility_increase = prediction.get('expected_increase', 0)
                    ci_tuple = prediction.get('confidence_interval', (0, visibility_increase))
                    visibility_ci = (
                        ci_tuple[0] if isinstance(ci_tuple, tuple) else 0,
                        ci_tuple[1] if isinstance(ci_tuple, tuple) else visibility_increase
                    )
                    
                    # Define action costs
                    action_costs = {}
                    if selected_actions.get('wikipedia_update'):
                        action_costs['wikipedia_update'] = wikipedia_cost
                    if selected_actions.get('youtube_content'):
                        action_costs['youtube_content'] = youtube_cost
                    if selected_actions.get('press_release'):
                        action_costs['press_release'] = press_release_cost
                    
                    roi_results = components['roi_calculator'].calculate_roi(
                        visibility_increase=visibility_increase,
                        visibility_confidence_interval=visibility_ci,
                        action_costs=action_costs,
                        business_assumptions=business_params
                    )
                    
                    # Display results
                    components['dashboard'].render_results_summary(
                        prediction_results=prediction,
                        roi_results=roi_results
                    )
                    
                    # Log business metrics
                    if roi_results.get('base_case'):
                        log_business_metric(
                            "predicted_roi", 
                            roi_results['base_case'].roi_percentage,
                            brand=selected_brand['brand_name'],
                            actions=list(selected_actions.keys())
                        )
    
    with tab2:
        st.header("📈 Causal Discovery")
        st.markdown("Discover causal relationships using multiple estimation methods")
        
        col1, col2 = st.columns([1, 2])
        
        with col1:
            st.subheader("🎛️ Analysis Controls")
            
            treatments_to_analyze = st.multiselect(
                "Select Treatments to Analyze",
                options=['wikipedia_update', 'youtube_content', 'press_release'],
                default=['wikipedia_update'],
                help="Choose which marketing actions to analyze"
            )
            
            outcome_variable = st.selectbox(
                "Outcome Variable",
                options=['visibility_score'],
                help="The outcome we want to understand"
            )
            
            if st.button("🔬 Run Causal Analysis", type="primary"):
                causal_results = {}
                
                for treatment in treatments_to_analyze:
                    with st.spinner(f"Analyzing {treatment}..."):
                        with log_duration(f"causal_discovery_{treatment}"):
                            result = components['causal_engine'].estimate_causal_effect(
                                data=time_series_data,
                                treatment=treatment,
                                outcome=outcome_variable
                            )
                            causal_results[treatment] = result
                
                # Store results in session state
                st.session_state['causal_results'] = causal_results
        
        with col2:
            st.subheader("📊 Results")
            
            if 'causal_results' in st.session_state:
                results = st.session_state['causal_results']
                
                # Display insights
                components['dashboard'].render_insights_panel(results)
                
                # Display detailed results for each treatment
                for treatment, result in results.items():
                    with st.expander(f"📋 Detailed Results: {treatment.replace('_', ' ').title()}"):
                        
                        # Consensus estimate
                        if 'consensus' in result:
                            consensus = result['consensus']
                            st.metric(
                                "Consensus Effect",
                                f"{consensus['effect']:.1%}",
                                help="Average across all estimation methods"
                            )
                        
                        # Method comparison
                        if 'estimates' in result:
                            estimates_df = pd.DataFrame([
                                {
                                    'Method': est.method,
                                    'Effect': f"{est.effect:.1%}",
                                    'CI Lower': f"{est.confidence_interval[0]:.1%}",
                                    'CI Upper': f"{est.confidence_interval[1]:.1%}",
                                    'P-value': f"{est.p_value:.3f}",
                                    'N Treated': est.n_treated,
                                    'N Control': est.n_control
                                }
                                for est in result['estimates'].values()
                                if hasattr(est, 'method')
                            ])
                            
                            if not estimates_df.empty:
                                st.dataframe(estimates_df, use_container_width=True)
                        
                        # Comparison with ground truth
                        if treatment in ground_truth:
                            gt_effect = ground_truth[treatment]
                            estimated_effect = result.get('consensus', {}).get('effect', 0)
                            
                            col_a, col_b, col_c = st.columns(3)
                            with col_a:
                                st.metric("Ground Truth", f"{gt_effect.get('large', 0):.1%}")
                            with col_b:
                                st.metric("Estimated", f"{estimated_effect:.1%}")
                            with col_c:
                                bias = abs(estimated_effect - gt_effect.get('large', 0))
                                st.metric("Estimation Bias", f"{bias:.1%}")
    
    with tab3:
        st.header("🎯 Heterogeneous Effects Analysis")
        st.markdown("Understand how effects vary across different brand characteristics")
        
        if 'causal_results' in st.session_state:
            results = st.session_state['causal_results']
            
            # Effect modifiers selection
            effect_modifiers = st.multiselect(
                "Select Brand Characteristics to Analyze",
                options=['brand_size', 'innovation_score', 'market_segment'],
                default=['brand_size', 'market_segment'],
                help="How do effects vary by these characteristics?"
            )
            
            if effect_modifiers and st.button("🔍 Analyze Heterogeneous Effects"):
                
                heterogeneous_results = {}
                
                for treatment in results.keys():
                    with st.spinner(f"Analyzing heterogeneous effects for {treatment}..."):
                        het_result = components['effects_analyzer'].analyze_heterogeneous_effects(
                            data=time_series_data,
                            treatment=treatment,
                            outcome='visibility_score',
                            effect_modifiers=effect_modifiers
                        )
                        heterogeneous_results[treatment] = het_result
                
                # Display heterogeneous effects insights
                components['dashboard'].render_insights_panel(
                    results, 
                    heterogeneous_effects=heterogeneous_results
                )
                
                # Detailed heterogeneous effects display
                for treatment, het_result in heterogeneous_results.items():
                    with st.expander(f"📊 {treatment.replace('_', ' ').title()} - Subgroup Analysis"):
                        
                        if 'subgroup_effects' in het_result:
                            for characteristic, groups in het_result['subgroup_effects'].items():
                                st.write(f"**By {characteristic.replace('_', ' ').title()}:**")
                                
                                subgroup_df = pd.DataFrame([
                                    {
                                        'Group': group,
                                        'Effect': f"{stats['mean_effect']:.1%}",
                                        'CI Lower': f"{stats['mean_effect'] - 1.96 * stats.get('std_effect', 0) / np.sqrt(stats.get('n_observations', stats.get('n_obs', 1))):.1%}",
                                        'CI Upper': f"{stats['mean_effect'] + 1.96 * stats.get('std_effect', 0) / np.sqrt(stats.get('n_observations', stats.get('n_obs', 1))):.1%}",
                                        'Sample Size': stats.get('n_observations', stats.get('n_obs', 0))
                                    }
                                    for group, stats in groups.items()
                                ])
                                
                                st.dataframe(subgroup_df, use_container_width=True)
        else:
            st.info("👈 Please run Causal Discovery analysis first to enable heterogeneous effects analysis.")
    
    with tab4:
        st.header("💰 Business Impact Analysis")
        st.markdown("Translate causal effects into business metrics and ROI")
        
        if 'causal_results' in st.session_state:
            results = st.session_state['causal_results']
            
            # Business scenario builder
            st.subheader("📈 Scenario Builder")
            
            col1, col2 = st.columns(2)
            
            with col1:
                # Monthly volume assumptions
                monthly_searches = st.number_input(
                    "Monthly AI Searches",
                    min_value=1000,
                    max_value=10000000,
                    value=100000,
                    step=10000,
                    help="Total monthly search volume in your category"
                )
                
                conversion_rate = st.slider(
                    "Visibility to Action Rate (%)",
                    min_value=0.1,
                    max_value=10.0,
                    value=2.0,
                    step=0.1,
                    format="%.1f%%"
                ) / 100
                
                revenue_per_action = st.number_input(
                    "Revenue per Action ($)",
                    min_value=1,
                    max_value=1000,
                    value=50,
                    step=5,
                    help="Average revenue when someone takes action after seeing your brand"
                )
            
            with col2:
                # Cost assumptions
                wikipedia_cost = st.number_input(
                    "Wikipedia Update Cost ($)",
                    min_value=100,
                    max_value=5000,
                    value=750,
                    step=50
                )
                
                youtube_cost = st.number_input(
                    "YouTube Content Cost ($)",
                    min_value=500,
                    max_value=10000,
                    value=2500,
                    step=100
                )
                
                press_release_cost = st.number_input(
                    "Press Release Cost ($)",
                    min_value=200,
                    max_value=3000,
                    value=1000,
                    step=50
                )
            
            # Generate scenarios
            if st.button("💹 Calculate Business Impact", type="primary"):
                
                # Define cost mapping
                action_costs = {
                    'wikipedia_update': wikipedia_cost,
                    'youtube_content': youtube_cost,
                    'press_release': press_release_cost
                }
                
                # Calculate business impact for each treatment
                business_results = {}
                
                for treatment, causal_result in results.items():
                    effect = causal_result.get('consensus', {}).get('effect', 0)
                    
                    # Calculate monthly impact
                    visibility_increase = effect
                    additional_monthly_revenue = (
                        monthly_searches * 
                        visibility_increase * 
                        conversion_rate * 
                        revenue_per_action
                    )
                    
                    cost = action_costs.get(treatment, 0)
                    monthly_profit = additional_monthly_revenue - cost
                    roi = (monthly_profit / cost * 100) if cost > 0 else 0
                    payback_days = (cost / (additional_monthly_revenue / 30)) if additional_monthly_revenue > 0 else float('inf')
                    
                    business_results[treatment] = {
                        'treatment': treatment,
                        'effect': effect,
                        'monthly_revenue': additional_monthly_revenue,
                        'cost': cost,
                        'monthly_profit': monthly_profit,
                        'roi_percentage': roi,
                        'payback_days': min(payback_days, 365)  # Cap at 1 year
                    }
                
                # Display business results
                st.subheader("📊 Business Impact Summary")
                
                business_df = pd.DataFrame([
                    {
                        'Action': result['treatment'].replace('_', ' ').title(),
                        'Visibility Effect': f"{result['effect']:.1%}",
                        'Monthly Revenue': f"${result['monthly_revenue']:,.0f}",
                        'Investment': f"${result['cost']:,.0f}",
                        'Monthly Profit': f"${result['monthly_profit']:,.0f}",
                        'ROI': f"{result['roi_percentage']:.0f}%",
                        'Payback (Days)': f"{result['payback_days']:.0f}"
                    }
                    for result in business_results.values()
                ])
                
                st.dataframe(business_df, use_container_width=True)
                
                # ROI visualization
                # Create ROI analysis structure for visualization
                roi_analysis_for_viz = {
                    "base_case": type('obj', (object,), {
                        'roi_percentage': business_results[list(business_results.keys())[0]]['roi_percentage'],
                        'break_even_probability': 0.8
                    })(),
                    "scenarios": {
                        "optimistic": type('obj', (object,), {
                            'roi_percentage': max(r['roi_percentage'] for r in business_results.values()),
                            'break_even_probability': 0.9
                        })(),
                        "pessimistic": type('obj', (object,), {
                            'roi_percentage': min(r['roi_percentage'] for r in business_results.values()),
                            'break_even_probability': 0.6
                        })(),
                        "with_competition": type('obj', (object,), {
                            'roi_percentage': sum(r['roi_percentage'] for r in business_results.values()) / len(business_results) * 0.7,
                            'break_even_probability': 0.7
                        })()
                    },
                    "sensitivity": {},
                    "time_metrics": {}
                }
                
                fig = components['visualizer'].plot_roi_scenarios(roi_analysis_for_viz)
                st.plotly_chart(fig, use_container_width=True)
                
                # Store business results
                st.session_state['business_results'] = business_results
        else:
            st.info("👈 Please run Causal Discovery analysis first to enable business impact analysis.")
    
    with tab5:
        st.header("📋 Action Recommendations")
        st.markdown("AI-powered recommendations for optimal marketing strategy")
        
        if 'business_results' in st.session_state and 'causal_results' in st.session_state:
            business_results = st.session_state['business_results']
            causal_results = st.session_state['causal_results']
            
            # Create recommender with causal results
            recommender = ActionRecommender(causal_results)
            
            # Brand selection for recommendations
            selected_brand = components['dashboard'].render_brand_selector(
                brand_profiles, 
                key="rec_brand"
            )
            
            # Budget constraint
            budget_constraint = st.number_input(
                "Monthly Budget ($)",
                min_value=500,
                max_value=50000,
                value=5000,
                step=500
            )
            
            # Generate recommendations for selected brand
            from src.data.schemas import BrandProfile
            brand_profile = BrandProfile(**selected_brand)
            
            constraints = {
                "max_budget": budget_constraint,
                "max_actions": 3,
                "time_horizon": 12,
                "min_roi": 50,
            }
            
            if st.button("🚀 Generate Recommendations", type="primary"):
                with st.spinner("Analyzing optimal actions..."):
                    recommendations = recommender.recommend_actions(
                        brand_profile=brand_profile,
                        constraints=constraints,
                        objective="maximize_roi"
                    )
                    
                    # Store in session state
                    st.session_state['recommendations'] = recommendations
            
            # Display recommendations if available
            if 'recommendations' in st.session_state:
                recommendations = st.session_state['recommendations']
                
                # Display recommendations
                if recommendations.get('status') == 'success':
                    # Create formatted recommendation cards from the single recommendation
                    rec_cards = []
                    
                    # Main recommendation card
                    rec_cards.append({
                        "priority": "high",
                        "action": "Recommended Action Mix",
                        "rationale": f"Investment: ${recommendations['investment_required']:,.0f} | "
                                   f"Expected ROI: {recommendations['expected_outcomes']['roi']} | "
                                   f"Payback: {recommendations['expected_outcomes']['payback_period']}"
                    })
                    
                    # Individual action cards
                    for action in recommendations.get('recommended_actions', []):
                        rec_cards.append({
                            "priority": "medium",
                            "action": action.replace('_', ' ').title(),
                            "rationale": next((r for r in recommendations.get('rationale', []) 
                                             if action.replace('_', ' ').lower() in r.lower()), 
                                            f"Implement {action.replace('_', ' ')} strategy")
                        })
                    
                    components['dashboard'].render_recommendation_cards(rec_cards)
                    
                    # Implementation timeline
                    roadmap = recommendations.get('implementation_roadmap', [])
                    if roadmap:
                        components['dashboard'].render_implementation_timeline(roadmap)
                    
                    # Monitoring plan
                    monitoring_plan = recommendations.get('monitoring_plan', {})
                    if monitoring_plan:
                        components['dashboard'].render_monitoring_dashboard(monitoring_plan)
                else:
                    st.warning(f"No recommendations available: {recommendations.get('reason', 'Unknown error')}")
        else:
            st.info("👈 Please run Business Impact analysis first to get recommendations.")
    
    # Footer
    components['dashboard'].render_footer()
    
    # Help section
    components['dashboard'].render_help_section()

if __name__ == "__main__":
    main()
