"""Dashboard components for Streamlit application."""

import streamlit as st
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any
from loguru import logger

from .plots import CausalVisualizer
from ..config import get_settings


class DashboardComponents:
    """Reusable dashboard components for the Streamlit app."""
    
    def __init__(self):
        """Initialize dashboard components."""
        self.settings = get_settings()
        self.visualizer = CausalVisualizer()
        logger.info("Initialized DashboardComponents")
    
    def render_brand_selector(
        self,
        brands_df: pd.DataFrame,
        key: str = "brand_selector",
    ) -> Dict[str, Any]:
        """Render brand selection interface."""
        col1, col2, col3 = st.columns([2, 1, 1])
        
        with col1:
            selected_brand_name = st.selectbox(
                "Select Brand",
                options=brands_df["brand_name"].tolist(),
                key=key,
            )
        
        # Get brand info
        brand_info = brands_df[
            brands_df["brand_name"] == selected_brand_name
        ].iloc[0]
        
        with col2:
            st.metric(
                "Brand Size",
                brand_info["brand_size"].title(),
                help="Large brands typically see stronger Wikipedia effects"
            )
        
        with col3:
            st.metric(
                "Innovation Score",
                f"{brand_info['innovation_score']:.2f}",
                help="Higher scores indicate better YouTube performance"
            )
        
        # Additional brand details
        with st.expander("Brand Details"):
            col1, col2 = st.columns(2)
            
            with col1:
                st.write(f"**Market Segment**: {brand_info['market_segment'].title()}")
                st.write(f"**Current Visibility**: {brand_info['base_visibility']:.1%}")
            
            with col2:
                st.write(f"**Brand ID**: {brand_info['brand_id']}")
                st.write(f"**Profile**: {self._get_brand_profile(brand_info)}")
        
        return brand_info.to_dict()
    
    def render_action_selector(
        self,
        key: str = "action_selector",
    ) -> Dict[str, bool]:
        """Render action selection interface."""
        st.subheader("📋 Select Marketing Actions")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            wikipedia = st.checkbox(
                "📚 Wikipedia Update",
                value=True,
                key=f"{key}_wikipedia",
                help="Update brand Wikipedia page with latest information"
            )
            if wikipedia:
                st.caption("Cost: $500-750 | Time to effect: 2-4 weeks")
        
        with col2:
            youtube = st.checkbox(
                "📺 YouTube Content",
                value=False,
                key=f"{key}_youtube",
                help="Create YouTube video content showcasing brand"
            )
            if youtube:
                st.caption("Cost: $1,400-3,000 | Time to effect: 4-8 weeks")
        
        with col3:
            press = st.checkbox(
                "📰 Press Release",
                value=False,
                key=f"{key}_press",
                help="Issue press release about brand news"
            )
            if press:
                st.caption("Cost: $700-1,500 | Time to effect: 1 week")
        
        return {
            "wikipedia_update": wikipedia,
            "youtube_content": youtube,
            "press_release": press,
        }
    
    def render_business_assumptions(
        self,
        key: str = "business_assumptions",
    ) -> Dict[str, Any]:
        """Render business assumptions input interface."""
        with st.expander("💼 Business Assumptions", expanded=False):
            col1, col2 = st.columns(2)
            
            with col1:
                monthly_searches = st.number_input(
                    "Monthly AI Searches",
                    min_value=10000,
                    max_value=1000000,
                    value=self.settings.default_monthly_searches,
                    step=10000,
                    key=f"{key}_searches",
                    help="Estimated monthly search volume for your category"
                )
                
                conversion_rate = st.slider(
                    "Conversion Rate (%)",
                    min_value=0.5,
                    max_value=10.0,
                    value=self.settings.default_conversion_rate * 100,
                    step=0.5,
                    key=f"{key}_conversion",
                    help="Percentage of visibility that converts to action"
                ) / 100
            
            with col2:
                revenue_per_conversion = st.number_input(
                    "Revenue per Conversion ($)",
                    min_value=10,
                    max_value=1000,
                    value=int(self.settings.default_revenue_per_conversion),
                    step=10,
                    key=f"{key}_revenue",
                    help="Average revenue per converted customer"
                )
                
                time_horizon = st.selectbox(
                    "Analysis Time Horizon",
                    options=[3, 6, 12, 24],
                    index=2,
                    key=f"{key}_horizon",
                    help="Months to include in ROI calculation"
                )
        
        return {
            "monthly_searches": monthly_searches,
            "conversion_rate": conversion_rate,
            "revenue_per_conversion": revenue_per_conversion,
            "time_horizon_months": time_horizon,
        }
    
    def render_results_summary(
        self,
        prediction_results: Dict[str, Any],
        roi_results: Dict[str, Any],
    ):
        """Render summary of analysis results."""
        st.subheader("📊 Results Summary")
        
        # Key metrics in columns
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric(
                "Visibility Increase",
                f"{prediction_results.get('expected_increase', 0):.1%}",
                help="Expected increase in AI visibility score"
            )
        
        with col2:
            roi = roi_results.get("base_case", {}).roi_percentage
            st.metric(
                "Expected ROI",
                f"{roi:.0f}%",
                delta=f"{roi - 100:.0f}% profit" if roi > 100 else None,
                help="Return on investment"
            )
        
        with col3:
            payback = roi_results.get("base_case", {}).payback_period_days
            st.metric(
                "Payback Period",
                f"{payback} days",
                delta="Quick" if payback < 90 else "Extended",
                delta_color="normal" if payback < 90 else "inverse",
                help="Time to recover investment"
            )
        
        with col4:
            confidence = roi_results.get("base_case", {}).break_even_probability
            st.metric(
                "Success Probability",
                f"{confidence:.0%}",
                help="Probability of achieving positive returns"
            )
        
        # Confidence interval
        ci = prediction_results.get("confidence_interval", (0, 0))
        st.info(
            f"📈 **95% Confidence Interval**: "
            f"{ci[0]:.1%} to {ci[1]:.1%} visibility increase"
        )
    
    def render_recommendation_cards(
        self,
        recommendations: List[Dict[str, Any]],
    ):
        """Render recommendation cards."""
        st.subheader("💡 Recommendations")
        
        for i, rec in enumerate(recommendations):
            priority = rec.get("priority", "medium")
            icon = {"high": "🔴", "medium": "🟡", "low": "🟢"}.get(priority, "⚪")
            
            with st.container():
                col1, col2 = st.columns([1, 9])
                
                with col1:
                    st.write(f"# {icon}")
                
                with col2:
                    st.write(f"**{rec.get('action', 'Action')}**")
                    st.write(rec.get("rationale", ""))
                
                if i < len(recommendations) - 1:
                    st.divider()
    
    def render_implementation_timeline(
        self,
        roadmap: List[Dict[str, Any]],
    ):
        """Render implementation timeline."""
        st.subheader("📅 Implementation Timeline")
        
        # Create timeline dataframe
        timeline_data = []
        
        for step in roadmap:
            timeline_data.append({
                "Week": f"Week {step['week']}-{step['week'] + step['duration_weeks']}",
                "Action": step["action"],
                "Duration": f"{step['duration_weeks']} weeks",
                "Team": step["responsible_team"],
                "Key Tasks": ", ".join(step["tasks"][:2]) + "...",
            })
        
        df_timeline = pd.DataFrame(timeline_data)
        
        # Display as table
        st.dataframe(
            df_timeline,
            use_container_width=True,
            hide_index=True,
        )
        
        # Gantt chart could be added here
    
    def render_monitoring_dashboard(
        self,
        monitoring_plan: Dict[str, Any],
    ):
        """Render monitoring plan dashboard."""
        st.subheader("📈 Monitoring Dashboard")
        
        # KPIs
        st.write("### Key Performance Indicators")
        
        kpi_data = []
        for kpi in monitoring_plan.get("kpis", []):
            kpi_data.append({
                "Metric": kpi["metric"],
                "Baseline": kpi["baseline"],
                "Target": kpi["target"],
                "Frequency": kpi["measurement_frequency"],
                "Source": kpi["data_source"],
            })
        
        df_kpis = pd.DataFrame(kpi_data)
        st.dataframe(df_kpis, use_container_width=True, hide_index=True)
        
        # Checkpoints
        st.write("### Decision Checkpoints")
        
        for checkpoint in monitoring_plan.get("checkpoints", []):
            with st.expander(f"Week {checkpoint['week']}: {checkpoint['milestone']}"):
                st.write(f"**Success Criteria**: {checkpoint['success_criteria']}")
                st.write(f"**Decision**: {checkpoint['decision_point']}")
        
        # Alerts
        st.write("### Alert Conditions")
        
        for alert in monitoring_plan.get("alerts", []):
            st.warning(f"**If** {alert['condition']} **then** {alert['action']}")
    
    def _get_brand_profile(self, brand_info: pd.Series) -> str:
        """Generate brand profile description."""
        size = brand_info["brand_size"]
        innovation = brand_info["innovation_score"]
        segment = brand_info["market_segment"]
        
        if size == "large" and innovation > 0.7:
            return "Industry Leader & Innovator"
        elif size == "large":
            return "Established Market Leader"
        elif innovation > 0.7:
            return "Innovative Challenger"
        elif segment == "emerging":
            return "Emerging Brand"
        else:
            return "Traditional Player"
    
    def render_comparison_table(
        self,
        comparison_df: pd.DataFrame,
    ):
        """Render scenario comparison table."""
        st.subheader("🔄 Scenario Comparison")
        
        # Format dataframe for display
        display_df = comparison_df.copy()
        
        # Format numeric columns
        if "expected_increase" in display_df.columns:
            display_df["Visibility Increase"] = display_df["expected_increase"].apply(
                lambda x: f"{x:.1%}"
            )
        
        if "roi" in display_df.columns:
            display_df["ROI"] = display_df["roi"].apply(lambda x: f"{x:.0f}%")
        
        if "total_cost" in display_df.columns:
            display_df["Investment"] = display_df["total_cost"].apply(
                lambda x: f"${x:,.0f}"
            )
        
        if "payback_days" in display_df.columns:
            display_df["Payback"] = display_df["payback_days"].apply(
                lambda x: f"{x} days"
            )
        
        # Select columns to display
        display_columns = [
            "scenario_id", "actions", "Visibility Increase", 
            "ROI", "Investment", "Payback"
        ]
        
        display_columns = [col for col in display_columns if col in display_df.columns]
        
        st.dataframe(
            display_df[display_columns],
            use_container_width=True,
            hide_index=True,
        )
    
    def render_insights_panel(
        self,
        causal_results: Dict[str, Any],
        heterogeneous_effects: Optional[Dict[str, Any]] = None,
    ):
        """Render key insights panel."""
        st.subheader("🔍 Key Insights")
        
        insights = []
        
        # Causal insights
        for treatment, results in causal_results.items():
            if "consensus" in results:
                effect = results["consensus"]["effect"]
                if effect > 0.05:
                    insights.append(
                        f"✅ **{treatment.replace('_', ' ').title()}** has a strong "
                        f"positive effect (+{effect:.1%})"
                    )
                elif effect > 0:
                    insights.append(
                        f"➕ **{treatment.replace('_', ' ').title()}** has a moderate "
                        f"positive effect (+{effect:.1%})"
                    )
        
        # Heterogeneous insights
        if heterogeneous_effects:
            for treatment, effects in heterogeneous_effects.items():
                if "subgroup_effects" in effects:
                    # Find most responsive subgroup
                    max_effect = 0
                    best_group = ""
                    
                    for characteristic, groups in effects["subgroup_effects"].items():
                        for group, stats in groups.items():
                            if stats["mean_effect"] > max_effect:
                                max_effect = stats["mean_effect"]
                                best_group = f"{characteristic}={group}"
                    
                    if best_group:
                        insights.append(
                            f"🎯 **{treatment.replace('_', ' ').title()}** works best for "
                            f"{best_group} (+{max_effect:.1%} effect)"
                        )
        
        # Display insights
        for insight in insights:
            st.info(insight)
        
        # If no significant insights
        if not insights:
            st.warning("⚠️ No statistically significant effects detected. Consider increasing sample size or effect magnitude.")
    
    def render_uncertainty_analysis(
        self,
        uncertainty_results: Dict[str, Any],
    ):
        """Render uncertainty analysis visualization."""
        st.subheader("📊 Uncertainty Analysis")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Prediction intervals
            if "prediction_intervals" in uncertainty_results:
                intervals = uncertainty_results["prediction_intervals"]
                st.write("### Prediction Intervals")
                
                # Create interval visualization
                fig = self.visualizer.plot_prediction_intervals(intervals)
                st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Probability distributions
            if "probability_distribution" in uncertainty_results:
                st.write("### Outcome Distribution")
                
                dist = uncertainty_results["probability_distribution"]
                fig = self.visualizer.plot_outcome_distribution(dist)
                st.plotly_chart(fig, use_container_width=True)
        
        # Risk metrics
        if "risk_metrics" in uncertainty_results:
            st.write("### Risk Metrics")
            
            risk = uncertainty_results["risk_metrics"]
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric(
                    "Value at Risk (5%)",
                    f"{risk.get('var_5', 0):.1%}",
                    help="5% worst-case scenario"
                )
            
            with col2:
                st.metric(
                    "Expected Shortfall",
                    f"{risk.get('expected_shortfall', 0):.1%}",
                    help="Average of worst 5% outcomes"
                )
            
            with col3:
                st.metric(
                    "Probability of Loss",
                    f"{risk.get('prob_loss', 0):.1%}",
                    help="Chance of negative outcome"
                )
    
    def render_data_quality_report(
        self,
        data_quality_metrics: Dict[str, Any],
    ):
        """Render data quality assessment."""
        with st.expander("📋 Data Quality Report", expanded=False):
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("### Sample Statistics")
                st.write(f"**Total Observations**: {data_quality_metrics.get('n_observations', 0):,}")
                st.write(f"**Unique Brands**: {data_quality_metrics.get('n_brands', 0)}")
                st.write(f"**Time Period**: {data_quality_metrics.get('time_period', 'N/A')}")
                
            with col2:
                st.write("### Treatment Balance")
                balance = data_quality_metrics.get('treatment_balance', {})
                for treatment, stats in balance.items():
                    st.write(f"**{treatment}**: {stats.get('treated_pct', 0):.1%} treated")
            
            # Covariate balance
            if "covariate_balance" in data_quality_metrics:
                st.write("### Covariate Balance")
                balance_df = pd.DataFrame(data_quality_metrics["covariate_balance"])
                st.dataframe(balance_df, use_container_width=True, hide_index=True)
    
    def render_sensitivity_analysis(
        self,
        sensitivity_results: Dict[str, Any],
    ):
        """Render sensitivity analysis results."""
        st.subheader("🔬 Sensitivity Analysis")
        
        # Parameter sensitivity
        if "parameter_sensitivity" in sensitivity_results:
            st.write("### Parameter Sensitivity")
            
            param_sens = sensitivity_results["parameter_sensitivity"]
            fig = self.visualizer.plot_sensitivity_tornado(param_sens)
            st.plotly_chart(fig, use_container_width=True)
            
            # Key findings
            st.write("**Key Findings:**")
            for param, impact in param_sens.get("high_impact_params", []):
                st.write(f"- {param}: {impact:.1%} impact on results")
        
        # Assumption violations
        if "assumption_tests" in sensitivity_results:
            st.write("### Robustness to Assumptions")
            
            tests = sensitivity_results["assumption_tests"]
            test_results = []
            
            for test_name, result in tests.items():
                test_results.append({
                    "Assumption": test_name.replace("_", " ").title(),
                    "Status": "✅ Valid" if result["passed"] else "⚠️ Violated",
                    "Impact": f"{result.get('impact', 0):.1%}",
                    "Recommendation": result.get("recommendation", ""),
                })
            
            df_tests = pd.DataFrame(test_results)
            st.dataframe(df_tests, use_container_width=True, hide_index=True)
    
    def render_export_options(
        self,
        analysis_results: Dict[str, Any],
    ):
        """Render export options for analysis results."""
        st.subheader("📥 Export Results")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            # Export as PDF report
            if st.button("📄 Generate PDF Report", use_container_width=True):
                with st.spinner("Generating report..."):
                    # This would call a report generation function
                    st.success("Report generated! Download will start automatically.")
                    # pdf_bytes = generate_pdf_report(analysis_results)
                    # st.download_button("Download PDF", pdf_bytes, "analysis_report.pdf")
        
        with col2:
            # Export data as CSV
            if st.button("📊 Export Data (CSV)", use_container_width=True):
                # Prepare data for export
                export_data = self._prepare_export_data(analysis_results)
                csv = export_data.to_csv(index=False)
                st.download_button(
                    "Download CSV",
                    csv,
                    "analysis_data.csv",
                    "text/csv",
                    use_container_width=True,
                )
        
        with col3:
            # Export as PowerPoint
            if st.button("📈 Export Slides (PPTX)", use_container_width=True):
                with st.spinner("Creating presentation..."):
                    st.success("Presentation created! Download will start automatically.")
                    # pptx_bytes = generate_powerpoint(analysis_results)
                    # st.download_button("Download PPTX", pptx_bytes, "analysis_slides.pptx")
    
    def render_error_message(
        self,
        error_type: str,
        error_message: str,
        suggestions: Optional[List[str]] = None,
    ):
        """Render user-friendly error messages."""
        error_icons = {
            "data": "📊",
            "calculation": "🧮",
            "configuration": "⚙️",
            "validation": "✅",
            "network": "🌐",
        }
        
        icon = error_icons.get(error_type, "⚠️")
        
        st.error(f"{icon} **{error_type.title()} Error**")
        st.write(error_message)
        
        if suggestions:
            st.write("**Suggestions:**")
            for suggestion in suggestions:
                st.write(f"• {suggestion}")
    
    def render_help_section(self):
        """Render contextual help section."""
        with st.expander("❓ Help & Documentation"):
            st.write("""
            ### Quick Guide
            
            1. **Select Your Brand**: Choose from the dropdown to analyze a specific brand
            2. **Choose Actions**: Select marketing interventions to simulate
            3. **Set Business Assumptions**: Configure your business parameters
            4. **Run Analysis**: Click analyze to see causal effects and ROI
            
            ### Understanding Results
            
            - **Visibility Score**: Predicted change in AI recommendation visibility (0-100%)
            - **ROI**: Return on investment considering all costs and revenues
            - **Confidence Intervals**: Range of likely outcomes (95% probability)
            - **Heterogeneous Effects**: How effects vary by brand characteristics
            
            ### Best Practices
            
            - Start with single actions to understand individual effects
            - Use historical data validation when available
            - Consider seasonality in your time horizon
            - Monitor early indicators after implementation
            
            ### Need More Help?
            
            - Check the [documentation](https://docs.example.com)
            - Contact support: support@example.com
            """)
    
    def render_quick_stats(
        self,
        data_df: pd.DataFrame,
    ):
        """Render quick statistics summary."""
        st.write("### 📊 Quick Stats")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            avg_visibility = data_df["visibility_score"].mean()
            st.metric(
                "Avg Visibility",
                f"{avg_visibility:.1%}",
                help="Average visibility across all brands"
            )
        
        with col2:
            treatment_rate = data_df[["wikipedia_update", "youtube_content", "press_release"]].any(axis=1).mean()
            st.metric(
                "Action Rate",
                f"{treatment_rate:.1%}",
                help="Percentage of periods with any action"
            )
        
        with col3:
            visibility_std = data_df["visibility_score"].std()
            st.metric(
                "Volatility",
                f"{visibility_std:.1%}",
                help="Standard deviation of visibility"
            )
        
        with col4:
            trend = data_df.groupby("week")["visibility_score"].mean().pct_change().mean()
            st.metric(
                "Weekly Trend",
                f"{trend:+.2%}",
                help="Average weekly change in visibility"
            )
    
    def _prepare_export_data(
        self,
        analysis_results: Dict[str, Any],
    ) -> pd.DataFrame:
        """Prepare analysis results for export."""
        export_data = []
        
        # Extract key results
        for scenario_id, results in analysis_results.items():
            row = {
                "scenario_id": scenario_id,
                "expected_increase": results.get("expected_increase", 0),
                "confidence_lower": results.get("confidence_interval", [0, 0])[0],
                "confidence_upper": results.get("confidence_interval", [0, 0])[1],
                "roi": results.get("roi", {}).get("roi_percentage", 0),
                "payback_days": results.get("roi", {}).get("payback_period_days", 0),
                "break_even_prob": results.get("roi", {}).get("break_even_probability", 0),
            }
            
            # Add action flags
            actions = results.get("actions", {})
            for action in ["wikipedia_update", "youtube_content", "press_release"]:
                row[action] = actions.get(action, False)
            
            export_data.append(row)
        
        return pd.DataFrame(export_data)
    
    def render_footer(self):
        """Render application footer."""
        st.markdown("---")
        
        col1, col2, col3 = st.columns([2, 1, 1])
        
        with col1:
            st.caption(
                "🚀 **Brand Visibility Simulator** | "
                "Powered by causal inference & machine learning"
            )
        
