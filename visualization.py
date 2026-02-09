"""
Interactive visualizations for A/B test results
"""

import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from typing import Dict, List, Optional

class ABVisualization:
    def __init__(self):
        self.color_scheme = {
            'control': '#1f77b4',
            'treatment': '#ff7f0e',
            'variant_a': '#2ca02c',
            'variant_b': '#d62728',
            'background': '#f8f9fa',
            'grid': '#e9ecef'
        }
    
    def create_dashboard(self, 
                        raw_data: pd.DataFrame,
                        time_series: pd.DataFrame,
                        analysis: Dict) -> go.Figure:
        """
        Create comprehensive A/B test dashboard
        """
        # ROBUSTNESS CHECK: Ensure time_series has required columns
        ts_data = time_series.copy()
        if 'conversion_rate' not in ts_data.columns and 'converted' in ts_data.columns:
            # If raw data was passed by mistake, or rate wasn't calculated
             if 'user_id' in ts_data.columns:
                 # Check if user_id is a count (aggregated) or IDs (raw)
                 if pd.api.types.is_numeric_dtype(ts_data['user_id']):
                     ts_data['conversion_rate'] = ts_data['converted'] / ts_data['user_id'].replace(0, 1)
                 else:
                     # It's raw data, create a simple aggregation
                     ts_data['timestamp_hour'] = ts_data['timestamp'].dt.floor('H')
                     ts_agg = ts_data.groupby(['variant', 'timestamp_hour']).agg({
                         'converted': 'mean',
                         'timestamp': 'first' # Keep timestamp
                     }).reset_index()
                     ts_data = ts_agg.rename(columns={'converted': 'conversion_rate'})

        fig = make_subplots(
            rows=3, cols=3,
            subplot_titles=(
                'Conversion Rate Over Time',
                'Cumulative Conversions',
                'Distribution of Session Metrics',
                'Segmentation Analysis',
                'Statistical Significance',
                'Revenue Impact',
                'Funnel Visualization',
                'Confidence Intervals',
                'Success Probability'
            ),
            specs=[
                [{'type': 'scatter'}, {'type': 'scatter'}, {'type': 'box'}],
                [{'type': 'bar'}, {'type': 'indicator'}, {'type': 'bar'}],
                [{'type': 'funnel'}, {'type': 'bar'}, {'type': 'indicator'}]
            ],
            vertical_spacing=0.12,
            horizontal_spacing=0.1
        )
        
        # 1. Conversion Rate Over Time
        if 'conversion_rate' in ts_data.columns:
            for variant in ts_data['variant'].unique():
                variant_ts = ts_data[ts_data['variant'] == variant]
                fig.add_trace(
                    go.Scatter(
                        x=variant_ts['timestamp'],
                        y=variant_ts['conversion_rate'],
                        mode='lines+markers',
                        name=variant,
                        line=dict(width=2)
                    ),
                    row=1, col=1
                )
        else:
             fig.add_annotation(text="Data unavailable", row=1, col=1)
        
        # 2. Cumulative Conversions
        for variant in raw_data['variant'].unique():
            variant_data = raw_data[raw_data['variant'] == variant].sort_values('timestamp')
            variant_data['cumulative_conversions'] = variant_data['converted'].cumsum()
            variant_data['cumulative_users'] = range(1, len(variant_data) + 1)
            
            fig.add_trace(
                go.Scatter(
                    x=variant_data['cumulative_users'],
                    y=variant_data['cumulative_conversions'],
                    mode='lines',
                    name=f'{variant} (cumulative)',
                    line=dict(width=2, dash='dash')
                ),
                row=1, col=2
            )
        
        # 3. Distribution of Session Metrics
        metrics = ['session_duration', 'page_views', 'clicks']
        # Use first metric found in data
        available_metric = next((m for m in metrics if m in raw_data.columns), None)
        
        if available_metric:
            for variant in raw_data['variant'].unique():
                variant_data = raw_data[raw_data['variant'] == variant]
                fig.add_trace(
                    go.Box(
                        y=variant_data[available_metric],
                        name=f'{variant} - {available_metric}',
                        boxpoints='outliers',
                        marker_color=self.color_scheme.get(variant.lower(), '#333')
                    ),
                    row=1, col=3
                )
        
        # 4. Segmentation Analysis
        if 'segment' in raw_data.columns:
            segment_analysis = raw_data.groupby(['variant', 'segment']).agg({
                'converted': ['mean', 'count']
            }).reset_index()
            
            segment_analysis.columns = ['variant', 'segment', 'conversion_rate', 'count']
            
            for variant in segment_analysis['variant'].unique():
                variant_segments = segment_analysis[segment_analysis['variant'] == variant]
                fig.add_trace(
                    go.Bar(
                        x=variant_segments['segment'],
                        y=variant_segments['conversion_rate'],
                        name=variant,
                        text=variant_segments['conversion_rate'].apply(lambda x: f'{x:.2%}'),
                        textposition='auto'
                    ),
                    row=2, col=1
                )
        
        # 5. Statistical Significance Indicator
        p_value = analysis.get('p_value', 0.05)
        fig.add_trace(
            go.Indicator(
                mode="gauge+number+delta",
                value=1 - p_value,
                title={'text': f"Confidence\n(p={p_value:.4f})"},
                domain={'row': 1, 'col': 1},
                gauge={
                    'axis': {'range': [0.9, 1]},
                    'bar': {'color': "green" if p_value < 0.05 else "red"},
                    'steps': [
                        {'range': [0.9, 0.95], 'color': "lightgray"},
                        {'range': [0.95, 1], 'color': "gray"}
                    ],
                    'threshold': {
                        'line': {'color': "red", 'width': 4},
                        'thickness': 0.75,
                        'value': 0.95
                    }
                }
            ),
            row=2, col=2
        )
        
        # 6. Revenue Impact
        if 'conversion_value' in raw_data.columns:
            revenue_by_variant = raw_data.groupby('variant')['conversion_value'].sum().reset_index()
            fig.add_trace(
                go.Bar(
                    x=revenue_by_variant['variant'],
                    y=revenue_by_variant['conversion_value'],
                    text=revenue_by_variant['conversion_value'].apply(lambda x: f'${x:,.0f}'),
                    textposition='auto',
                    marker_color=[self.color_scheme.get(v.lower(), '#333') 
                                for v in revenue_by_variant['variant']]
                ),
                row=2, col=3
            )
        
        # 7. Funnel Visualization
        if all(col in raw_data.columns for col in ['page_views', 'clicks']):
            funnel_data = raw_data.groupby('variant').agg({
                'user_id': 'count',
                'converted': 'sum',
                'page_views': 'mean',
                'clicks': 'mean'
            }).reset_index()
            
            for i, variant in enumerate(funnel_data['variant'].unique()):
                variant_data = funnel_data[funnel_data['variant'] == variant].iloc[0]
                fig.add_trace(
                    go.Funnel(
                        name=variant,
                        y=['Users', 'Page Views', 'Clicks', 'Conversions'],
                        x=[
                            variant_data['user_id'],
                            variant_data['user_id'] * variant_data['page_views'],
                            variant_data['user_id'] * variant_data['clicks'],
                            variant_data['converted']
                        ],
                        textinfo="value+percent initial"
                    ),
                    row=3, col=1
                )
        
        # 8. Confidence Intervals
        ci_data = self._calculate_confidence_intervals(raw_data)
        if not ci_data.empty:
            fig.add_trace(
                go.Bar(
                    x=ci_data['variant'],
                    y=ci_data['conversion_rate'],
                    error_y=dict(
                        type='data',
                        array=ci_data['ci_upper'] - ci_data['conversion_rate'],
                        arrayminus=ci_data['conversion_rate'] - ci_data['ci_lower'],
                        visible=True
                    ),
                    text=ci_data['conversion_rate'].apply(lambda x: f'{x:.2%}'),
                    textposition='auto'
                ),
                row=3, col=2
            )
        
        # 9. Success Probability
        uplift = analysis.get('uplift_percentage', 0)
        success_prob = min(100, max(0, 80 + (uplift * 10)))  # Simplified model
        
        fig.add_trace(
            go.Indicator(
                mode="gauge+number",
                value=success_prob,
                title={'text': "Success Probability"},
                gauge={
                    'axis': {'range': [0, 100]},
                    'bar': {'color': "darkblue"},
                    'steps': [
                        {'range': [0, 50], 'color': "red"},
                        {'range': [50, 75], 'color': "yellow"},
                        {'range': [75, 100], 'color': "green"}
                    ],
                    'threshold': {
                        'line': {'color': "black", 'width': 4},
                        'thickness': 0.75,
                        'value': 80
                    }
                }
            ),
            row=3, col=3
        )
        
        # Update layout
        fig.update_layout(
            height=1200,
            showlegend=True,
            plot_bgcolor=self.color_scheme['background'],
            paper_bgcolor=self.color_scheme['background'],
            title_text="A/B Test Results Dashboard",
            title_x=0.5,
            title_font_size=20
        )
        
        return fig
    
    def create_statistical_chart(self, analysis: Dict) -> go.Figure:
        """Create statistical analysis visualization"""
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=(
                'P-value Distribution',
                'Effect Size Distribution',
                'Power Analysis',
                'Multiple Comparison Adjustment'
            )
        )
        
        # Simulate p-value distribution under null hypothesis
        null_p_values = np.random.beta(1, 10, 1000)
        fig.add_trace(
            go.Histogram(
                x=null_p_values,
                nbinsx=50,
                name='Null Distribution',
                opacity=0.7
            ),
            row=1, col=1
        )
        
        # Add observed p-value
        obs_p_value = analysis.get('p_value', 0.05)
        fig.add_vline(
            x=obs_p_value,
            line_dash="dash",
            line_color="red",
            row=1, col=1
        )
        
        # Effect size
        effect_sizes = np.random.normal(0.05, 0.02, 1000)
        fig.add_trace(
            go.Histogram(
                x=effect_sizes,
                nbinsx=50,
                name='Effect Size Distribution',
                opacity=0.7
            ),
            row=1, col=2
        )
        
        # Power analysis
        sample_sizes = np.arange(100, 10000, 100)
        power_values = 1 - 0.2 * np.exp(-sample_sizes / 2000)  # Simplified power curve
        
        fig.add_trace(
            go.Scatter(
                x=sample_sizes,
                y=power_values,
                mode='lines',
                name='Power Curve',
                line=dict(width=3)
            ),
            row=2, col=1
        )
        
        fig.add_hline(
            y=0.8,
            line_dash="dash",
            line_color="green",
            annotation_text="80% Power",
            row=2, col=1
        )
        
        # Multiple comparison
        comparisons = np.arange(1, 21)
        alpha_values = 0.05 / comparisons  # Bonferroni correction
        
        fig.add_trace(
            go.Scatter(
                x=comparisons,
                y=alpha_values,
                mode='lines+markers',
                name='Alpha after Correction',
                line=dict(width=2)
            ),
            row=2, col=2
        )
        
        fig.update_layout(
            height=800,
            showlegend=True,
            title_text="Statistical Analysis",
            title_x=0.5
        )
        
        return fig
    
    def _calculate_confidence_intervals(self, data: pd.DataFrame, confidence: float = 0.95) -> pd.DataFrame:
        """Calculate confidence intervals for conversion rates"""
        from scipy import stats
        
        results = []
        if 'variant' not in data.columns or 'converted' not in data.columns:
            return pd.DataFrame(results)

        variants = data['variant'].unique()
        
        for variant in variants:
            variant_data = data[data['variant'] == variant]
            n = len(variant_data)
            conversions = variant_data['converted'].sum()
            
            if n > 0:
                p = conversions / n
                # Wilson score interval
                z = stats.norm.ppf(1 - (1 - confidence) / 2)
                denominator = 1 + z**2 / n
                centre_adjusted_probability = p + z**2 / (2 * n)
                adjusted_standard_deviation = np.sqrt((p * (1 - p) + z**2 / (4 * n)) / n)
                
                lower_bound = (centre_adjusted_probability - z * adjusted_standard_deviation) / denominator
                upper_bound = (centre_adjusted_probability + z * adjusted_standard_deviation) / denominator
                
                results.append({
                    'variant': variant,
                    'conversion_rate': p,
                    'ci_lower': lower_bound,
                    'ci_upper': upper_bound,
                    'sample_size': n,
                    'conversions': conversions
                })
        
        return pd.DataFrame(results)
    
    def create_segmentation_heatmap(self, data: pd.DataFrame) -> go.Figure:
        """Create heatmap of performance across segments"""
        if 'segment' not in data.columns:
            return go.Figure()
        
        pivot_data = data.groupby(['variant', 'segment']).agg({
            'converted': 'mean',
            'session_duration': 'mean',
            'conversion_value': 'mean'
        }).reset_index()
        
        fig = px.density_heatmap(
            pivot_data,
            x='variant',
            y='segment',
            z='converted',
            histfunc='avg',
            color_continuous_scale='Viridis',
            title='Conversion Rate Heatmap by Segment'
        )
        
        return fig

#viz = ABVisualization()
