"""
Realistic A/B test data simulator with user segmentation and behavior patterns
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Tuple
import random

class ABDataSimulator:
    def __init__(self, seed: int = 42):
        np.random.seed(seed)
        random.seed(seed)
        
        # Common user segments
        self.segments = {
            'new_users': {'weight': 0.3, 'conversion_rate': 0.03},
            'returning_users': {'weight': 0.4, 'conversion_rate': 0.05},
            'power_users': {'weight': 0.2, 'conversion_rate': 0.08},
            'at_risk_users': {'weight': 0.1, 'conversion_rate': 0.01}
        }
        
        # Time patterns
        self.time_patterns = {
            'hourly': {'peak_hours': [9, 10, 14, 15, 20, 21], 'peak_multiplier': 1.5},
            'weekly': {'weekend_multiplier': 1.2, 'monday_multiplier': 0.8},
            'seasonal': {'trend': 0.001}  # Daily growth rate
        }
    
    def simulate_experiment_data(self,
                                experiment_design: Dict,
                                uplift: float = 0.05,
                                duration_days: int = 7,
                                daily_traffic: int = 1000) -> pd.DataFrame:
        """
        Simulate realistic A/B test data with patterns and noise
        """
        variants = experiment_design.get('variants', [{'name': 'control'}, {'name': 'treatment'}])
        
        data = []
        start_date = datetime.now() - timedelta(days=duration_days)
        
        for day in range(duration_days):
            current_date = start_date + timedelta(days=day)
            
            for variant in variants:
                variant_name = variant['name']
                allocation = variant.get('allocation', 0.5)
                
                # Daily traffic for this variant
                daily_variant_traffic = int(daily_traffic * allocation)
                
                for _ in range(daily_variant_traffic):
                    # Generate user properties
                    user_id = f"user_{random.randint(100000, 999999)}"
                    segment = self._assign_segment()
                    
                    # Base conversion rate
                    base_cr = self.segments[segment]['conversion_rate']
                    
                    # Apply uplift for treatment variants
                    if variant_name != 'control' and 'control' not in variant_name.lower():
                        variant_cr = base_cr * (1 + uplift + np.random.normal(0, 0.02))
                    else:
                        variant_cr = base_cr + np.random.normal(0, 0.01)
                    
                    # Apply time patterns
                    time_multiplier = self._get_time_multiplier(current_date)
                    variant_cr *= time_multiplier
                    
                    # Ensure realistic bounds
                    variant_cr = max(0.001, min(0.5, variant_cr))
                    
                    # Simulate conversion
                    converted = np.random.binomial(1, variant_cr)
                    
                    # Generate session metrics
                    session_data = self._generate_session_metrics(segment, variant_name, converted)
                    
                    # Create data row
                    row = {
                        'timestamp': current_date + timedelta(hours=np.random.randint(0, 24),
                                                              minutes=np.random.randint(0, 60)),
                        'user_id': user_id,
                        'variant': variant_name,
                        'segment': segment,
                        'converted': converted,
                        'conversion_value': converted * np.random.lognormal(3, 1) if converted else 0,
                        'session_duration': session_data['duration'],
                        'page_views': session_data['page_views'],
                        'clicks': session_data['clicks'],
                        'bounce': session_data['bounce'],
                        'device': np.random.choice(['mobile', 'desktop', 'tablet'], 
                                                  p=[0.6, 0.35, 0.05]),
                        'browser': np.random.choice(['chrome', 'safari', 'firefox', 'edge'],
                                                   p=[0.65, 0.2, 0.1, 0.05]),
                        'country': np.random.choice(['US', 'UK', 'CA', 'AU', 'DE', 'FR', 'JP'],
                                                   p=[0.4, 0.15, 0.1, 0.05, 0.1, 0.1, 0.1]),
                        'referrer': np.random.choice(['direct', 'organic', 'social', 'email', 'paid'],
                                                    p=[0.3, 0.4, 0.15, 0.1, 0.05])
                    }
                    
                    # Add variant-specific effects
                    if variant_name != 'control':
                        row = self._apply_variant_effects(row, variant_name)
                    
                    data.append(row)
        
        df = pd.DataFrame(data)
        
        # Add derived metrics
        if not df.empty:
            df['hour_of_day'] = df['timestamp'].dt.hour
            df['day_of_week'] = df['timestamp'].dt.dayofweek
            df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)
        
        return df
    
    def _assign_segment(self) -> str:
        """Assign user segment based on weights"""
        segments = list(self.segments.keys())
        weights = [self.segments[s]['weight'] for s in segments]
        return np.random.choice(segments, p=weights)
    
    def _get_time_multiplier(self, dt: datetime) -> float:
        """Calculate time-based multiplier"""
        multiplier = 1.0
        
        # Hourly pattern
        if dt.hour in self.time_patterns['hourly']['peak_hours']:
            multiplier *= self.time_patterns['hourly']['peak_multiplier']
        
        # Weekly pattern
        if dt.weekday() == 0:  # Monday
            multiplier *= self.time_patterns['weekly']['monday_multiplier']
        elif dt.weekday() in [5, 6]:  # Weekend
            multiplier *= self.time_patterns['weekly']['weekend_multiplier']
        
        # Seasonal trend
        days_since_start = (dt - datetime(dt.year, 1, 1)).days
        multiplier *= (1 + self.time_patterns['seasonal']['trend']) ** days_since_start
        
        return multiplier
    
    def _generate_session_metrics(self, segment: str, variant: str, converted: int) -> Dict:
        """Generate realistic session metrics"""
        # Base metrics by segment
        base_metrics = {
            'new_users': {'duration': 60, 'page_views': 2.5, 'clicks': 3, 'bounce_prob': 0.6},
            'returning_users': {'duration': 120, 'page_views': 4.0, 'clicks': 6, 'bounce_prob': 0.4},
            'power_users': {'duration': 180, 'page_views': 6.0, 'clicks': 10, 'bounce_prob': 0.2},
            'at_risk_users': {'duration': 30, 'page_views': 1.5, 'clicks': 1, 'bounce_prob': 0.8}
        }
        
        base = base_metrics[segment]
        
        # Add randomness
        duration = max(10, np.random.exponential(base['duration']))
        page_views = max(1, int(np.random.poisson(base['page_views'])))
        clicks = max(0, int(np.random.poisson(base['clicks'])))
        bounce = np.random.binomial(1, base['bounce_prob'])
        
        # Conversion increases engagement
        if converted:
            duration *= np.random.uniform(1.5, 2.5)
            page_views = int(page_views * np.random.uniform(1.2, 1.8))
            clicks = int(clicks * np.random.uniform(1.2, 1.8))
            bounce = 0
        
        # Variant effects (treatment variants often increase engagement)
        if 'treatment' in variant.lower() or 'variant' in variant.lower():
            duration *= np.random.uniform(1.0, 1.3)
            page_views = int(page_views * np.random.uniform(1.0, 1.2))
            clicks = int(clicks * np.random.uniform(1.0, 1.2))
            bounce_prob_reduction = np.random.uniform(0.8, 1.0)
            if bounce:
                bounce = np.random.binomial(1, base['bounce_prob'] * bounce_prob_reduction)
        
        return {
            'duration': duration,
            'page_views': page_views,
            'clicks': clicks,
            'bounce': bounce
        }
    
    def _apply_variant_effects(self, row: Dict, variant_name: str) -> Dict:
        """Apply variant-specific effects to metrics"""
        effects = {
            'button_color_red': {'clicks_multiplier': 1.15},
            'simplified_checkout': {'conversion_multiplier': 1.2, 'duration_multiplier': 0.8},
            'personalized_recommendations': {'page_views_multiplier': 1.3, 'conversion_value_multiplier': 1.25},
            'free_shipping_banner': {'conversion_multiplier': 1.1}
        }
        
        if variant_name in effects:
            effect = effects[variant_name]
            
            if 'clicks_multiplier' in effect:
                row['clicks'] = int(row['clicks'] * effect['clicks_multiplier'])
            
            if 'duration_multiplier' in effect:
                row['session_duration'] *= effect['duration_multiplier']
            
            if 'conversion_value_multiplier' in effect and row['converted']:
                row['conversion_value'] *= effect['conversion_value_multiplier']
        
        return row
    
    def generate_time_series_data(self, 
                                 variant_data: pd.DataFrame,
                                 interval: str = 'hour') -> pd.DataFrame:
        """
        Aggregate data into time series for monitoring
        """
        # SAFE COPY: Don't modify the original dataframe in place
        df_copy = variant_data.copy()
        
        # Check if index is already set
        if 'timestamp' in df_copy.columns:
             df_copy.set_index('timestamp', inplace=True)
        
        if interval == 'hour':
            resample_rule = '1H'
        elif interval == 'day':
            resample_rule = '1D'
        else:
            resample_rule = '1H'
        
        time_series = df_copy.groupby('variant').resample(resample_rule).agg({
            'user_id': 'count',
            'converted': 'sum',
            'conversion_value': 'sum',
            'session_duration': 'mean',
            'page_views': 'mean',
            'clicks': 'mean',
            'bounce': 'mean'
        }).reset_index()
        
        # Avoid division by zero
        time_series['conversion_rate'] = np.where(
            time_series['user_id'] > 0, 
            time_series['converted'] / time_series['user_id'], 
            0
        )
        time_series['value_per_user'] = np.where(
            time_series['user_id'] > 0,
            time_series['conversion_value'] / time_series['user_id'],
            0
        )
        
        return time_series

#simulator = ABDataSimulator()
