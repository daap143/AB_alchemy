"""
Gemini-powered A/B Testing Autonomous Agent
Handles hypothesis generation, analysis, and decision making
"""

import google.genai as genai
import json
import pandas as pd
from typing import Dict, List, Optional
import re

class GeminiABAgent:
    def __init__(self, api_key: str):
        """Initialize Gemini with API key"""
        self.client = genai.Client(api_key=api_key)
        
        self.agents = {
            "strategist": "You are a senior product strategist with 10+ years experience in growth and A/B testing.",
            "statistician": "You are a PhD statistician specializing in experimental design and causal inference.",
            "analyst": "You are a data analyst who excels at finding insights in user behavior data.",
            "engineer": "You are a full-stack engineer who implements A/B tests in production systems.",
            "ethicist": "You are an AI ethicist who ensures experiments are fair and unbiased."
        }
    
    def _clean_json(self, text: str) -> str:
        """Helper to clean JSON returned by LLM"""
        # Remove markdown code blocks if present
        text = re.sub(r'^```json\s*', '', text, flags=re.MULTILINE)
        text = re.sub(r'^```\s*', '', text, flags=re.MULTILINE)
        text = re.sub(r'```$', '', text, flags=re.MULTILINE)
        return text.strip()

    def generate_hypotheses(self, 
                           business_goal: str,
                           user_data: Optional[Dict] = None,
                           past_experiments: Optional[List] = None) -> List[Dict]:
        prompt = f"""
        {self.agents["strategist"]}
        
        Business Goal: {business_goal}
        {f'User Data Summary: {user_data}' if user_data else 'No user data provided'}
        
        Generate 5 specific, testable A/B test hypotheses.
        Return ONLY valid JSON array with keys: 
        id, title, hypothesis, primary_metric, secondary_metrics, 
        expected_impact, confidence, complexity, risk, rationale
        """
        
        try:
            response = self.client.models.generate_content(
                 model='gemini-3-pro-preview', 
                 contents=prompt
            )
            cleaned_text = self._clean_json(response.text)
            return json.loads(cleaned_text)
        except Exception as e:
            print(f"Error in generate_hypotheses: {e}")
            return []
    
    def design_experiment(self, hypothesis: Dict, traffic_available: int = 10000) -> Dict:
        """
        Design complete experiment including sample size, duration, and variants
        """
        # Safety check if client failed to load
        if not self.client: 
            return {
                "total_sample_size": traffic_available,
                "recommended_duration_days": 14,
                "variants": [{"name": "control", "allocation": 0.5}, {"name": "treatment", "allocation": 0.5}]
            }

        prompt = f"""
        {self.agents["statistician"]}
        
        Hypothesis: {hypothesis.get('hypothesis', '')}
        Available Traffic: {traffic_available} users/day
        
        Design a statistically sound A/B test.
        Return ONLY valid JSON with keys:
        - sample_size_per_variant
        - total_sample_size
        - minimum_duration_days
        - recommended_duration_days
        - variants (array with name, description, allocation)
        - randomization_method
        - bias_mitigation
        - success_criteria
        - stopping_rules
        """
        try:
            response = self.client.models.generate_content(
                model='gemini-3-pro-preview',
                contents=prompt
            )
            
            cleaned_text = self._clean_json(response.text)
            design = json.loads(cleaned_text)

            implementation = self._generate_implementation_plan(hypothesis, design)
            design['implementation'] = implementation
            
            return design
            
        except Exception as e:
            print(f"Error designing experiment: {e}")
            # Fallback to prevent app crash
            return {
                "total_sample_size": traffic_available,
                "recommended_duration_days": 14,
                "variants": [{"name": "control", "allocation": 0.5}, {"name": "treatment", "allocation": 0.5}]
            }

    def analyze_results(self, 
                       raw_data: pd.DataFrame,
                       experiment_design: Dict,
                       hypothesis: Dict) -> Dict:
        """
        Perform statistical analysis on A/B test results
        """
        if not self.client: return {}

        # Convert data to analysis-friendly format
        # We use include='all' to get summaries of categorical data too, if valuable
        data_summary = raw_data.describe(include='all').to_dict() if not raw_data.empty else {}
        
        prompt = f"""
        {self.agents["analyst"]}
        
        Hypothesis: {hypothesis.get('hypothesis', '')}
        Experiment Design: {json.dumps(experiment_design, indent=2)}
        
        # --- FIX IS HERE: Added default=str to handle Timestamps ---
        Data Summary: {json.dumps(data_summary, indent=2, default=str)}
        # -----------------------------------------------------------
        
        Perform comprehensive statistical analysis.
        Return ONLY valid JSON with keys: p_value, significant (bool), uplift_percentage, effect_size, recommendation.
        """
        try:
            response = self.client.models.generate_content(
                model='gemini-3-pro',
                contents=prompt
            )
            cleaned_text = self._clean_json(response.text)
            analysis = json.loads(cleaned_text)
            
            # Add automated decision
            analysis['automated_decision'] = self._make_decision(analysis)
            return analysis
        except Exception as e:
            print(f"Error analyzing results: {e}")
            return {"significant": False, "uplift_percentage": 0, "p_value": 1.0, "recommendation": "Error in analysis"}

    
    def generate_report(self, hypothesis: Dict, design: Dict, analysis: Dict, business_context: str) -> str:
        prompt = f"""
        Generate a markdown report for this A/B test.
        Hypothesis: {hypothesis.get('title')}
        Results: {analysis}
        """
        try:
            response = self.client.models.generate_content(
                model='gemini-3-pro-preview',
                contents=prompt
            )
            return response.text
        except Exception as e:
            return f"Error generating report: {e}"
    
    def _generate_implementation_plan(self, hypothesis: Dict, design: Dict) -> Dict:
        prompt = f"""
        {self.agents["engineer"]}
        Implement this A/B test: {hypothesis.get('title', '')}
        Return JSON with technical specifications.
        """
        try:
            response = self.client.models.generate_content(
                model='gemini-3-pro-preview',
                contents=prompt
            )
            return json.loads(self._clean_json(response.text))
        except:
            return {}
    
    def _make_decision(self, analysis: Dict) -> Dict:
        prompt = f"""
        Based on this analysis: {json.dumps(analysis)}
        Make a clear implementation decision (IMPLEMENT, ITERATE, ABANDON).
        Return JSON with: decision, confidence, reasoning.
        """
        try:
            response = self.client.models.generate_content(
                model='gemini-3-pro-preview',
                contents=prompt
            )
            return json.loads(self._clean_json(response.text))
        except:
            return {"decision": "Review Manually"}
