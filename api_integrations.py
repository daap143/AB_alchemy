"""
API integrations for various A/B testing and analytics platforms
"""

import requests
import json
from typing import Dict, List, Optional, Any
import pandas as pd
from datetime import datetime
import numpy as np

class PlatformIntegrations:
    def __init__(self):
        self.platforms = {
            'google_analytics': {
                'base_url': 'https://analyticsdata.googleapis.com/v1beta',
                'required_params': ['property_id']
            },
            'optimizely': {
                'base_url': 'https://api.optimizely.com/v2',
                'required_params': ['project_id', 'access_token']
            },
            'statsig': {
                'base_url': 'https://api.statsig.com/v1',
                'required_params': ['secret_key']
            },
            'launchdarkly': {
                'base_url': 'https://app.launchdarkly.com/api/v2',
                'required_params': ['access_token']
            },
            'mixpanel': {
                'base_url': 'https://mixpanel.com/api/2.0',
                'required_params': ['project_id', 'api_secret']
            },
            'amplitude': {
                'base_url': 'https://amplitude.com/api/2',
                'required_params': ['api_key', 'secret_key']
            }
        }
    
    def fetch_analytics_data(self, 
                            platform: str,
                            params: Dict,
                            start_date: str,
                            end_date: str,
                            metrics: List[str]) -> pd.DataFrame:
        """
        Fetch analytics data from various platforms
        """
        if platform not in self.platforms:
            print(f"Platform {platform} not supported")
            return pd.DataFrame()
        
        platform_config = self.platforms[platform]
        
        # Check required parameters
        for param in platform_config['required_params']:
            if param not in params:
                print(f"Missing required parameter: {param}")
                return pd.DataFrame()
        
        # Platform-specific implementations
        if platform == 'google_analytics':
            return self._fetch_ga_data(params, start_date, end_date, metrics)
        elif platform == 'mixpanel':
            return self._fetch_mixpanel_data(params, start_date, end_date, metrics)
        elif platform == 'amplitude':
            return self._fetch_amplitude_data(params, start_date, end_date, metrics)
        else:
            # Generic API call for other platforms
            return self._generic_api_fetch(platform, params, start_date, end_date, metrics)
    
    def _fetch_ga_data(self, params: Dict, start_date: str, end_date: str, metrics: List[str]) -> pd.DataFrame:
        """Fetch data from Google Analytics 4"""
        property_id = params['property_id']
        
        # GA4 API request body
        request_body = {
            "dateRanges": [
                {
                    "startDate": start_date,
                    "endDate": end_date
                }
            ],
            "dimensions": [
                {"name": "date"},
                {"name": "deviceCategory"},
                {"name": "country"}
            ],
            "metrics": [{"name": metric} for metric in metrics],
            "keepEmptyRows": True
        }
        
        # Add access token if provided
        headers = {
            'Authorization': f'Bearer {params.get("access_token", "")}',
            'Content-Type': 'application/json'
        }
        
        try:
            response = requests.post(
                f"{self.platforms['google_analytics']['base_url']}/properties/{property_id}:runReport",
                headers=headers,
                json=request_body
            )
            
            if response.status_code == 200:
                data = response.json()
                return self._parse_ga_response(data)
            else:
                print(f"GA API Error: {response.status_code} - {response.text}")
                return pd.DataFrame()
                
        except Exception as e:
            print(f"Error fetching GA data: {e}")
            return pd.DataFrame()
    
    def _fetch_mixpanel_data(self, params: Dict, start_date: str, end_date: str, metrics: List[str]) -> pd.DataFrame:
        """Fetch data from Mixpanel"""
        # Mixpanel JQL query
        query = {
            "from_date": start_date,
            "to_date": end_date,
            "events": [
                {
                    "event_selectors": [
                        {"event": metric} for metric in metrics
                    ]
                }
            ]
        }
        
        headers = {
            'Authorization': f'Basic {params["api_secret"]}',
            'Content-Type': 'application/json'
        }
        
        try:
            response = requests.post(
                f"{self.platforms['mixpanel']['base_url']}/jql",
                headers=headers,
                json=query
            )
            
            if response.status_code == 200:
                data = response.json()
                return self._parse_mixpanel_response(data)
            else:
                print(f"Mixpanel API Error: {response.status_code}")
                return pd.DataFrame()
                
        except Exception as e:
            print(f"Error fetching Mixpanel data: {e}")
            return pd.DataFrame()
    
    def _fetch_amplitude_data(self, params: Dict, start_date: str, end_date: str, metrics: List[str]) -> pd.DataFrame:
        """Fetch data from Amplitude"""
        # Amplitude Export API
        query_params = {
            'start': start_date,
            'end': end_date,
            'api_key': params['api_key'],
            'secret_key': params['secret_key']
        }
        
        try:
            response = requests.get(
                f"{self.platforms['amplitude']['base_url']}/export",
                params=query_params
            )
            
            if response.status_code == 200:
                # Amplitude returns gzipped JSON lines
                import gzip
                import io
                
                try:
                    compressed_file = io.BytesIO(response.content)
                    with gzip.GzipFile(fileobj=compressed_file) as f:
                        content = f.read().decode('utf-8')
                    
                    # Parse JSON lines
                    lines = content.strip().split('\n')
                    data = [json.loads(line) for line in lines]
                    return pd.DataFrame(data)
                except OSError:
                    # Not a gzip file (likely text error message)
                    print(f"Amplitude API returned non-gzip data: {response.text[:100]}...")
                    return pd.DataFrame()
                
            else:
                print(f"Amplitude API Error: {response.status_code}")
                return pd.DataFrame()
                
        except Exception as e:
            print(f"Error fetching Amplitude data: {e}")
            return pd.DataFrame()
    
    def _generic_api_fetch(self, platform: str, params: Dict, start_date: str, end_date: str, metrics: List[str]) -> pd.DataFrame:
        """Generic API fetch for platforms"""
        # This is a template - actual implementation would be platform-specific
        try:
            # For demonstration, return mock data
            print(f"Fetching data from {platform} (mock implementation)")
            return self._generate_mock_platform_data(start_date, end_date, metrics)
            
        except Exception as e:
            print(f"Error fetching from {platform}: {e}")
            return pd.DataFrame()
    
    def _parse_ga_response(self, data: Dict) -> pd.DataFrame:
        """Parse Google Analytics API response"""
        rows = []
        
        for row in data.get('rows', []):
            row_data = {}
            for i, dimension in enumerate(data.get('dimensionHeaders', [])):
                row_data[dimension['name']] = row['dimensionValues'][i]['value']
            
            for i, metric in enumerate(data.get('metricHeaders', [])):
                row_data[metric['name']] = row['metricValues'][i]['value']
            
            rows.append(row_data)
        
        return pd.DataFrame(rows)
    
    def _parse_mixpanel_response(self, data: List) -> pd.DataFrame:
        """Parse Mixpanel JQL response"""
        # Mixpanel returns array of events
        all_events = []
        
        for event_list in data:
            for event in event_list:
                all_events.append(event)
        
        return pd.DataFrame(all_events)
    
    def _generate_mock_platform_data(self, start_date: str, end_date: str, metrics: List[str]) -> pd.DataFrame:
        """Generate mock platform data for demonstration"""
        date_range = pd.date_range(start=start_date, end=end_date)
        
        data = []
        for date in date_range:
            for metric in metrics:
                data.append({
                    'date': date.strftime('%Y-%m-%d'),
                    'metric': metric,
                    'value': np.random.uniform(100, 1000),
                    'platform': 'mock'
                })
        
        return pd.DataFrame(data)
    
    def deploy_experiment(self, 
                         platform: str,
                         experiment_config: Dict,
                         params: Dict) -> Dict:
        """
        Deploy A/B test to external platform
        """
        deployment_templates = {
            'optimizely': {
                'url': f"{self.platforms['optimizely']['base_url']}/experiments",
                'method': 'POST',
                'payload': {
                    'project_id': params.get('project_id'),
                    'name': experiment_config.get('name'),
                    'description': experiment_config.get('description'),
                    'status': 'not_started',
                    'variations': experiment_config.get('variations', []),
                    'metrics': experiment_config.get('metrics', []),
                    'audience_conditions': experiment_config.get('audience', 'all')
                }
            },
            'statsig': {
                'url': f"{self.platforms['statsig']['base_url']}/experiments",
                'method': 'POST',
                'payload': {
                    'name': experiment_config.get('name'),
                    'idType': 'userID',
                    'groups': experiment_config.get('variations', []),
                    'targetApp': params.get('target_app', 'web'),
                    'config': experiment_config.get('config', {})
                }
            }
        }
        
        if platform not in deployment_templates:
            return {
                'success': False,
                'error': f"Platform {platform} deployment not supported"
            }
        
        template = deployment_templates[platform]
        headers = self._get_auth_headers(platform, params)
        
        try:
            response = requests.request(
                method=template['method'],
                url=template['url'],
                headers=headers,
                json=template['payload']
            )
            
            if response.status_code in [200, 201]:
                return {
                    'success': True,
                    'platform_experiment_id': response.json().get('id'),
                    'dashboard_url': response.json().get('dashboard_url', ''),
                    'message': 'Experiment deployed successfully'
                }
            else:
                return {
                    'success': False,
                    'error': f"Platform API error: {response.status_code}",
                    'details': response.text
                }
                
        except Exception as e:
            return {
                'success': False,
                'error': str(e)
            }
    
    def _get_auth_headers(self, platform: str, params: Dict) -> Dict:
        """Get authentication headers for platform"""
        if platform == 'optimizely':
            return {
                'Authorization': f"Bearer {params.get('access_token', '')}",
                'Content-Type': 'application/json'
            }
        elif platform == 'statsig':
            return {
                'STATSIG-API-KEY': params.get('secret_key', ''),
                'Content-Type': 'application/json'
            }
        else:
            return {'Content-Type': 'application/json'}

#integrations = PlatformIntegrations()
