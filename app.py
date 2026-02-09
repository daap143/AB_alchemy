"""
Main Streamlit application for the Autonomous A/B Testing Platform
"""

import streamlit as st
import pandas as pd
import numpy as np
import json
from datetime import datetime, timedelta
import plotly.graph_objects as go

# Import our modules
from gemini_agent import GeminiABAgent #, GEMINI_API_KEY
from data_simulator import ABDataSimulator
from visualization import ABVisualization
from api_integrations import PlatformIntegrations

# Page configuration
st.set_page_config(
    page_title="Autonomous A/B Testing Platform",
    page_icon="🧪",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1E3A8A;
        font-weight: 700;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #374151;
        font-weight: 600;
        margin-top: 1.5rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        margin-bottom: 1rem;
    }
    .stButton > button {
        width: 100%;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        font-weight: 600;
        border: none;
        padding: 0.5rem 1rem;
        border-radius: 5px;
    }
    .stButton > button:hover {
        background: linear-gradient(135deg, #5a67d8 0%, #6b46c1 100%);
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'experiments' not in st.session_state:
    st.session_state.experiments = []
if 'current_experiment' not in st.session_state:
    st.session_state.current_experiment = None
if 'GEMINI_API_KEY' not in st.session_state:
    st.session_state.GEMINI_API_KEY = ""
if 'gemini_agent' not in st.session_state:
    # NEW: Initialize agent only when API key is available
    # We'll create it later when user enters API key
    st.session_state.gemini_agent = None

# NEW: Function to safely get agent
def get_gemini_agent():
    """Get Gemini agent, or show error if not initialized"""
    if st.session_state.gemini_agent is None:
        st.error("⚠️ Please enter your Gemini API key in the sidebar first!")
        return None
    return st.session_state.gemini_agent

# Initialize other components
simulator = ABDataSimulator()
viz = ABVisualization()
platforms = PlatformIntegrations()

# Main header
st.markdown('<h1 class="main-header">🧪 Autonomous A/B Testing Platform</h1>', unsafe_allow_html=True)
st.markdown("**Powered by Gemini 3 AI** • Automate your entire experimentation workflow")

# Sidebar
with st.sidebar:
    st.image("https://placehold.co/300x100/667eea/ffffff?text=AB+Alchemy", width=250)
    
    st.markdown("### 🔧 Navigation")
    page = st.radio(
        "Select Page",
        ["🏠 Dashboard", "💡 Generate Hypotheses", "⚡ Run Experiment", 
         "📊 Analyze Results", "🔄 Platform Integrations", "📈 Reports"]
    )
    
    st.markdown("---")
    
    st.markdown("### ⚙️ Settings")
    
    # NEW: Better API key handling
    if 'GEMINI_API_KEY' not in st.session_state:
        st.session_state.GEMINI_API_KEY = ""
    
    api_key = st.text_input(
        "🔑 Gemini API Key",
        value=st.session_state.GEMINI_API_KEY,
        type="password",
        help="Enter your Gemini API key. Get it from Google AI Studio."
    )
    
    # NEW: Only update if key changed and is valid
    if api_key and api_key != "YOUR_API_KEY_HERE" and api_key != st.session_state.GEMINI_API_KEY:
        try:
            st.session_state.GEMINI_API_KEY = api_key
            # Re-initialize agent with new key
            st.session_state.gemini_agent = GeminiABAgent(api_key=api_key)
            st.success("✅ API Key updated and agent reinitialized!")
        except Exception as e:
            st.error(f"❌ Error with API key: {e}")
            st.session_state.GEMINI_API_KEY = ""  # Reset on error

    st.markdown("---")
    
    st.markdown("### 📚 Quick Actions")
    if st.button("🆕 New Experiment"):
        st.session_state.current_experiment = None
        st.rerun()
    
    if st.button("🔄 Clear All Data"):
        st.session_state.experiments = []
        st.session_state.current_experiment = None
        st.rerun()

# Page routing
if page == "🏠 Dashboard":
    st.markdown('<h2 class="sub-header">Experiment Dashboard</h2>', unsafe_allow_html=True)
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric("Total Experiments", len(st.session_state.experiments))
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric("Active Tests", sum(1 for exp in st.session_state.experiments 
                                     if exp.get('status') == 'running'))
        st.markdown('</div>', unsafe_allow_html=True)

    with col3:
        st.metric("Success Rate", f"{sum(1 for exp in st.session_state.experiments if exp.get('result') == 'success') / max(1, len(st.session_state.experiments)):.0%}")

    with col4:
        # Calculate uplifts list safely
        uplifts = [exp.get('uplift', 0) for exp in st.session_state.experiments if exp.get('uplift')]
        # Only calculate mean if list is not empty
        avg_uplift = np.mean(uplifts) if uplifts else 0.0
        
        st.metric("Avg. Uplift", f"{avg_uplift:.1f}%")

    # Recent experiments table
    if st.session_state.experiments:
        st.markdown("### 📋 Recent Experiments")
        recent_df = pd.DataFrame(st.session_state.experiments[-5:])
        st.dataframe(
            recent_df[['name', 'status', 'start_date', 'sample_size', 'uplift', 'result']],
            use_container_width=True
        )
    
    # Quick start
    st.markdown("### 🚀 Quick Start")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### Generate Hypotheses")
        business_goal = st.text_area(
            "Business Goal",
            value="Increase user sign-up conversion rate",
            height=100
        )
        if st.button("🚀 Generate AI Hypotheses", type="primary"):
            # NEW: Get agent safely
            agent = get_gemini_agent()
            if agent is None:
                st.stop()  # Stop execution if no agent
    
            # NEW: Loading indicator with expanded status
            with st.status("🧠 Gemini is analyzing your business and generating hypotheses...", expanded=True) as status:
                try:
                    # Prepare user data
                    user_data = {
                        "app_type": "SaaS", # Defaulting for quick start
                        "target_metric": "Conversion Rate",
                        "monthly_traffic": 10000,
                        "data_sources": ["analytics", "feedback", "transactions"]
                    }
                    
                    # Generate hypotheses
                    hypotheses = agent.generate_hypotheses(
                         business_goal=business_goal,
                         user_data=user_data,
                         past_experiments=st.session_state.experiments[-10:] if st.session_state.experiments else None
                    )
            
                    st.session_state.generated_hypotheses = hypotheses
                    status.update(label="✅ Hypotheses generated successfully!", state="complete")
                    st.success(f"✅ Generated {len(hypotheses)} data-driven hypotheses!")
            
                except Exception as e:
                    status.update(label="❌ Error generating hypotheses", state="error")
                    st.error(f"Error: {str(e)}")
                    # Provide debugging help
                    if "API key" in str(e).lower():
                        st.info("💡 Check that your Gemini API key is valid and has sufficient quota.")
    
    with col2:
        st.markdown("#### Sample Experiment")
        if st.button("🎯 Load Sample Experiment"):
            sample_exp = {
                "id": "exp_001",
                "name": "Homepage Headline Test",
                "hypothesis": "If we change the headline to focus on benefits, conversion will increase by 5-10%",
                "status": "completed",
                "result": "success",
                "uplift": 7.2,
                "sample_size": 5000,
                "start_date": "2024-01-15",
                "end_date": "2024-01-22"
            }
            st.session_state.current_experiment = sample_exp
            st.success("Sample experiment loaded!")
            st.rerun()

elif page == "💡 Generate Hypotheses":
    st.markdown('<h2 class="sub-header">AI-Powered Hypothesis Generation</h2>', unsafe_allow_html=True)
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("#### Business Context")
        
        business_goal = st.text_area(
            "Primary Business Goal",
            value="Increase monthly recurring revenue for our SaaS product",
            height=100
        )
        
        app_type = st.selectbox(
            "Application Type",
            ["Web Application", "Mobile App", "E-commerce", "SaaS", "Content Platform", "Marketplace"]
        )
        
        target_metric = st.selectbox(
            "Primary Metric to Improve",
            ["Conversion Rate", "Revenue per User", "Retention", "Engagement", "Customer Satisfaction"]
        )
        
        available_traffic = st.number_input(
            "Monthly Unique Visitors",
            min_value=1000,
            max_value=10000000,
            value=10000,
            step=1000
        )
    
    with col2:
        st.markdown("#### Data Sources")
        
        st.checkbox("📈 Google Analytics Connected", value=True)
        st.checkbox("📊 Mixpanel/Amplitude", value=True)
        st.checkbox("📝 User Feedback", value=True)
        st.checkbox("💳 Transaction Data", value=True)
        
        st.markdown("---")
        
        past_experiments = st.slider(
            "Past Experiments to Analyze",
            min_value=0,
            max_value=50,
            value=10
        )
    
    if st.button("🚀 Generate AI Hypotheses", type="primary"):
        with st.spinner("🧠 Gemini is analyzing your business and generating hypotheses..."):
            try:
                # Prepare user data
                user_data = {
                    "app_type": app_type,
                    "target_metric": target_metric,
                    "monthly_traffic": available_traffic,
                    "data_sources": ["analytics", "feedback", "transactions"]
                }
                
                # Generate hypotheses
                hypotheses = st.session_state.gemini_agent.generate_hypotheses(
                    business_goal=business_goal,
                    user_data=user_data,
                    past_experiments=st.session_state.experiments[-past_experiments:] if past_experiments > 0 else None
                )
                
                st.session_state.generated_hypotheses = hypotheses
                st.success(f"✅ Generated {len(hypotheses)} data-driven hypotheses!")
                
            except Exception as e:
                st.error(f"Error generating hypotheses: {e}")
    
    # Display generated hypotheses
    if 'generated_hypotheses' in st.session_state:
        st.markdown("---")
        st.markdown("### 📋 Generated Hypotheses")
        
        for i, hypothesis in enumerate(st.session_state.generated_hypotheses):
            with st.expander(f"**{i+1}. {hypothesis.get('title', 'Hypothesis')}**"):
                col1, col2 = st.columns([3, 1])
                
                with col1:
                    st.markdown(f"**Hypothesis:** {hypothesis.get('hypothesis', '')}")
                    st.markdown(f"**Primary Metric:** {hypothesis.get('primary_metric', '')}")
                    st.markdown(f"**Expected Impact:** {hypothesis.get('expected_impact', '')}")
                    st.markdown(f"**Rationale:** {hypothesis.get('rationale', '')}")
                
                with col2:
                    st.metric("Confidence", hypothesis.get('confidence', 'Medium'))
                    st.metric("Complexity", hypothesis.get('complexity', 'Medium'))
                    st.metric("Risk", hypothesis.get('risk', 'Medium'))
                
                # Action buttons for each hypothesis
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    # In the "Design Experiment" section
                    if st.button("🤖 Generate Optimal Design with Gemini", key=f"design_{i}"):
                        agent = get_gemini_agent()
                        if agent is None:
                            st.stop()
        
                        with st.status("🤖 Gemini is calculating optimal experiment design...", expanded=True) as status:
                            try:
                                design = agent.design_experiment(
                                   hypothesis,
                                   traffic_available=available_traffic
                                )
                
                                st.session_state.experiment_design = design
                                status.update(label="✅ Experiment design generated!", state="complete")
                                st.success("✅ Design complete!")
                
                                # Display design
                                st.json(design, expanded=False)
                
                            except Exception as e:
                                status.update(label="❌ Error generating design", state="error")
                                st.error(f"Error: {str(e)}")
                with col2:
                    if st.button(f"📊 Quick Simulate", key=f"sim_{i}"):
                        with st.spinner("Running simulation..."):
                            # Quick simulation
                            design = st.session_state.gemini_agent.design_experiment(
                                hypothesis, 
                                traffic_available=available_traffic
                            )
                            
                            # Simulate data
                            data = simulator.simulate_experiment_data(
                                design,
                                uplift=0.05,
                                duration_days=7,
                                daily_traffic=available_traffic // 30
                            )
                            
                            # Analyze
                            analysis = st.session_state.gemini_agent.analyze_results(
                                data, design, hypothesis
                            )
                            
                            st.subheader("Quick Simulation Results")
                            st.metric("Expected Uplift", f"{analysis.get('uplift_percentage', 0):.1f}%")
                            st.metric("Statistical Significance", 
                                     "✅ Yes" if analysis.get('significant', False) else "❌ No")
                
                with col3:
                    if st.button(f"💾 Save", key=f"save_{i}"):
                        st.session_state.experiments.append({
                            "name": hypothesis.get('title'),
                            "hypothesis": hypothesis.get('hypothesis'),
                            "status": "draft",
                            "created_date": datetime.now().strftime("%Y-%m-%d")
                        })
                        st.success("Hypothesis saved to experiments!")

elif page == "⚡ Run Experiment":
    st.markdown('<h2 class="sub-header">Design & Run A/B Test</h2>', unsafe_allow_html=True)
    
    # Step wizard
    steps = ["1. Select Hypothesis", "2. Design Experiment", "3. Configure Variants", 
             "4. Set Metrics", "5. Launch"]
    current_step = st.radio("Progress", steps, horizontal=True)
    
    if current_step == "1. Select Hypothesis":
        st.markdown("#### Choose a Hypothesis")
        
        # Show saved hypotheses or generate new
        hypothesis_source = st.radio(
            "Hypothesis Source",
            ["Use Generated Hypothesis", "Create New Hypothesis", "Import from File"]
        )
        
        if hypothesis_source == "Use Generated Hypothesis":
            if 'generated_hypotheses' in st.session_state:
                hypothesis_options = {h['title']: h for h in st.session_state.generated_hypotheses}
                selected_title = st.selectbox("Select Hypothesis", list(hypothesis_options.keys()))
                
                if selected_title:
                    selected_hypothesis = hypothesis_options[selected_title]
                    st.session_state.selected_hypothesis = selected_hypothesis
                    
                    st.markdown("**Selected Hypothesis:**")
                    st.info(selected_hypothesis['hypothesis'])
                    
                    if st.button("Next: Design Experiment →"):
                        st.rerun()
            else:
                st.warning("No hypotheses generated yet. Go to 'Generate Hypotheses' first.")
        
        elif hypothesis_source == "Create New Hypothesis":
            col1, col2 = st.columns(2)
            
            with col1:
                hypothesis_title = st.text_input("Experiment Title")
                hypothesis_statement = st.text_area(
                    "Hypothesis Statement (If... then... because...)",
                    height=100
                )
            
            with col2:
                primary_metric = st.selectbox(
                    "Primary Metric",
                    ["Conversion Rate", "Revenue", "Engagement", "Retention", "Customer Satisfaction"]
                )
                expected_impact = st.slider("Expected Impact (%)", 1, 50, 10)
            
            if st.button("Create & Continue"):
                st.session_state.selected_hypothesis = {
                    "title": hypothesis_title,
                    "hypothesis": hypothesis_statement,
                    "primary_metric": primary_metric,
                    "expected_impact": f"{expected_impact}%"
                }
                st.success("Hypothesis created!")
                st.rerun()
    
    elif current_step == "2. Design Experiment":
        if 'selected_hypothesis' not in st.session_state:
            st.warning("Please select a hypothesis first.")
            st.stop()
        
        hypothesis = st.session_state.selected_hypothesis
        
        st.markdown(f"#### Designing: {hypothesis.get('title')}")
        st.info(f"**Hypothesis:** {hypothesis.get('hypothesis')}")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("##### Experiment Parameters")
            
            sample_size = st.number_input(
                "Total Sample Size",
                min_value=100,
                max_value=1000000,
                value=5000,
                step=100
            )
            
            test_duration = st.slider(
                "Test Duration (Days)",
                min_value=1,
                max_value=60,
                value=14
            )
            
            confidence_level = st.select_slider(
                "Confidence Level",
                options=[90, 95, 99],
                value=95
            )
            
            statistical_power = st.slider(
                "Statistical Power",
                min_value=70,
                max_value=99,
                value=80
            )
        
        with col2:
            st.markdown("##### AI-Powered Design")
            
            if st.button("🤖 Generate Optimal Design with Gemini"):
                with st.spinner("Gemini is calculating optimal experiment design..."):
                    try:
                        design = st.session_state.gemini_agent.design_experiment(
                            hypothesis,
                            traffic_available=sample_size
                        )
                        
                        st.session_state.experiment_design = design
                        st.success("Experiment design generated!")
                        
                        # Display design
                        st.json(design, expanded=False)
                        
                    except Exception as e:
                        st.error(f"Error generating design: {e}")
            
            if 'experiment_design' in st.session_state:
                design = st.session_state.experiment_design
                st.metric("Required Sample", design.get('total_sample_size', 0))
                st.metric("Duration", f"{design.get('recommended_duration_days', 0)} days")
                st.metric("Variants", len(design.get('variants', [])))
    
    elif current_step == "3. Configure Variants":
        st.markdown("#### Configure Test Variants")
        
        # Default variants
        variants = [
            {"name": "control", "description": "Original version", "allocation": 0.5},
            {"name": "treatment", "description": "New version", "allocation": 0.5}
        ]
        
        if 'experiment_design' in st.session_state:
            variants = st.session_state.experiment_design.get('variants', variants)
        
        # Editable variant configuration
        for i, variant in enumerate(variants):
            with st.expander(f"Variant: {variant['name'].upper()}", expanded=(i == 0)):
                col1, col2, col3 = st.columns([1, 2, 1])
                
                with col1:
                    new_name = st.text_input(
                        "Name",
                        value=variant['name'],
                        key=f"name_{i}"
                    )
                
                with col2:
                    description = st.text_area(
                        "Description",
                        value=variant.get('description', ''),
                        height=50,
                        key=f"desc_{i}"
                    )
                
                with col3:
                    allocation = st.slider(
                        "Traffic Allocation",
                        min_value=0.0,
                        max_value=1.0,
                        value=variant.get('allocation', 0.5),
                        step=0.05,
                        key=f"alloc_{i}"
                    )
                
                # Variant-specific configuration
                config_type = st.selectbox(
                    "Configuration Type",
                    ["Frontend (HTML/CSS)", "Backend (API/Logic)", "Content (Copy/Images)", "Pricing"],
                    key=f"type_{i}"
                )
                
                if config_type == "Frontend (HTML/CSS)":
                    code = st.text_area(
                        "HTML/CSS Changes",
                        value="",
                        height=100,
                        key=f"code_{i}"
                    )
        
        # Add new variant button
        if st.button("➕ Add Another Variant"):
            variants.append({"name": f"variant_{len(variants)}", "allocation": 0.1})
    
    elif current_step == "4. Set Metrics":
        st.markdown("#### Define Success Metrics")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("##### Primary Metric")
            primary_metric = st.selectbox(
                "Primary Success Metric",
                ["Conversion Rate", "Revenue per User", "Click-Through Rate", 
                 "Session Duration", "Retention Rate", "Customer Satisfaction Score"],
                index=0
            )
            
            success_threshold = st.number_input(
                "Minimum Improvement for Success (%)",
                min_value=1.0,
                max_value=100.0,
                value=5.0,
                step=0.5
            )
            
            st.markdown("##### Statistical Criteria")
            p_value_threshold = st.number_input(
                "p-value Threshold",
                min_value=0.001,
                max_value=0.1,
                value=0.05,
                step=0.005,
                format="%.3f"
            )
        
        with col2:
            st.markdown("##### Guardrail Metrics")
            guardrail_metrics = st.multiselect(
                "Metrics to Monitor for Negative Impact",
                ["Revenue", "Bounce Rate", "Customer Support Tickets", 
                 "Page Load Time", "Error Rate", "User Satisfaction"],
                default=["Revenue", "Bounce Rate"]
            )
            
            st.markdown("##### Early Stopping Rules")
            early_stop_enabled = st.checkbox("Enable Early Stopping", value=True)
            
            if early_stop_enabled:
                harm_threshold = st.number_input(
                    "Stop if harm exceeds (%)",
                    value=10.0,
                    step=1.0
                )
                
                certainty_threshold = st.number_input(
                    "Stop if certainty exceeds (%)",
                    value=95.0,
                    step=1.0
                )
        
        # AI-powered metric suggestions
        if st.button("🤖 Get AI Metric Recommendations"):
            with st.spinner("Gemini is analyzing your hypothesis for optimal metrics..."):
                try:
                    # This would call Gemini for metric recommendations
                    st.info("Based on your hypothesis, Gemini recommends tracking:\n"
                           "- Conversion Rate (Primary)\n"
                           "- Revenue per Session (Guardrail)\n"
                           "- Time to First Action (Secondary)\n"
                           "- Customer Satisfaction Score (Long-term)")
                except:
                    pass
    
    elif current_step == "5. Launch":
        st.markdown("#### Launch A/B Test")
        
        # Summary
        st.markdown("##### Experiment Summary")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Hypothesis", st.session_state.selected_hypothesis.get('title', 'N/A'))
            st.metric("Sample Size", "5,000")
        
        with col2:
            st.metric("Duration", "14 days")
            st.metric("Variants", "2")
        
        with col3:
            st.metric("Primary Metric", "Conversion Rate")
            st.metric("Target Improvement", "5%")
        
        # Launch options
        st.markdown("##### Launch Method")
        
        launch_method = st.radio(
            "How would you like to launch?",
            ["Simulation Only", "Deploy to Platform", "Manual Implementation"]
        )
        
        if launch_method == "Simulation Only":
            if st.button("🚀 Run Simulation", type="primary"):
                with st.spinner("Running A/B test simulation..."):
                    # Simulate the experiment
                    design = st.session_state.get('experiment_design', {
                        'variants': [
                            {'name': 'control', 'allocation': 0.5},
                            {'name': 'treatment', 'allocation': 0.5}
                        ]
                    })
                    
                    # Generate realistic data
                    data = simulator.simulate_experiment_data(
                        design,
                        uplift=0.05,  # 5% uplift
                        duration_days=14,
                        daily_traffic=5000 // 14
                    )
                    
                    # Analyze results
                    analysis = st.session_state.gemini_agent.analyze_results(
                        data, design, st.session_state.selected_hypothesis
                    )
                    
                    # Generate report
                    report = st.session_state.gemini_agent.generate_report(
                        st.session_state.selected_hypothesis,
                        design,
                        analysis,
                        "SaaS product optimization"
                    )
                    
                    # Save to session state
                    st.session_state.simulation_results = {
                        'data': data,
                        'analysis': analysis,
                        'report': report,
                        'design': design
                    }
                    
                    st.success("Simulation completed! Navigate to 'Analyze Results' to view.")
                    
                    # Save to experiments
                    st.session_state.experiments.append({
                        'id': f"exp_{len(st.session_state.experiments) + 1}",
                        'name': st.session_state.selected_hypothesis.get('title'),
                        'status': 'completed',
                        'result': 'success' if analysis.get('significant', False) else 'inconclusive',
                        'uplift': analysis.get('uplift_percentage', 0),
                        'sample_size': len(data),
                        'start_date': datetime.now().strftime("%Y-%m-%d"),
                        'data': data.to_dict('records')[:100]  # Store sample
                    })
        
        elif launch_method == "Deploy to Platform":
            st.markdown("Select Platform for Deployment")
            
            platform = st.selectbox(
                "A/B Testing Platform",
                ["Optimizely", "Statsig", "LaunchDarkly", "Google Optimize", "Custom API"]
            )
            
            # Platform configuration
            if platform == "Optimizely":
                access_token = st.text_input("Optimizely Access Token", type="password")
                project_id = st.text_input("Project ID")
            
            if st.button(f"🚀 Deploy to {platform}", type="primary"):
                with st.spinner(f"Deploying to {platform}..."):
                    # This would call the actual platform API
                    st.info(f"Deployment to {platform} would happen here via API")
                    st.success(f"Experiment deployed to {platform} successfully!")

elif page == "📊 Analyze Results":
    st.markdown('<h2 class="sub-header">Analyze A/B Test Results</h2>', unsafe_allow_html=True)
    
    # Select experiment to analyze
    if st.session_state.experiments:
        experiment_names = [exp.get('name', f"Experiment {i+1}") 
                           for i, exp in enumerate(st.session_state.experiments)]
        
        selected_exp_name = st.selectbox(
            "Select Experiment to Analyze",
            experiment_names
        )
        
        selected_exp = next((exp for exp in st.session_state.experiments 
                            if exp.get('name') == selected_exp_name), None)
        
        if selected_exp:
            # Display experiment info
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Status", selected_exp.get('status', 'unknown').upper())
            
            with col2:
                st.metric("Sample Size", selected_exp.get('sample_size', 0))
            
            with col3:
                st.metric("Uplift", f"{selected_exp.get('uplift', 0):.1f}%" 
                         if selected_exp.get('uplift') else "N/A")
            
            with col4:
                st.metric("Result", selected_exp.get('result', 'pending').upper())
            
            # Analysis tabs
            tab1, tab2, tab3, tab4 = st.tabs(["📈 Dashboard", "📊 Statistics", "👥 Segments", "📝 AI Report"])
            
            with tab1:
                # Check if we have simulation results
                if 'simulation_results' in st.session_state:
                    data = st.session_state.simulation_results['data']
                    analysis = st.session_state.simulation_results['analysis']
                    
                    # Create dashboard
                    fig = viz.create_dashboard(data, data, analysis)
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Key metrics
                    st.markdown("### 📊 Key Metrics")
                    
                    metric_cols = st.columns(4)
                    
                    with metric_cols[0]:
                        st.metric(
                            "Conversion Rate (Control)",
                            f"{data[data['variant'] == 'control']['converted'].mean():.2%}"
                        )
                    
                    with metric_cols[1]:
                        st.metric(
                            "Conversion Rate (Treatment)",
                            f"{data[data['variant'] == 'treatment']['converted'].mean():.2%}"
                        )
                    
                    with metric_cols[2]:
                        st.metric(
                            "Uplift",
                            f"{analysis.get('uplift_percentage', 0):.1f}%",
                            delta=f"{analysis.get('uplift_percentage', 0):.1f}%"
                        )
                    
                    with metric_cols[3]:
                        st.metric(
                            "Statistical Significance",
                            "✅ Yes" if analysis.get('significant', False) else "❌ No"
                        )
                else:
                    st.info("No simulation data available. Run an experiment first.")
            
            with tab2:
                st.markdown("### 🧮 Statistical Analysis")
                
                if 'simulation_results' in st.session_state:
                    analysis = st.session_state.simulation_results['analysis']
                    
                    # Display statistical tests
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.markdown("##### Hypothesis Test")
                        st.metric("p-value", f"{analysis.get('p_value', 0):.4f}")
                        st.metric("Confidence Level", "95%")
                        st.metric("Test Power", "80%")
                    
                    with col2:
                        st.markdown("##### Effect Size")
                        st.metric("Cohen's d", f"{analysis.get('effect_size', 0):.3f}")
                        st.metric("Relative Improvement", f"{analysis.get('uplift_percentage', 0):.1f}%")
                        st.metric("Practical Significance", 
                                 "High" if analysis.get('effect_size', 0) > 0.5 else "Medium")
                    
                    # Statistical visualization
                    st.markdown("##### Statistical Distributions")
                    stat_fig = viz.create_statistical_chart(analysis)
                    st.plotly_chart(stat_fig, use_container_width=True)
                else:
                    st.info("Run a simulation to see statistical analysis.")
            
            with tab3:
                st.markdown("### 👥 Segmentation Analysis")
                
                if 'simulation_results' in st.session_state:
                    data = st.session_state.simulation_results['data']
                    
                    if 'segment' in data.columns:
                        # Segment performance
                        segment_performance = data.groupby(['variant', 'segment']).agg({
                            'converted': ['mean', 'count']
                        }).reset_index()
                        
                        segment_performance.columns = ['variant', 'segment', 'conversion_rate', 'count']
                        
                        # Create heatmap
                        heatmap_fig = viz.create_segmentation_heatmap(data)
                        st.plotly_chart(heatmap_fig, use_container_width=True)
                        
                        # Segment comparison
                        st.markdown("##### Segment-wise Conversion Rates")
                        st.dataframe(
                            segment_performance.pivot(
                                index='segment',
                                columns='variant',
                                values='conversion_rate'
                            ).style.format("{:.2%}").background_gradient(cmap='Blues'),
                            use_container_width=True
                        )
                    else:
                        st.info("No segment data available in this simulation.")
                else:
                    st.info("Run a simulation to see segmentation analysis.")
            
            with tab4:
                st.markdown("### 🤖 AI-Powered Analysis Report")
                
                if 'simulation_results' in st.session_state:
                    report = st.session_state.simulation_results['report']
                    
                    # Display report
                    st.markdown(report)
                    
                    # Download button
                    st.download_button(
                        label="📥 Download Full Report",
                        data=report,
                        file_name=f"ab_test_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md",
                        mime="text/markdown"
                    )
                    
                    # Ask follow-up questions
                    st.markdown("---")
                    st.markdown("#### 💬 Ask Gemini About Results")
                    
                    question = st.text_input(
                        "Ask a question about these results:",
                        placeholder="e.g., Why did variant B perform better? What should we test next?"
                    )
                    
                    if question and st.button("Ask"):
                        with st.spinner("Gemini is analyzing..."):
                            prompt = f"""
                            Based on this A/B test analysis: {json.dumps(st.session_state.simulation_results['analysis'], indent=2)}
                            
                            Question: {question}
                            
                            Provide a detailed, data-driven answer.
                            """
                            
                            response = st.session_state.gemini_agent.model_pro.generate_content(prompt)
                            st.markdown(response.text)
                else:
                    st.info("Run a simulation to generate AI analysis report.")
    
    else:
        st.info("No experiments to analyze. Run an experiment first.")

elif page == "🔄 Platform Integrations":
    st.markdown('<h2 class="sub-header">Platform Integrations</h2>', unsafe_allow_html=True)
    
    tab1, tab2, tab3, tab4 = st.tabs(["📊 Analytics", "🧪 A/B Platforms", "🎯 Personalization", "🔌 Custom APIs"])
    
    with tab1:
        st.markdown("#### Analytics Platform Connections")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("##### Google Analytics 4")
            ga_property = st.text_input("GA4 Property ID", placeholder="G-XXXXXXXXXX")
            ga_connected = st.checkbox("Connected", value=False)
            
            if ga_connected and ga_property:
                st.success("✅ Connected to GA4")
                
                # Fetch sample data
                if st.button("Fetch GA4 Data"):
                    with st.spinner("Fetching analytics data..."):
                        # This would actually call the GA4 API
                        sample_data = pd.DataFrame({
                            'date': pd.date_range('2024-01-01', periods=30),
                            'users': np.random.randint(1000, 5000, 30),
                            'conversions': np.random.randint(50, 300, 30)
                        })
                        
                        st.dataframe(sample_data, use_container_width=True)
        
        with col2:
            st.markdown("##### Mixpanel")
            mixpanel_project = st.text_input("Mixpanel Project ID")
            mixpanel_secret = st.text_input("API Secret", type="password")
            
            if st.button("Test Mixpanel Connection"):
                st.info("Mixpanel connection would be tested here")
    
    with tab2:
        st.markdown("#### A/B Testing Platforms")
        
        platform = st.selectbox(
            "Select Platform",
            ["Optimizely", "Statsig", "LaunchDarkly", "VWO", "AB Tasty", "Google Optimize"]
        )
        
        if platform == "Optimizely":
            st.markdown("##### Optimizely Configuration")
            
            col1, col2 = st.columns(2)
            
            with col1:
                opt_token = st.text_input("Personal Access Token", type="password")
                opt_project = st.text_input("Project ID")
            
            with col2:
                opt_sdk = st.text_input("SDK Key")
                opt_env = st.selectbox("Environment", ["Production", "Development", "Staging"])
            
            if st.button("Connect to Optimizely"):
                st.success(f"Connected to {platform} successfully!")
                
                # Show experiments from platform
                st.markdown("##### Active Experiments on Optimizely")
                
                # Mock data
                platform_experiments = pd.DataFrame({
                    'name': ['Homepage Test', 'Checkout Optimization', 'Pricing Test'],
                    'status': ['Running', 'Paused', 'Completed'],
                    'variants': [2, 3, 2],
                    'traffic': [5000, 10000, 15000]
                })
                
                st.dataframe(platform_experiments, use_container_width=True)
    
    with tab3:
        st.markdown("#### Personalization Engines")
        
        st.info("Connect to personalization platforms for targeted A/B tests")
        
        personalization_platforms = st.multiselect(
            "Select Platforms",
            ["Dynamic Yield", "Monetate", "Evergage", "Adobe Target", "Custom ML Model"]
        )
        
        if personalization_platforms:
            st.markdown("##### Configuration")
            
            for platform in personalization_platforms:
                with st.expander(f"{platform} Settings"):
                    api_key = st.text_input(f"{platform} API Key", type="password", 
                                           key=f"{platform}_key")
                    endpoint = st.text_input(f"{platform} Endpoint", 
                                            key=f"{platform}_endpoint")
                    
                    if st.button(f"Test {platform} Connection", key=f"{platform}_test"):
                        st.success(f"Connection to {platform} successful!")

elif page == "📈 Reports":
    st.markdown('<h2 class="sub-header">Experiment Reports & Insights</h2>', unsafe_allow_html=True)
    
    # Report generation
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("#### Generate Comprehensive Report")
        
        report_type = st.selectbox(
            "Report Type",
            ["Experiment Summary", "Statistical Deep Dive", "Business Impact", 
             "Executive Summary", "Team Retrospective", "Learning Database"]
        )
        
        time_range = st.select_slider(
            "Time Range",
            options=["Last Week", "Last Month", "Last Quarter", "Last Year", "All Time"]
        )
        
        include_charts = st.checkbox("Include Interactive Charts", value=True)
        include_raw_data = st.checkbox("Include Raw Data", value=False)
        include_recommendations = st.checkbox("Include AI Recommendations", value=True)
    
    with col2:
        st.markdown("#### Export Options")
        
        export_format = st.radio(
            "Format",
            ["PDF", "HTML", "Markdown", "Notion", "Google Docs", "PowerPoint"]
        )
        
        if st.button("📊 Generate Report", type="primary"):
            with st.spinner("Generating report with Gemini..."):
                # Generate comprehensive report
                if st.session_state.experiments:
                    # Use Gemini to create report
                    prompt = f"""
                    Create a comprehensive {report_type} report for these A/B tests:
                    {json.dumps(st.session_state.experiments, indent=2)}
                    
                    Time Range: {time_range}
                    
                    Include:
                    - Executive summary
                    - Key findings and insights
                    - Statistical validity assessment
                    - Business impact analysis
                    - Recommendations for future tests
                    - Lessons learned
                    
                    Format as a professional business report.
                    """
                    
                    try:
                        response = st.session_state.gemini_agent.model_pro.generate_content(prompt)
                        report_content = response.text
                        
                        st.session_state.generated_report = report_content
                        st.success("Report generated successfully!")
                        
                    except Exception as e:
                        st.error(f"Error generating report: {e}")
                else:
                    st.warning("No experiments to report on.")
    
    # Display generated report
    if 'generated_report' in st.session_state:
        st.markdown("---")
        st.markdown("### 📋 Generated Report")
        
        st.markdown(st.session_state.generated_report)
        
        # Download options
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.download_button(
                label="📥 Download as Markdown",
                data=st.session_state.generated_report,
                file_name=f"ab_testing_report_{datetime.now().strftime('%Y%m%d')}.md",
                mime="text/markdown"
            )
        
        with col2:
            st.download_button(
                label="📊 Export to CSV",
                data=pd.DataFrame(st.session_state.experiments).to_csv(index=False),
                file_name=f"experiment_data_{datetime.now().strftime('%Y%m%d')}.csv",
                mime="text/csv"
            )
        
        with col3:
            if st.button("📊 Create Dashboard"):
                st.info("Dashboard would be created here")
    
    # Learning database
    st.markdown("---")
    st.markdown("### 🧠 AI Learning Database")
    
    if st.session_state.experiments:
        # Create a vector of learnings
        learnings = []
        
        for exp in st.session_state.experiments:
            if exp.get('result') and exp.get('hypothesis'):
                learning = {
                    'hypothesis': exp.get('hypothesis'),
                    'result': exp.get('result'),
                    'uplift': exp.get('uplift', 0),
                    'key_insight': f"Test showed {exp.get('result')} with {exp.get('uplift', 0)}% uplift"
                }
                learnings.append(learning)
        
        if learnings:
            learnings_df = pd.DataFrame(learnings)
            st.dataframe(learnings_df, use_container_width=True)
            
            # AI-powered insights from learnings
            if st.button("🔍 Extract Patterns with AI"):
                with st.spinner("Gemini is analyzing patterns across experiments..."):
                    try:
                        insights_prompt = f"""
                        Analyze these A/B test results and extract patterns:
                        {json.dumps(learnings, indent=2)}
                        
                        Provide insights on:
                        1. What types of tests tend to succeed?
                        2. What common factors lead to failure?
                        3. Optimal sample sizes based on results
                        4. Recommended testing strategies
                        5. Areas for deeper investigation
                        """
                        
                        response = st.session_state.gemini_agent.model_pro.generate_content(insights_prompt)
                        st.markdown("#### AI-Generated Insights")
                        st.markdown(response.text)
                        
                    except Exception as e:
                        st.error(f"Error generating insights: {e}")
    else:
        st.info("Run experiments to build a learning database.")

# Footer
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center; color: #666; font-size: 0.9rem;'>
    <p>🧪 Autonomous A/B Testing Platform v1.0 • Powered by Gemini 3 AI</p>
    <p>Built with Streamlit • All data is simulated for demonstration</p>
    </div>
    """,
    unsafe_allow_html=True
)
