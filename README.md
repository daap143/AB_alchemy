# 🧪 Autonomous A/B Testing Platform

An AI-powered platform that automates the entire A/B testing workflow using Google's Gemini 3 Pro API.

## 🚀 Features

### 🤖 **AI-Powered Intelligence**
- **Gemini 3 pro** generates data-driven hypotheses
- Automated experiment design with statistical rigor
- Intelligent result analysis and interpretation
- Natural language reports and insights

### 🔄 **Complete Workflow Automation**
1. **Hypothesis Generation** - AI creates testable hypotheses with expected impact
2. **Experiment Design** - Calculates sample size, duration, and variants
3. **Simulation** - Generates realistic synthetic user data to test your design
4. **Analysis** - Performs statistical tests (t-tests, Bayesian) and calculates business impact
5. **Reporting** - Generates professional markdown reports for stakeholders

### 📊 **Advanced Analytics**
- Real-time statistical analysis
- Interactive visualizations with Plotly (Funnel analysis, Heatmaps, Box plots)
- Conversion rate confidence intervals
- User segmentation analysis

## 🛠️ Installation

### Prerequisites
- Python 3.10+
- Gemini API key (get from [Google AI Studio](https://aistudio.google.com/))

### Setup

1. **Clone repository**
   ```bash
   git clone [https://github.com/daap143/AB_alchemy](https://github.com/daap143/AB_alchemy)
   cd AB_alchemy```
	
 * Install dependencies
   ```pip install -r requirements.txt```

 * Set up API Key
   * You can enter your API key directly in the application sidebar.
   * Alternatively, set it in your environment (optional):
     export GEMINI_API_KEY="your_api_key_here"

🏃‍♂️ Running the Application

Option 1: Local Run (Standard)
streamlit run app.py

Access the app at http://localhost:8501.

Option 2: Running on Colab/Cloud (with ngrok)
If you are running this in a cloud environment (like Google Colab) where you cannot access localhost directly, use ngrok to create a public tunnel.
 * Install pyngrok:
   ```pip install pyngrok```

 * Run with tunnel:You will need a free ngrok auth token from ngrok.com
```streamlit run app.py & npx localtunnel --port 8501```

   Note: If using localtunnel, you may need to enter the tunnel IP password found at https://loca.lt/mytunnelpassword.
📂 Project Structure
 * app.py: Main Streamlit application entry point
 * gemini_agent.py: Core logic for interacting with Google Gemini API
 * data_simulator.py: Generates realistic synthetic A/B test data
 * visualization.py: Plotly-based charting and dashboarding
 * api_integrations.py: Connectors for GA4, Mixpanel, etc. (Mock/Template)
⚠️ Troubleshooting
 * API Key Errors: Ensure you are using a valid key from Google AI Studio.
 * "Object of type Timestamp is not JSON serializable": This is fixed in the latest version by handling date serialization in gemini_agent.py.
 * JSONDecodeError: The agent now includes a robust JSON cleaner to handle Markdown formatting from the LLM.
🤝 Contributing
Contributions are welcome! Please feel free to submit a Pull Request.

