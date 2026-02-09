# Save this as run_with_ngrok.py
from pyngrok import ngrok
import os

# Set your auth token (sign up at ngrok.com to get one - it's free)
ngrok.set_auth_token("your_auth_token") 

# Open a tunnel to port 8501
public_url = ngrok.connect(8501).public_url
print(f"🚀 Your App is live at: {public_url}")

# Run Streamlit
os.system("streamlit run app.py --server.port 8501")
