# Deploy WolfSportsAI V6 to Streamlit Community Cloud

1. Create a GitHub repository and upload the contents of `WolfSportsAI_V6_CLOUD`.
2. Do NOT upload `.env` or a real `.streamlit/secrets.toml`.
3. Sign in to Streamlit Community Cloud with GitHub.
4. Create a new app and select the repository.
5. Main file path: `streamlit_app.py`
6. In Advanced settings / Secrets, add:
   ODDS_API_KEY = "YOUR_REAL_KEY"
7. Deploy.

The API key stays server-side and is not displayed to visitors.

Free hosting note:
Streamlit Community Cloud apps can sleep/restart. WolfSportsAI can automatically
bootstrap historical data and retrain while the app is awake, but this is not a
guaranteed always-on 24/7 worker. For true 24/7 scheduled training, use a paid
always-on host or an external scheduler/database.
