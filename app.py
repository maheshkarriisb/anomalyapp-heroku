import streamlit as st
from multiapp import MultiApp
from apps import dataProfiling, comingSoon, anomalyDetection, modelPlayground

# Page layout - expand to full screen width
######################################
st.set_page_config(page_title='Anomaly Detection for Accounting',
                   layout='wide')


######################################
## Page Title and sub title
######################################
st.title("Fraud Detection for Accounting")
st.write("**Author: [ByteIQ](https://github.com/yasarc4)**")

app = MultiApp()

# Add all your application here
app.add_app("Data Profiling", dataProfiling.app)
app.add_app("Data Forensic", comingSoon.app)
app.add_app("Accounting Landscape", comingSoon.app)
app.add_app("Transaction Risk Scoring", comingSoon.app)
app.add_app("Anomaly Detection", anomalyDetection.app)
app.add_app("Model Playground", modelPlayground.app)
app.add_app("Forecast Model", comingSoon.app)

# The main app
app.run()
