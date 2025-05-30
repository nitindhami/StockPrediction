import streamlit as st
import pandas as pd
import numpy as np
import joblib  # to load saved model

# -------------------------
# 1. Load your trained model
# -------------------------
model = joblib.load("/Users/nitindhami/Desktop/@Projects/PythonProjects/StockPrediction/archive/nifty_model.pkl")  # Save your trained model as .pkl first

st.set_page_config(page_title="Nifty Opening Price Predictor", layout="centered")
st.title("🔮 Nifty Opening Price Predictor")
st.caption("Using past 5 days of Open and Close to forecast the next day's Open")

# -------------------------
# 2. User Input for last 5 days
# -------------------------

st.subheader("📅 Enter last 5 days of data (latest first):")

data = []
for i in range(5):
    st.markdown(f"### Day {i+1}")
    open_price = st.number_input(f"Open Price (Day {i+1})", key=f"open_{i}", format="%.2f")
    close_price = st.number_input(f"Close Price (Day {i+1})", key=f"close_{i}", format="%.2f")
    data.append((open_price, close_price))

# Validate input
if all(x[0] != 0 and x[1] != 0 for x in data):
    # Reverse to go oldest to newest
    data = list(reversed(data))

    # Create lag feature dictionary
    input_dict = {}
    for i in range(1, 6):
        input_dict[f'Open_lag_{i}'] = data[-i][0]
        input_dict[f'Close_lag_{i}'] = data[-i][1]

    input_df = pd.DataFrame([input_dict])

    # -------------------------
    # 3. Predict and Show
    # -------------------------
    if st.button("🚀 Predict Opening Price"):
        pred = model.predict(input_df)[0]
        st.success(f"Predicted Opening Price for Tomorrow: ₹{round(pred, 2)}")

else:
    st.warning("Please fill all 5 days of open and close prices to enable prediction.")
