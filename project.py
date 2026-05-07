import pandas as pd
import numpy as np
import streamlit as st
import plotly.graph_objects as go
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.model_selection import train_test_split

# --- 1. SETTINGS & DATA LOADING ---
st.set_page_config(page_title="IE Production Dashboard", layout="wide")
CSV_FILE = 'hybrid_manufacturing_categorical.csv'

def load_and_process():
    df = pd.read_csv(CSV_FILE)
    df.columns = df.columns.str.strip()
    
    # Based on your screenshot, we will predict Processing_Time
    target = 'Processing_Time'
    
    if target not in df.columns:
        st.error(f"Target '{target}' not found. Using {df.columns[4]} instead.")
        target = df.columns[4]

    # Engineering features: Lag and Rolling Mean
    df['time_lag1'] = df[target].shift(1)
    df['rolling_avg_5'] = df[target].rolling(5).mean()
    df.dropna(inplace=True)
    return df, target

try:
    df, target_col = load_and_process()

    # --- 2. MODEL PREPARATION ---
    # We need to turn text (Job_Status, Material) into numbers
    cols_to_drop = [target_col, 'Job_ID', 'Scheduled_Start', 'Scheduled_End', 'Actual_Start', 'Actual_End']
    X = df.drop(columns=[c for c in cols_to_drop if c in df.columns])
    X = pd.get_dummies(X) # One-hot encoding for categories
    
    y = df[target_col]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
    model = LinearRegression()
    model.fit(X_train, y_train)
    preds = model.predict(X_test)

    # --- 3. DASHBOARD UI ---
    st.title("🏭 Industrial Production Admin Dashboard")
    st.markdown(f"### Monitoring: {target_col.replace('_', ' ')}")
    
    st.sidebar.header("Plant Manager Controls")
    sigma_val = st.sidebar.slider("Alert Sensitivity (Sigma)", 1.0, 4.0, 3.0)
    
    # Metrics Scorecard
    r2 = r2_score(y_test, preds)
    rmse = np.sqrt(mean_squared_error(y_test, preds))
    c1, c2, c3 = st.columns(3)
    c1.metric("Prediction Accuracy (R²)", f"{r2:.2%}")
    c2.metric("Avg Time Error", f"{rmse:.2f} min")

    # SPC Alert Logic
    residuals = y_test - preds
    ucl = residuals.mean() + (sigma_val * residuals.std())
    lcl = residuals.mean() - (sigma_val * residuals.std())
    alerts = np.where((residuals > ucl) | (residuals < lcl))[0]
    c3.metric("System Alerts", f"{len(alerts)} Points", 
              delta="Downtime Risk" if len(alerts) > 0 else "Optimal", 
              delta_color="inverse")

    # Interactive Control Chart
    st.subheader("Process Control Chart (Variance Analysis)")
    fig = go.Figure()
    fig.add_trace(go.Scatter(y=residuals, mode='lines+markers', name='Process Error', line=dict(color='#00CC96')))
    fig.add_hline(y=ucl, line_dash="dash", line_color="red", annotation_text="Upper Limit")
    fig.add_hline(y=lcl, line_dash="dash", line_color="red", annotation_text="Lower Limit")
    fig.update_layout(template="plotly_dark", height=450)
    st.plotly_chart(fig, use_container_width=True)

    # Export for Midterm Requirement
    if st.sidebar.button("Download Executive Report"):
        csv = df.to_csv(index=False).encode('utf-8')
        st.sidebar.download_button("Click to Download", data=csv, file_name="IE_Midterm_Report.csv")

except Exception as e:
    st.error(f"Setup Error: {e}")