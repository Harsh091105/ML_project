import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(page_title="EV Battery Lifetime Predictor", layout="wide", page_icon="🔋")

# --- CUSTOM CSS ---
st.markdown("""
<style>
    .stApp {
        background-color: #0E1117;
    }
    h1, h2, h3, h4, span, p {
        font-family: 'Inter', sans-serif;
    }
</style>
""", unsafe_allow_html=True)

# Load Models
@st.cache_resource
def load_assets():
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    with open(os.path.join(BASE_DIR, 'models', 'best_model.pkl'), 'rb') as f:
        model = pickle.load(f)
    with open(os.path.join(BASE_DIR, 'models', 'training_metadata.pkl'), 'rb') as f:
        metadata = pickle.load(f)
    return model, metadata

try:
    model, metadata = load_assets()
except Exception as e:
    st.error("Models not found. Please run the backend training script first.")
    st.stop()

st.title("🔋 EV Battery Lifetime & Capacity Predictor")
st.markdown("### Next-generation embedded analytics for battery cycle forecasting.")

# -- SIDEBAR INFERENCE --
st.sidebar.header("Input Sensor Data")
feature_inputs = {}
for i, feature in enumerate(metadata['feature_names']):
    val = st.sidebar.number_input(f"Enter {feature}", value=0.0)
    feature_inputs[i] = val

if st.sidebar.button("Predict Capacity", use_container_width=True):
    input_arr = np.array(list(feature_inputs.values())).reshape(1, -1)
    
    # XGBoost and Scikit-learn expect slightly different pandas layouts. 
    # Try converting to exactly match features
    try:
        scaled_input = metadata['scaler'].transform(input_arr)
        scaled_df = pd.DataFrame(scaled_input, columns=metadata['feature_names'])
        prediction = model.predict(scaled_df)[0]
    except Exception:
        scaled_input = metadata['scaler'].transform(input_arr)
        prediction = model.predict(scaled_input)[0]
        
    st.sidebar.success(f"**Predicted Capacity:** {prediction:.4f} Ah")

# -- DASHBOARD GRAPHS --
col1, col2 = st.columns(2)

with col1:
    st.subheader("1. Battery Degradation Curve")
    deg_df = pd.DataFrame({
        'Actual Capacity': metadata['deg_actual'],
        'Predicted Capacity': metadata['deg_pred']
    })
    st.line_chart(deg_df)

with col2:
    st.subheader("2. Model Training Loss")
    loss_df = pd.DataFrame({
        'Train RMSE': metadata['train_loss'],
        'Validation RMSE': metadata['val_loss']
    })
    st.line_chart(loss_df)

st.markdown("---")
col3, col4 = st.columns(2)

with col3:
    st.subheader("3. Feature Importance Map")
    if metadata['feature_importance'] is not None:
        imp_df = pd.DataFrame({
            'Importance': np.abs(metadata['feature_importance'])
        }, index=metadata['feature_names']).sort_values(by='Importance', ascending=True)
        
        # Clean up the long dataset names into short, readable labels
        # (e.g., turns "Current_measured_mean" into "Current Mean")
        imp_df.index = imp_df.index.str.replace('_measured_', ' ').str.replace('_', ' ').str.title()
        
        st.bar_chart(imp_df)
    else:
        st.info("Feature importance not supported by the winning model type.")

with col4:
    st.subheader("4. Residual Distribution")
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.histplot(metadata['residuals'], bins=30, kde=True, color='#3399ff', ax=ax)
    ax.set_facecolor('#0E1117')
    fig.patch.set_facecolor('#0E1117')
    ax.tick_params(colors='white')
    ax.xaxis.label.set_color('white')
    ax.yaxis.label.set_color('white')
    ax.title.set_color('white')
    ax.set_title("Residual Error Frequency")
    st.pyplot(fig)
