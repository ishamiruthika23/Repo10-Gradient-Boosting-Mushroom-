import streamlit as st
import joblib
import pandas as pd
import numpy as np

# Page configuration for a professional look
st.set_page_config(page_title="Mushroom Guard", page_icon="🍄", layout="centered")

# Custom CSS for "Pretty UI"
st.markdown("""
    <style>
    .main { background-color: #f8f9fa; }
    .stButton>button {
        width: 100%;
        border-radius: 20px;
        background-color: #4CAF50;
        color: white;
        font-weight: bold;
    }
    .result-box {
        padding: 25px;
        border-radius: 15px;
        text-align: center;
        margin-top: 20px;
        font-size: 24px;
        font-weight: bold;
    }
    </style>
    """, unsafe_allow_html=True)

# Load your specific model
@st.cache_resource
def load_model():
    return joblib.load("gb_mush.pkl")

model = load_model()

st.title("🍄 Mushroom Edibility Classifier")
st.markdown("Enter the physical characteristics of the mushroom below to predict if it is **Edible** or **Poisonous**.")

# Sidebar for Input Features
st.sidebar.header("Mushroom Features")

def get_user_input():
    # Numerical Inputs
    cap_diam = st.sidebar.slider("Cap Diameter (cm)", 0.0, 25.0, 5.0)
    stem_h = st.sidebar.slider("Stem Height (cm)", 0.0, 15.0, 3.0)
    stem_w = st.sidebar.slider("Stem Width (mm)", 0.0, 50.0, 10.0)
    
    # Categorical Inputs (Standardized based on mushroom datasets)
    cap_shape = st.sidebar.selectbox("Cap Shape", [0, 1, 2, 3, 4, 5, 6], format_func=lambda x: ["Bell", "Conical", "Convex", "Flat", "Sunken", "Spherical", "Others"][x])
    gill_attach = st.sidebar.selectbox("Gill Attachment", [0, 1, 2, 3, 4, 5, 6, 7], format_func=lambda x: ["Adnate", "Adnexed", "Decurrent", "Free", "Sinuate", "Pores", "None", "Unknown"][x])
    gill_color = st.sidebar.selectbox("Gill Color", [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11], format_func=lambda x: ["Black", "Brown", "Gray", "Pink", "White", "Yellow", "Green", "Purple", "Red", "Buff", "Chocolate", "Orange"][x])
    stem_color = st.sidebar.selectbox("Stem Color", [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12], format_func=lambda x: ["White", "Yellow", "Tan", "Gray", "Red", "Pink", "Buff", "Purple", "Cinnamon", "Green", "Orange", "Black", "Brown"][x])
    season = st.sidebar.selectbox("Season", [0, 1, 2, 3], format_func=lambda x: ["Spring", "Summer", "Autumn", "Winter"][x])

    features = {
        'cap_diameter': cap_diam,
        'cap_shape': cap_shape,
        'gill_attachment': gill_attach,
        'gill_color': gill_color,
        'stem_height': stem_h,
        'stem_width': stem_w,
        'stem_color': stem_color,
        'season': season
    }
    return pd.DataFrame([features])

input_df = get_user_input()

# Main Interface UI
col1, col2 = st.columns([1, 1])

with col1:
    st.info("**Current Selection**")
    st.write(input_df.T.rename(columns={0: 'Value'}))

with col2:
    st.warning("**Safety First!**")
    st.write("This AI model is for educational purposes. Never eat a mushroom you cannot 100% identify yourself.")

# Prediction Section
if st.button("Analyze Mushroom"):
    prediction = model.predict(input_df)
    probability = model.predict_proba(input_df)
    
    # Assuming 0 = Edible, 1 = Poisonous (Standard encoding)
    if prediction[0] == 0:
        st.markdown(f'<div class="result-box" style="background-color: #d4edda; color: #155724; border: 2px solid #c3e6cb;">'
                    f'✅ Result: EDIBLE<br><span style="font-size: 14px;">Confidence: {np.max(probability)*100:.2f}%</span></div>', unsafe_allow_html=True)
        st.balloons()
    else:
        st.markdown(f'<div class="result-box" style="background-color: #f8d7da; color: #721c24; border: 2px solid #f5c6cb;">'
                    f'🚨 Result: POISONOUS<br><span style="font-size: 14px;">Confidence: {np.max(probability)*100:.2f}%</span></div>', unsafe_allow_html=True)