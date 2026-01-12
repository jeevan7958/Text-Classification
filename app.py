import streamlit as st
import pickle
import re
import os
import nltk
import numpy as np
import json
import time
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from scipy.sparse import hstack

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="Spam Detector",
    layout="wide"
)

#AUTO-TRAINING LOGIC
#Checks if model files exist. If not, triggers the training script.
required_files = ['model.pkl', 'vectorizer.pkl', 'scaler.pkl', 'metrics.json', 'confusion_matrix.png', 'feature_importance.png']

if not all(os.path.exists(f) for f in required_files):
    with st.spinner("First-time setup: Training Hybrid SVM Model & Generating Graphs... (This takes ~15 seconds)"):
        try:
            # Import the training function dynamically
            from spam_classifier import train_and_evaluate
            train_and_evaluate()
            st.success("Training Complete! Loading App...")
            time.sleep(1) # Small pause for UX
            st.rerun() # Refresh the app to load new files
        except Exception as e:
            st.error(f"Fatal Error during training: {e}")
            st.stop()

# --- 2. LOAD RESOURCES ---
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords', quiet=True)

@st.cache_resource
def load_resources():
    with open('model.pkl', 'rb') as f: m = pickle.load(f)
    with open('vectorizer.pkl', 'rb') as f: v = pickle.load(f)
    with open('scaler.pkl', 'rb') as f: s = pickle.load(f)
    return m, v, s

model, vectorizer, scaler = load_resources()

#HELPER FUNCTIONS
def clean_text(text):
    if not isinstance(text, str): return ""
    ps = PorterStemmer()
    stop_words = set(stopwords.words('english'))
    text = re.sub(r'[^a-zA-Z]', ' ', text).lower()
    words = text.split()
    words = [ps.stem(word) for word in words if word not in stop_words]
    return ' '.join(words)

def get_meta_features(text):
    if not isinstance(text, str): return [0, 0, 0, 0, 0]
    return [
        len(text),
        1 if re.search(r'http[s]?://|www\.', text) else 0,
        sum(1 for c in text if c.isupper()) / len(text) if len(text) > 0 else 0,
        text.count('!'),
        1 if re.search(r'[$£€]', text) else 0
    ]

#UI LAYOUT
st.title("Spam Detector")
#st.markdown("### Ardentix Assignment Submission")

if model is None:
    st.error("System Files Missing!")
    st.stop()

#TABS
tab1, tab2 = st.tabs(["Live Detector", "Model Performance"])

# TAB 1: The Detector
with tab1:
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("Test Suspicious Messages")
        
        # Form allows 'Enter' key submission and cleaner UI
        with st.form(key='analysis_form'):
            user_input = st.text_area("Message Content:", height=150, placeholder="Paste SMS text here (e.g., 'URGENT! Verify your account...')")
            submit_button = st.form_submit_button(label='Analyze Risk', type='primary')
        
        if submit_button:
            if user_input:
                # 1. Text Features (TF-IDF)
                txt_vec = vectorizer.transform([clean_text(user_input)])
                
                # 2. Meta Features (Scaled)
                meta_vec = scaler.transform(np.array([get_meta_features(user_input)]))
                
                # 3. Stack Features
                final_vec = hstack([txt_vec, meta_vec])
                
                # 4. Predict
                prediction = model.predict(final_vec)[0]
                probability = model.predict_proba(final_vec)[0][1] # Probability of Spam
                
                st.divider()
                
                if prediction == 1:
                    st.error(f"**SPAM DETECTED**")
                    st.write(f"**Confidence Level:** {probability:.1%}")
                    st.caption("Risk Factors: High urgency, suspicious patterns, or structural anomalies.")
                else:
                    st.success(f"**SAFE / HAM**")
                    st.write(f"**Safety Score:** {1-probability:.1%}")
                    st.caption("Message appears legitimate.")
            else:
                st.warning("Please enter some text.")

    with col2:
        st.info("ℹ**System Architecture**")
        st.markdown("""
        **Hybrid SVM Engine**
        
        This model analyzes:
        * **Semantic Content:** TF-IDF (Bi-grams)
        * **Structural Metadata:** * Caps Lock Ratio
            * URL Density
            * Symbol Frequency
        
        *Trained on 2012 UCI Data + 2024 Synthetic Injection.*
        """)

# TAB 2: The Graphs & Metrics
with tab2:
    st.header("Evaluation Metrics")
    st.write("Performance metrics on the held-out test dataset (20%).")
    
    # 1. Display Scorecards (Accuracy, Precision, Recall, F1)
    if os.path.exists('metrics.json'):
        with open('metrics.json', 'r') as f:
            metrics = json.load(f)
        
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Accuracy", f"{metrics['accuracy']:.1%}")
        m2.metric("Precision", f"{metrics['precision']:.1%}", help="Minimizes False Positives")
        m3.metric("Recall", f"{metrics['recall']:.1%}", help="Catches actual Spam")
        m4.metric("F1 Score", f"{metrics['f1']:.1%}", help="Balanced Metric")
    else:
        st.warning("Metrics file missing. Re-run training.")

    st.divider()
    
    # 2. Display Graphs
    col_a, col_b = st.columns(2)
    
    with col_a:
        st.subheader("Confusion Matrix")
        if os.path.exists("confusion_matrix.png"):
            st.image("confusion_matrix.png", caption="Test Set Accuracy Breakdown", use_column_width=True)
        else:
            st.warning("Graph not found.")
            
    with col_b:
        st.subheader("Feature Importance")
        if os.path.exists("feature_importance.png"):
            st.image("feature_importance.png", caption="Top 10 Predictors of Spam", use_column_width=True)
        else:
            st.warning("Graph not found.")