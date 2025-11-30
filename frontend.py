import streamlit as st
import requests
import pandas as pd
from rag_pipeline import initialize_rag_pipeline, call_rag_pipeline, init_session_memory, add_to_chat_history, get_chat_history
import os
from dotenv import load_dotenv

load_dotenv()
API_URL = "http://localhost:8000/predict"

st.set_page_config(
    page_title="Ingredient Risk Classifier",
    page_icon="🍪",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        color: #FF4B4B;
        text-align: center;
        font-weight: bold;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #333333;
        text-align: center;
        margin-bottom: 2rem;
    }
    .st-Button button {
        background-color: #FF4B4B;
        color: white;
        border-radius: 5px;
        border: none;
        padding: 0.5rem 1rem;
        font-weight: bold;
    }
    .st-Button button:hover {
        background-color: #FF6B6B;
        transform: scale(1.02);
    }
    </style>
    """, unsafe_allow_html=True)

# Initialize session memory for RAG pipeline
init_session_memory()

# Initialize session state for classification results
if "classification_result" not in st.session_state:
    st.session_state.classification_result = None

st.markdown("<p class='main-header'>🍪 Ingredient Risk Classifier</p>", unsafe_allow_html=True)
st.markdown("<p class='sub-header'>Analyze food ingredients and understand their safety levels</p>", unsafe_allow_html=True)

# Input section
st.subheader("Enter Ingredients")
default_example = "refined wheat flour, sugar, edible vegetable oil (palmolein), emulsifier (322), synthetic food colour (INS 133)"
text = st.text_area(
    "Ingredients",
    value=default_example,
    height=150,
    help="Paste ingredients from a label, e.g. 'wheat flour, sugar, palm oil, emulsifier (322)'."
)

# Analyze button
analyze_btn = st.button("🔍 Analyze Ingredients", type="primary", use_container_width=True)

# Information columns
col1, col2 = st.columns([1, 1])
with col1:
    st.subheader("How it works")
    st.info("""
    1. **Classify**: Our AI model analyzes your ingredients
    2. **Explain**: Get detailed explanations about risk levels
    3. **Understand**: Learn how to make safer choices
    """)

with col2:
    st.subheader("Risk Levels")
    st.markdown("""
    - 🟢 **1-2**: Very Safe/Safe - Natural ingredients
    - 🟡 **3**: Moderate - Refined but generally safe
    - 🟠 **4**: Concerning - Artificial additives
    - 🔴 **5**: High Risk - Potentially harmful substances
    """)

# Initialize RAG pipeline (cached)
@st.cache_resource
def get_rag_pipeline():
    try:
        return initialize_rag_pipeline()
    except Exception as e:
        st.warning(f"⚠️ RAG pipeline initialization failed: {e}")
        return None

# Display results function
def show_result(data: dict) -> None:
    """Render the model result in the Streamlit app."""
    try:
        risk_level = data.get("risk_level", "Unknown")
        risk_category = data.get("risk_category", "Unknown")
        probs = data.get("probabilities", {})
        
        # Color mapping for risk levels
        color_map = {
            1: "🟢",
            2: "🟢",
            3: "🟡",
            4: "🟠",
            5: "🔴",
        }
        icon = color_map.get(risk_level, "❓")
        
        st.subheader("Analysis Result")
        st.info(f"{icon} **Risk Level {risk_level}** - {risk_category}")
        
        # Map probabilities from labels 0-4 to risk levels 1-5
        risk_level_mapping = {0: 1, 1: 2, 2: 3, 3: 4, 4: 5}
        mapped_probs = {}
        for label_str, prob_value in probs.items():
            label_int = int(label_str)
            risk_lvl = risk_level_mapping.get(label_int, label_int + 1)
            mapped_probs[f"Risk Level {risk_lvl}"] = prob_value
        
        st.subheader("Probability Distribution")
        df_probs = pd.DataFrame(
            [(k.replace("Risk Level ", ""), v) for k, v in mapped_probs.items()],
            columns=["Risk Level", "Probability"]
        )
        st.bar_chart(df_probs.set_index("Risk Level"), height=300)
        st.caption("Model: Hacktrix-121deberta-v3-base-ingredients")
        
    except Exception as e:
        st.error(f"Error displaying results: {str(e)}")

# Main analysis logic
if analyze_btn:
    if not text.strip():
        st.warning("⚠️ Please enter some ingredients.")
    else:
        with st.spinner("🔄 Analyzing ingredients..."):
            try:
                # Make API request
                response = requests.post(
                    API_URL,
                    json={"text": text},
                    timeout=30
                )
                
                # Check if request was successful
                if response.status_code != 200:
                    st.error(f"❌ API Error {response.status_code}: {response.text}")
                else:
                    data = response.json()
                    
                    # Check for errors in response
                    if data.get("error"):
                        st.error(f"❌ Classification Error: {data['error']}")
                    else:
                        # Store classification result and display
                        st.session_state.classification_result = data
                        show_result(data)
                        
                        # Try to get RAG explanation
                        try:
                            with st.spinner("✨ Generating detailed explanation..."):
                                rag_chain = get_rag_pipeline()
                                if rag_chain:
                                    rag_response = call_rag_pipeline(rag_chain, text, data)
                                    # Check if we got a valid response
                                    if rag_response and not rag_response.startswith("Could not generate"):
                                        st.subheader("Concise Ingredient Analysis")
                                        st.success(rag_response)
                                        st.caption("✅ Concise responses with key information for each ingredient")
                                    else:
                                        st.warning("⚠️ Could not generate detailed explanation. " + (rag_response or "No response"))
                                else:
                                    st.warning("⚠️ RAG pipeline unavailable. Showing classification only.")
                        except Exception as e:
                            st.error(f"❌ Error in RAG pipeline: {str(e)}")
                            
            except requests.exceptions.ConnectionError:
                st.error("❌ Cannot connect to API. Is the backend running at http://localhost:8000?")
            except requests.exceptions.Timeout:
                st.error("❌ API request timed out. Please try again.")
            except Exception as e:
                st.error(f"❌ Unexpected error: {str(e)}")
