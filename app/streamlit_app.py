"""
Streamlit Web Application — College Admission Predictor (India)

A premium, interactive dashboard for predicting college admission probability.
Features validated inputs (sliders, dropdowns), color-coded results,
SHAP explanations, and prediction history.

Usage:
    streamlit run app/streamlit_app.py
"""

from __future__ import annotations

import sys
from pathlib import Path

# Resolve project root for imports
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import streamlit as st
import pandas as pd
import numpy as np

st.set_page_config(
    page_title="🎓 College Admission Predictor (India)",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ==============================================================================
# Custom CSS — Premium dark gradient theme
# ==============================================================================

st.markdown("""
<style>
    /* --- Main container --- */
    .stApp {
        background: linear-gradient(135deg, #0f0c29 0%, #302b63 50%, #24243e 100%);
    }

    /* --- Sidebar --- */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #1a1a2e 0%, #16213e 100%);
        border-right: 1px solid rgba(255,255,255,0.1);
    }

    [data-testid="stSidebar"] .stMarkdown h1,
    [data-testid="stSidebar"] .stMarkdown h2,
    [data-testid="stSidebar"] .stMarkdown h3 {
        color: #e0e0e0 !important;
    }

    /* --- Cards --- */
    .result-card {
        background: linear-gradient(135deg, rgba(255,255,255,0.1), rgba(255,255,255,0.05));
        backdrop-filter: blur(10px);
        border: 1px solid rgba(255,255,255,0.15);
        border-radius: 16px;
        padding: 2rem;
        margin: 1rem 0;
        text-align: center;
        transition: transform 0.3s ease, box-shadow 0.3s ease;
    }

    .result-card:hover {
        transform: translateY(-4px);
        box-shadow: 0 12px 40px rgba(0,0,0,0.3);
    }

    .result-high {
        border-left: 4px solid #00e676;
        box-shadow: 0 4px 20px rgba(0,230,118,0.2);
    }

    .result-medium {
        border-left: 4px solid #ffab00;
        box-shadow: 0 4px 20px rgba(255,171,0,0.2);
    }

    .result-low {
        border-left: 4px solid #ff1744;
        box-shadow: 0 4px 20px rgba(255,23,68,0.2);
    }

    .big-number {
        font-size: 4rem;
        font-weight: 800;
        margin: 0.5rem 0;
        text-shadow: 0 2px 10px rgba(0,0,0,0.3);
    }

    .confidence-label {
        font-size: 1.2rem;
        font-weight: 600;
        letter-spacing: 2px;
        text-transform: uppercase;
        margin-top: 0.5rem;
    }

    /* --- Header styling --- */
    .header-title {
        text-align: center;
        color: white;
        font-size: 2.5rem;
        font-weight: 800;
        margin-bottom: 0.5rem;
        background: linear-gradient(90deg, #667eea, #764ba2);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }

    .header-subtitle {
        text-align: center;
        color: rgba(255,255,255,0.7);
        font-size: 1.1rem;
        margin-bottom: 2rem;
    }

    /* --- Progress bar --- */
    .progress-container {
        background: rgba(255,255,255,0.1);
        border-radius: 10px;
        padding: 4px;
        margin: 1rem 0;
    }

    .progress-bar {
        height: 24px;
        border-radius: 8px;
        transition: width 0.8s ease;
    }

    /* --- Factor cards --- */
    .factor-positive {
        color: #00e676;
        font-weight: 600;
    }
    .factor-negative {
        color: #ff1744;
        font-weight: 600;
    }

    /* --- Hide Streamlit branding --- */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}

    /* --- Smooth animations --- */
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(20px); }
        to { opacity: 1; transform: translateY(0); }
    }
    .animated {
        animation: fadeIn 0.6s ease-out;
    }
</style>
""", unsafe_allow_html=True)


# ==============================================================================
# Helper functions
# ==============================================================================

@st.cache_resource
def load_model_pipeline():
    """Load the trained pipeline (cached)."""
    try:
        from src.predict import load_pipeline
        return load_pipeline()
    except Exception as e:
        st.error(f"⚠️ Model not found. Please run training first: `make train`\n\nError: {e}")
        return None


def get_color_class(chance: float) -> tuple[str, str, str]:
    """Return (css_class, color_hex, emoji) based on admission chance."""
    if chance >= 0.70:
        return "result-high", "#00e676", "🟢"
    elif chance >= 0.40:
        return "result-medium", "#ffab00", "🟡"
    else:
        return "result-low", "#ff1744", "🔴"


# ==============================================================================
# Sidebar — Input Form
# ==============================================================================

with st.sidebar:
    st.markdown("## 📋 Student Profile")
    st.markdown("---")

    # Academic Scores
    st.markdown("### 📚 Academic Scores")
    tenth = st.slider(
        "10th Percentage", min_value=0.0, max_value=100.0,
        value=75.0, step=0.5, key="tenth"
    )
    twelfth = st.slider(
        "12th Percentage", min_value=0.0, max_value=100.0,
        value=78.0, step=0.5, key="twelfth"
    )
    jee = st.slider(
        "JEE Percentile", min_value=0.0, max_value=100.0,
        value=85.0, step=0.1, key="jee"
    )
    cuet = st.number_input(
        "CUET Score", min_value=0, max_value=800,
        value=550, step=10, key="cuet"
    )
    cgpa = st.slider(
        "CGPA", min_value=0.0, max_value=10.0,
        value=8.0, step=0.1, key="cgpa"
    )

    st.markdown("---")
    st.markdown("### 👤 Personal Details")
    gender = st.selectbox("Gender", ["Male", "Female"], key="gender")
    category = st.selectbox(
        "Category", ["General", "OBC", "SC", "ST", "EWS"], key="category"
    )
    state = st.selectbox(
        "State",
        ["Maharashtra", "Delhi", "Karnataka", "Tamil_Nadu",
         "Uttar_Pradesh", "West_Bengal", "Rajasthan", "Gujarat",
         "Telangana", "Kerala"],
        key="state",
    )
    income = st.number_input(
        "Family Income (Lakhs/year)", min_value=0.1, max_value=200.0,
        value=8.0, step=0.5, key="income"
    )
    gap = st.selectbox("Gap Year(s)", [0, 1, 2], key="gap")

    st.markdown("---")
    st.markdown("### 🎯 College Preferences")
    branch = st.selectbox(
        "Desired Branch",
        ["CSE", "ECE", "ME", "CE", "EE", "IT", "Chemical", "Biotech"],
        key="branch",
    )
    tier = st.selectbox(
        "College Tier",
        ["Tier_1", "Tier_2", "Tier_3"],
        format_func=lambda x: {
            "Tier_1": "🏆 Tier 1 (IITs/NITs)",
            "Tier_2": "⭐ Tier 2 (Good State/Private)",
            "Tier_3": "📌 Tier 3 (Other)",
        }[x],
        key="tier",
    )

    st.markdown("---")
    st.markdown("### 🏅 Achievements")
    backlogs = st.number_input(
        "Number of Backlogs", min_value=0, max_value=50,
        value=0, step=1, key="backlogs"
    )
    extracurricular = st.toggle("Extracurricular Activities", value=True, key="extra")
    research = st.number_input(
        "Research Papers", min_value=0, max_value=20,
        value=0, step=1, key="research"
    )
    internship = st.number_input(
        "Internships", min_value=0, max_value=10,
        value=1, step=1, key="internship"
    )


# ==============================================================================
# Main Content
# ==============================================================================

# Header
st.markdown(
    '<div class="animated">'
    '<p class="header-title">🎓 College Admission Predictor</p>'
    '<p class="header-subtitle">'
    'AI-powered admission probability prediction for Indian students'
    '</p>'
    '</div>',
    unsafe_allow_html=True,
)

# Predict button
col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    predict_clicked = st.button(
        "🔮 Predict My Admission Chance",
        type="primary",
        use_container_width=True,
    )

if predict_clicked:
    # Build input dict
    input_data = {
        "Tenth_Percentage": tenth,
        "Twelfth_Percentage": twelfth,
        "JEE_Percentile": jee,
        "CUET_Score": float(cuet),
        "Category": category,
        "State": state,
        "Family_Income": income,
        "Gender": gender,
        "Gap_Year": gap,
        "CGPA": cgpa,
        "Backlogs": backlogs,
        "Extracurricular": int(extracurricular),
        "Research_Paper": research,
        "Internship": internship,
        "Desired_Branch": branch,
        "College_Tier": tier,
    }

    pipeline = load_model_pipeline()

    if pipeline is not None:
        try:
            from src.predict import predict_admission

            result = predict_admission(input_data, pipeline=pipeline)
            chance = result["admission_chance"]
            confidence = result["confidence_level"]
            css_class, color, emoji = get_color_class(chance)

            st.markdown("---")

            # Result display
            res_col1, res_col2, res_col3 = st.columns([1, 2, 1])
            with res_col2:
                st.markdown(
                    f'<div class="result-card {css_class} animated">'
                    f'<div style="color: rgba(255,255,255,0.7); font-size: 1rem;">'
                    f'Your Admission Chance</div>'
                    f'<div class="big-number" style="color: {color};">'
                    f'{chance * 100:.1f}%</div>'
                    f'<div class="confidence-label" style="color: {color};">'
                    f'{emoji} {confidence} Confidence</div>'
                    f'</div>',
                    unsafe_allow_html=True,
                )

            # Progress bar
            st.markdown(
                f'<div class="progress-container">'
                f'<div class="progress-bar" style="width: {chance * 100}%; '
                f'background: linear-gradient(90deg, {color}88, {color});"></div>'
                f'</div>',
                unsafe_allow_html=True,
            )

            # Detailed metrics
            st.markdown("### 📊 Prediction Details")
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("CGPA", f"{cgpa:.1f}/10")
            m2.metric("JEE Percentile", f"{jee:.1f}%")
            m3.metric("College Tier", tier.replace("_", " "))
            m4.metric("Branch", branch)

            # SHAP explanation (if available)
            st.markdown("---")
            st.markdown("### 🔍 Why This Prediction?")
            try:
                from src.explain import explain_prediction
                from src.pipeline_builder import get_feature_names_from_pipeline

                # Transform input through preprocessor
                input_df = pd.DataFrame([input_data])
                preprocessor = pipeline.named_steps["preprocessor"]
                X_transformed = preprocessor.transform(input_df)
                feature_names = get_feature_names_from_pipeline(pipeline)
                model = pipeline.named_steps["model"]

                contributions = explain_prediction(
                    model, X_transformed, feature_names
                )

                if contributions:
                    # Show top factors
                    top_factors = list(contributions.items())[:8]
                    factor_cols = st.columns(4)
                    for i, (feat, val) in enumerate(top_factors):
                        with factor_cols[i % 4]:
                            # Clean feature name
                            display_name = (
                                feat.replace("num__", "")
                                .replace("cat__", "")
                                .replace("ord__", "")
                                .replace("bin__", "")
                                .replace("_", " ")
                            )
                            sign = "+" if val > 0 else ""
                            css = "factor-positive" if val > 0 else "factor-negative"
                            st.markdown(
                                f'<div style="background: rgba(255,255,255,0.05); '
                                f'padding: 1rem; border-radius: 8px; margin: 0.5rem 0;">'
                                f'<div style="color: rgba(255,255,255,0.7); '
                                f'font-size: 0.8rem;">{display_name}</div>'
                                f'<div class="{css}" style="font-size: 1.3rem;">'
                                f'{sign}{val:.4f}</div>'
                                f'</div>',
                                unsafe_allow_html=True,
                            )
                else:
                    st.info("SHAP explanation not available for this model type.")

            except Exception as e:
                st.info(f"Explanation module not available: {e}")

        except Exception as e:
            st.error(f"Prediction failed: {e}")

else:
    # Default state — show instructions
    st.markdown("---")

    info_col1, info_col2, info_col3 = st.columns(3)
    with info_col1:
        st.markdown(
            '<div class="result-card animated" style="text-align: center;">'
            '<div style="font-size: 2.5rem; margin-bottom: 0.5rem;">📝</div>'
            '<div style="color: white; font-weight: 600; margin-bottom: 0.5rem;">'
            'Enter Your Profile</div>'
            '<div style="color: rgba(255,255,255,0.6); font-size: 0.9rem;">'
            'Fill in your academic scores and personal details in the sidebar'
            '</div></div>',
            unsafe_allow_html=True,
        )
    with info_col2:
        st.markdown(
            '<div class="result-card animated" style="text-align: center;">'
            '<div style="font-size: 2.5rem; margin-bottom: 0.5rem;">🤖</div>'
            '<div style="color: white; font-weight: 600; margin-bottom: 0.5rem;">'
            'AI Prediction</div>'
            '<div style="color: rgba(255,255,255,0.6); font-size: 0.9rem;">'
            'Our ML model analyzes 16 features to predict your admission chance'
            '</div></div>',
            unsafe_allow_html=True,
        )
    with info_col3:
        st.markdown(
            '<div class="result-card animated" style="text-align: center;">'
            '<div style="font-size: 2.5rem; margin-bottom: 0.5rem;">📊</div>'
            '<div style="color: white; font-weight: 600; margin-bottom: 0.5rem;">'
            'Explainable AI</div>'
            '<div style="color: rgba(255,255,255,0.6); font-size: 0.9rem;">'
            'Understand exactly which factors influence your prediction'
            '</div></div>',
            unsafe_allow_html=True,
        )

    st.markdown("---")
    st.markdown(
        '<div style="text-align: center; color: rgba(255,255,255,0.5); '
        'font-size: 0.9rem;">'
        '👈 Fill in your details in the sidebar and click '
        '<strong>Predict My Admission Chance</strong>'
        '</div>',
        unsafe_allow_html=True,
    )

# Footer
st.markdown("---")
st.markdown(
    '<div style="text-align: center; color: rgba(255,255,255,0.3); '
    'font-size: 0.8rem; padding: 1rem 0;">'
    'Built with ❤️ using Python, Scikit-learn, XGBoost, and Streamlit | '
    'Chance of Admission Prediction (India)'
    '</div>',
    unsafe_allow_html=True,
)
