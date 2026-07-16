import joblib
import pandas as pd
import streamlit as st
import plotly.graph_objects as go

st.set_page_config(
    page_title="Heart Disease Prediction",
    page_icon="❤️",
    layout="wide",
    initial_sidebar_state="auto"
)


def create_gauge_chart(probability):
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=probability,
        gauge={'axis': {'range': [0, 100]},
               'bar': {'color': "darkred"},
               'steps': [{'range': [0, 25], 'color': "green"},
                         {'range': [25, 50], 'color': "yellow"},
                         {'range': [50, 75], 'color': "orange"},
                         {'range': [75, 100], 'color': "red"}]}))
    return fig


# -------------------------------------------------
# Page Config
# -------------------------------------------------

st.markdown(
     """
      <div class="app-header">
        <div class="header-icon">❤️‍🩹</div>
        <div>
            <h1>🫀Heart Disease Prediction</h1>
               <p>AI-powered risk assessment</p>
        </div>
     </div>
</h3>
""",
     unsafe_allow_html=True
    )



    # -------------------------------------------------
# Custom CSS
# -------------------------------------------------
st.markdown("""
<style>
    .main {
        background-color: #0f172a;
    }

    .block-container {
    max-width: 100% !important;
    padding-top: 0.5rem !important;
    padding-left: 1rem !important;
    padding-right: 1rem !important;
}


    .title-text {
        font-size: 2.2rem;
        font-weight: 700;
        color: white;
        margin-bottom: 0.2rem;
    }

    .subtitle-text {
        font-size: 1rem;
        color: #cbd5e1;
        margin-bottom: 2rem;
    }

    .section-card {
        background-color: #1e293b;
        padding: 1.2rem;
        border-radius: 16px;
        margin-bottom: 1rem;
        border: 1px solid #334155;
    }

    .metric-card {
        background-color: #1e293b;
        padding: 1rem;
        border-radius: 16px;
        text-align: center;
        border: 1px solid #334155;
    }

    .result-good {
        color: #22c55e;
        font-weight: 700;
        font-size: 1.2rem;
    }

    .result-bad {
        color: #ef4444;
        font-weight: 700;
        font-size: 1.2rem;
    }

    .small-text {
        color: #cbd5e1;
        font-size: 0.9rem;
    }
    <style>
.section-card {
    background: #ffffff;
    border: 1px solid #eeeeee;
    border-radius: 14px;
    padding: 24px 26px;
    margin-bottom: 16px;
    box-shadow: 0 2px 8px rgba(0,0,0,0.04);
}

.section-title {
    font-size: 26px;
    font-weight: 800;
    color: #1f2937;
    margin-bottom: 4px;
}

.section-subtitle {
    font-size: 15px;
    color: #6b7280;
}

.danger-card {
    background: linear-gradient(135deg, #ef233c 0%, #d90429 100%);
    color: white;
    padding: 28px;
    border-radius: 14px;
    margin: 18px 0 22px 0;
    box-shadow: 0 8px 22px rgba(217, 4, 41, 0.25);
}

.success-card {
    background: linear-gradient(135deg, #22c55e 0%, #16a34a 100%);
    color: white;
    padding: 28px;
    border-radius: 14px; 
    margin: 18px 0 22px 0;
    box-shadow: 0 8px 22px rgba(22, 163, 74, 0.25);
}

.main-result-content {
    display: flex;
    align-items: center;
    gap: 22px;
}

.result-icon {
    width: 78px;
    height: 78px;
    min-width: 78px;
    border-radius: 50%;
    background: rgba(255, 255, 255, 0.95);
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 38px;
}

.result-title {
    font-size: 27px;
    font-weight: 800;
    margin-bottom: 4px;
}

.result-message {
    font-size: 15px;
    opacity: 0.95;
    margin-bottom: 18px;
}

.probability-box {
    background: rgba(255,255,255,0.15);
    padding: 10px 16px;
    border-radius: 10px;
    display: flex;
    justify-content: space-between;
    align-items: center;
    min-width: 300px;
}

.probability-box span {
    font-size: 14px;
}

.probability-box strong {
    font-size: 26px;
}

.small-status-card {
    background: #ffffff;
    border: 1px solid #eeeeee;
    border-radius: 14px;
    padding: 22px 18px;
    display: flex;
    align-items: center;
    gap: 16px;
    min-height: 120px;
    box-shadow: 0 2px 8px rgba(0,0,0,0.04);
    margin-bottom: 20px;
}

.small-icon {
    width: 58px;
    height: 58px;
    min-width: 58px;
    border-radius: 50%;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 30px;
}

.healthy-icon {
    background: #dcfce7;
}

.disease-icon {
    background: #fee2e2;
}

.small-label {
    font-size: 15px;
    color: #374151;
    margin-bottom: 3px;
}

.healthy-value {
    font-size: 28px;
    font-weight: 800;
    color: #22c55e;
}

.disease-value {
    font-size: 28px;
    font-weight: 800;
    color: #e11d48;
}

.small-caption {
    font-size: 13px;
    color: #6b7280;
}
</style>
""", unsafe_allow_html=True)


# -------------------------------------------------
# Load Model
# -------------------------------------------------
@st.cache_resource
def load_model():
    model = joblib.load("models/final_random_forest.pkl")
    return model


model = load_model()


# -------------------------------------------------
# Sidebar
# -------------------------------------------------
with st.sidebar:
    st.markdown("## ❤️ Streamlit")

    st.markdown(
        """
        <div class="nav-item active">🏠 Dashboard</div>
        <div class="nav-item">ⓘ About the Project</div>
        <div class="nav-item">▥ Model Information</div>
        """,
        unsafe_allow_html=True
    )

    st.divider()

    st.markdown("### ♡ About")
    st.write(
        "This application uses a machine learning model to predict "
        "the likelihood of heart disease based on clinical and "
        "demographic information."
    )

    st.divider()

    st.markdown("### 🛡️ Disclaimer")
    st.write(
        "This tool is for educational and informational purposes only "
        "and does not replace professional medical advice, diagnosis, "
        "or treatment."
    )

    st.markdown(
        """
        <div class="model-card">
            <span>🧠 &nbsp; Model</span>
            <p>Tuned Random Forest</p>
            <span class="trained-badge">Trained ✓</span>
        </div>
        """,
        unsafe_allow_html=True
    )


# -------------------------------------------------
# Header
# -------------------------------------------------
st.markdown('<div class="title-text">Heart Disease Prediction Dashboard</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="subtitle-text">Enter patient clinical information to estimate heart disease risk.</div>',
    unsafe_allow_html=True
)


# -------------------------------------------------
# Input Layout
# -------------------------------------------------
sex_options = {
    "Male": "M",
    "Female": "F"
}

chest_pain_options = {
    "Typical Angina": "TA",
    "Atypical Angina": "ATA",
    "Non-Anginal Pain": "NAP",
    "Asymptomatic": "ASY"
}

fasting_bs_options = {
    "No - Fasting Blood Sugar ≤ 120 mg/dl": 0,
    "Yes - Fasting Blood Sugar > 120 mg/dl": 1
}

resting_ecg_options = {
    "Normal": "Normal",
    "ST-T Wave Abnormality": "ST",
    "Left Ventricular Hypertrophy": "LVH"
}

exercise_angina_options = {
    "No": "N",
    "Yes": "Y"
}

st_slope_options = {
    "Upsloping": "Up",
    "Flat": "Flat",
    "Downsloping": "Down"
}

# تعریف دیتافریم ورودی بر اساس مقادیر دریافت شده از کاربر


col1, col2, col3 = st.columns([0.9, 0.9, 1])

with col1:
    with st.container(border=True):
        st.markdown("### 👥 Demographics & Symptoms")
        st.caption("Provide basic patient information")

        age = st.slider(
            "Age (years)",
            min_value=18,
            max_value=100,
            value=54
        )

        sex_label = st.selectbox(
            "Sex",
            list(sex_options.keys())
        )
        sex = sex_options[sex_label]

        cp_label = st.selectbox(
            "Chest Pain Type (CP)",
            list(chest_pain_options.keys())
        )
        cp = chest_pain_options[cp_label]

        st.info(
            "💡 **Tip:** Select the chest pain type that best "
            "describes the patient's condition."
        )
        st.markdown(
            """
            <div style="height: 43px;"></div>
            """,
            unsafe_allow_html=True
        )


with col2:
    with st.container(border=True):
        st.markdown("### 〽️ Clinical Measurements")
        st.caption("Enter clinical and test measurements")

        trestbps = st.number_input(
            "Resting Blood Pressure (mm Hg)",
            min_value=80,
            max_value=220,
            value=140
        )

        chol = st.number_input(
            "Serum Cholesterol (mg/dl)",
            min_value=100,
            max_value=600,
            value=240
        )

        fbs_label = st.selectbox(
            "Fasting Blood Sugar > 120 mg/dl",
            list(fasting_bs_options.keys())
        )
        fbs = fasting_bs_options[fbs_label]

        restecg_label = st.selectbox(
            "Resting ECG",
            list(resting_ecg_options.keys())
        )
        restecg = resting_ecg_options[restecg_label]

        thalach = st.number_input(
            "Maximum Heart Rate Achieved",
            min_value=60,
            max_value=250,
            value=130
        )

        # in col2
        exang_label = st.selectbox(
            label="Exercise Induced Angina",
            options=list(exercise_angina_options.keys()),
            key="exang_selectbox"
        )
        exang = exercise_angina_options[exang_label]

        # برای بخش ST Segment Slope
        slope_label = st.selectbox(
            label="ST Segment Slope",
            options=list(st_slope_options.keys()),
            key="slope_selectbox"
        )
        slope = st_slope_options[slope_label]
        oldpeak = st.number_input(
            "ST Depression (Oldpeak)",
            min_value=0.0,
            max_value=10.0,
            value=1.0,
            step=0.1
        )
        st.markdown(
            """
            <div style="height: 43px;"></div>
            """,
            unsafe_allow_html=True
        )

input_df = pd.DataFrame({
    'Age': [age],
    'Sex': [sex],
    'ChestPainType': [cp],
    'RestingBP': [trestbps],
    'Cholesterol': [chol],
    'FastingBS': [fbs],
    'RestingECG': [restecg],
    'MaxHR': [thalach],
    'ExerciseAngina': [exang],
    'Oldpeak': [oldpeak],
    'ST_Slope': [slope]
})

with col3:
    with st.container(border=True):
        st.markdown("### 🛡️ Diagnostic Output")
        st.caption("Model prediction and probability")

        if st.button("Predict", use_container_width=True):

            prediction = model.predict(input_df)[0]
            probabilities = model.predict_proba(input_df)[0]

            healthy_probability = probabilities[0] * 100
            disease_probability = probabilities[1] * 100

            if prediction == 1:
                result_title = "Heart Disease Detected"
                result_message = "This patient is highly likely to have heart disease."
                main_probability = disease_probability
                main_card_class = "danger-card"
                icon = "💔"
            else:
                result_title = "No Heart Disease Detected"
                result_message = "This patient is likely to be healthy."
                main_probability = healthy_probability
                main_card_class = "success-card"
                icon = "💚"

            st.markdown(f"""
                <div class="{main_card_class}">
                    <div class="main-result-content">
                        <div class="result-icon">{icon}</div>
                        <div>
                            <div class="result-title">{result_title}</div>
                            <div class="result-message">{result_message}</div>
                             <div class="probability-box">
                                 <span>Probability</span>
                                 <strong>{main_probability:.2f}%</strong>
                             </div>
                         </div>
                     </div>
                 </div>
            """, unsafe_allow_html=True)

            card1, card2 = st.columns(2)

            with card1:
                 st.markdown(f"""
                     <div class="small-status-card">
                         <div class="small-icon healthy-icon">💚</div>
                         <div>
                             <div class="small-label">Healthy</div>
                             <div class="healthy-value">{healthy_probability:.2f}%</div>
                             <div class="small-caption">Probability</div>
                         </div>
                     </div>                 """, unsafe_allow_html=True)

            with card2:
                 st.markdown(f"""
                     <div class="small-status-card">
                         <div class="small-icon disease-icon">💔</div>
                         <div>
                             <div class="small-label">Heart Disease</div>
                             <div class="disease-value">{disease_probability:.2f}%</div>
                             <div class="small-caption">Probability</div>
                         </div>
                     </div>
                 """, unsafe_allow_html=True)

            st.markdown("""
            <div style="
                font-size: 1.15rem;
                font-weight: 700;
                margin-top: 0.3rem;
                margin-bottom: -1.5rem;
            ">
                Prediction Probability Gauge
            </div>
            """, unsafe_allow_html=True)

            fig = create_gauge_chart(disease_probability)

            fig.update_layout(
                height=220,
                margin=dict(l=0, r=0, t=0, b=0),
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)"
            )

            st.plotly_chart(
                fig,
                use_container_width=True,
                key="diagnostic_gauge_chart"
            )

