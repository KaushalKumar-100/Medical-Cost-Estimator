import streamlit as st
import pandas as pd
import pickle
import requests

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Medical Cost Estimator",
    page_icon="🏥",
    layout="centered",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=DM+Serif+Display&display=swap');

  html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

  .stApp { background: linear-gradient(135deg, #0f1923 0%, #1a2a3a 50%, #0f1923 100%); }

  .hero-card {
    background: linear-gradient(135deg, #1e3a5f 0%, #0d2137 100%);
    border: 1px solid rgba(99,179,237,0.25);
    border-radius: 20px;
    padding: 2.5rem 2.5rem 2rem;
    margin-bottom: 2rem;
    text-align: center;
    box-shadow: 0 20px 60px rgba(0,0,0,0.4);
  }
  .hero-badge {
    display: inline-block;
    background: rgba(99,179,237,0.15);
    border: 1px solid rgba(99,179,237,0.4);
    color: #63b3ed;
    font-size: 0.72rem;
    font-weight: 600;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    padding: 0.3rem 0.9rem;
    border-radius: 100px;
    margin-bottom: 1.1rem;
  }
  .hero-title {
    font-family: 'DM Serif Display', serif;
    font-size: 2.4rem;
    color: #e8f4fd;
    line-height: 1.2;
    margin: 0 0 0.6rem;
  }
  .hero-title span { color: #63b3ed; }
  .hero-subtitle {
    color: #7fa8c4;
    font-size: 0.95rem;
    font-weight: 400;
    margin: 0;
  }

  /* Info box explaining what is being predicted */
  .info-box {
    background: rgba(99,179,237,0.07);
    border-left: 3px solid #63b3ed;
    border-radius: 0 10px 10px 0;
    padding: 0.9rem 1.2rem;
    margin-bottom: 1.5rem;
    font-size: 0.85rem;
    color: #90c4e0;
    line-height: 1.6;
  }
  .info-box strong { color: #63b3ed; }

  .section-label {
    font-size: 0.7rem;
    font-weight: 700;
    letter-spacing: 0.14em;
    text-transform: uppercase;
    color: #4a90a4;
    margin: 1.6rem 0 0.8rem;
  }

  .input-panel {
    background: rgba(255,255,255,0.04);
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 14px;
    padding: 1.5rem 1.5rem 0.5rem;
    margin-bottom: 1rem;
  }

  /* Result card */
  .result-card {
    background: linear-gradient(135deg, #0d3b2e 0%, #0a2a20 100%);
    border: 1px solid rgba(72,187,120,0.35);
    border-radius: 16px;
    padding: 1.8rem 2rem;
    margin-top: 1.5rem;
    text-align: center;
    box-shadow: 0 8px 32px rgba(72,187,120,0.15);
  }
  .result-label {
    font-size: 0.72rem;
    font-weight: 600;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    color: #48bb78;
    margin-bottom: 0.4rem;
  }
  .result-amount {
    font-family: 'DM Serif Display', serif;
    font-size: 3rem;
    color: #f0fff4;
    line-height: 1;
    margin-bottom: 0.25rem;
  }
  .result-unit { font-size: 1rem; color: #68d391; font-weight: 500; }
  .result-note { font-size: 0.8rem; color: #4a7c5f; margin-top: 0.6rem; }

  /* What this means breakdown */
  .breakdown-card {
    background: rgba(255,255,255,0.03);
    border: 1px solid rgba(255,255,255,0.07);
    border-radius: 12px;
    padding: 1rem 1.4rem;
    margin-top: 0.8rem;
    font-size: 0.82rem;
    color: #7fa8c4;
    line-height: 1.7;
  }
  .breakdown-card strong { color: #a0d0f0; }

  /* Converted result */
  .convert-card {
    background: rgba(99,179,237,0.08);
    border: 1px solid rgba(99,179,237,0.2);
    border-radius: 12px;
    padding: 1.1rem 1.5rem;
    margin-top: 1rem;
    text-align: center;
  }
  .convert-text { color: #a0d0f0; font-size: 1rem; font-weight: 500; }
  .convert-highlight { color: #63b3ed; font-weight: 700; font-size: 1.25rem; }

  div[data-testid="stSlider"] > label,
  div[data-testid="stSelectbox"] > label,
  div[data-testid="stNumberInput"] > label,
  div[data-testid="stTextInput"] > label { color: #a0bfd0 !important; font-size: 0.85rem !important; font-weight: 500 !important; }

  .stButton > button {
    width: 100%;
    background: linear-gradient(135deg, #2b6cb0, #1a4a7a);
    color: #e8f4fd;
    border: 1px solid rgba(99,179,237,0.4);
    border-radius: 10px;
    padding: 0.75rem 1rem;
    font-size: 1rem;
    font-weight: 600;
    letter-spacing: 0.02em;
    transition: all 0.2s ease;
    margin-top: 0.5rem;
  }
  .stButton > button:hover {
    background: linear-gradient(135deg, #3182ce, #2b6cb0);
    border-color: rgba(99,179,237,0.7);
    transform: translateY(-1px);
    box-shadow: 0 4px 20px rgba(49,130,206,0.4);
  }

  #MainMenu, footer, header { visibility: hidden; }
  .block-container { padding-top: 2rem; max-width: 780px; }
</style>
""", unsafe_allow_html=True)

# ── Load models ───────────────────────────────────────────────────────────────
@st.cache_resource
def load_models():
    with open('log_model.pkl', 'rb') as f:
        log_model = pickle.load(f)
    with open('retrain_model.pkl', 'rb') as f:
        retrain_model = pickle.load(f)
    with open('currency.pkl', 'rb') as f:
        currency = pickle.load(f)
    return log_model, retrain_model, currency

log_model, retrain_model, currency = load_models()

# ── Hero header ───────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero-card">
  <div class="hero-badge">AI-Powered Medical Cost Estimator</div>
  <h1 class="hero-title">Expected <span>Medical Charges</span> Predictor</h1>
  <p class="hero-subtitle">
    Enter your personal details to estimate the total healthcare costs your company
    is likely to cover — based on your health profile and lifestyle.
  </p>
</div>
""", unsafe_allow_html=True)

# ── What this predicts ────────────────────────────────────────────────────────
st.markdown("""
<div class="info-box">
  <strong>What does this predict?</strong><br>
  This tool estimates your expected <strong>annual medical charges</strong> — the total
  healthcare costs (hospital bills, treatments, procedures) that an company would
  likely pay out for someone with your profile and insurance policy. 
</div>
""", unsafe_allow_html=True)

# ── Input form ────────────────────────────────────────────────────────────────
st.markdown('<p class="section-label">Your Health Profile</p>', unsafe_allow_html=True)
with st.container():
    st.markdown('<div class="input-panel">', unsafe_allow_html=True)
    col1, col2 = st.columns(2)
    with col1:
        age = st.slider("Age", min_value=4, max_value=80, value=30)
        sex = st.selectbox("Gender", ["Male", "Female"])
        bmi = st.number_input(
            "BMI",
            min_value=0.0, max_value=100.0, value=25.0, step=0.1,
            help="Body Mass Index — weight (kg) ÷ height² (m²). A BMI of 18.5–24.9 is considered healthy."
        )
    with col2:
        children = st.slider("Number of Dependants", min_value=0, max_value=6, value=0)
        smoker = st.selectbox(
            "Smoker",
            ["No", "Yes"],
            help="Smoking significantly increases expected medical costs."
        )
        region = st.text_input(
            "Region",
            placeholder="e.g. northeast, southwest, southeast, northwest"
        )
    st.markdown('</div>', unsafe_allow_html=True)

st.markdown('<p class="section-label">Currency Conversion</p>', unsafe_allow_html=True)
target_currency = st.selectbox(
    "Convert result to",
    currency,
    help="The predicted charges are in INR. Select a currency to see the equivalent amount."
)

predict_btn = st.button("🔍  Estimate My Medical Charges")

# ── Logic ─────────────────────────────────────────────────────────────────────
def predict_charges(age, sex, bmi, children, smoker, region):
    df = pd.DataFrame([{
        "age": age,
        "sex": sex.lower(),
        "bmi": bmi,
        "children": children,
        "smoker": smoker.lower(),
        "region": region.lower().strip()
    }])
    result = log_model.predict(df)[0]
    return round(result*10, 2)

def convert_currency(amount_inr, target):
    try:
        response = requests.get("https://api.exchangerate-api.com/v4/latest/INR", timeout=5)
        if response.status_code == 200:
            rate = response.json()['rates'].get(target)
            if rate:
                return round(rate * amount_inr, 2)
        return None
    except Exception:
        return None

# ── Output ────────────────────────────────────────────────────────────────────
if predict_btn:
    if not region.strip():
        st.warning("⚠️  Please enter your region before estimating (e.g. northeast, southwest).")
    else:
        with st.spinner("Estimating medical charges…"):
            charges = predict_charges(age, sex, bmi, children, smoker, region)

        st.markdown(f"""
        <div class="result-card">
          <p class="result-label">Estimated Annual Medical Charges</p>
          <p class="result-amount">₹{charges:,.2f}</p>
          <p class="result-unit">Indian Rupees (INR)</p>
          <p class="result-note">
            This is the projected total healthcare cost — the amount your insurer
            would likely pay out based on your profile. Actual costs may vary.
          </p>
        </div>
        
        """, unsafe_allow_html=True)

        converted = convert_currency(charges, target_currency)
        if converted is not None:
            st.markdown(f"""
            <div class="convert-card">
              <p class="convert-text">
                ₹{charges:,.2f} INR &nbsp;≈&nbsp;
                <span class="convert-highlight">{converted:,.2f} {target_currency}</span>
              </p>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.info("Currency conversion is currently unavailable. Please check your internet connection and try again.")


        st.markdown(f"""
         <div class="breakdown-card">
          <strong>What this figure represents:</strong><br>
          This estimate covers expected medical expenses such as hospitalisation, treatments,
          and procedures — <em>not</em> the premium you pay for your policy. For context,
          a policyholder who pays ₹15,000/year as a premium might incur ₹3,00,000 or more
          in actual medical charges during a serious illness.
        </div>
        """, unsafe_allow_html=True)
