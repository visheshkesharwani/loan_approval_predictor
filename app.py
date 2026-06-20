import streamlit as st
import joblib
import numpy as np

# --- PAGE CONFIGURATION ---
st.set_page_config(page_title="Loan Approval Predictor", layout="wide", initial_sidebar_state="expanded")

# --- SESSION STATE INITIALIZATION (Memory for Sidebar) ---
if 'calc_amount' not in st.session_state:
    st.session_state.calc_amount = 250
if 'calc_term' not in st.session_state:
    st.session_state.calc_term = 120

# --- PROFESSIONAL CSS (DARK CURRENCY BACKGROUND & CLEAN UI) ---
st.markdown("""
    <style>
    .stApp { 
        background: linear-gradient(rgba(11, 15, 25, 0.90), rgba(11, 15, 25, 0.95)), 
                    url('https://images.unsplash.com/photo-1611974789855-9c2a0a7236a3?q=80&w=2070&auto=format&fit=crop') no-repeat center center fixed;
        background-size: cover; color: #e2e8f0; font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    }
    .premium-header {
        text-align: center; padding: 2rem; background: rgba(17, 24, 39, 0.8);
        border-radius: 12px; border: 1px solid #374151; box-shadow: 0 4px 6px rgba(0,0,0,0.3); margin-bottom: 2rem; margin-top: 2rem;
    }
    .premium-header h1 { color: #ffffff; font-weight: 700; font-size: 2.5rem; text-transform: uppercase; margin: 0; letter-spacing: 1.5px; }
    .premium-header p { color: #9ca3af; font-size: 1.1rem; margin-top: 8px; }
    div[data-testid="stForm"] {
        background-color: rgba(17, 24, 39, 0.85); border: 1px solid #374151;
        border-radius: 12px; padding: 2rem; box-shadow: 0 10px 30px rgba(0,0,0,0.5);
    }
    .section-title {
        color: #60a5fa; font-size: 1.3rem; font-weight: 600; margin-top: 1.5rem;
        margin-bottom: 0.5rem; text-transform: uppercase; letter-spacing: 0.5px;
    }
    .divider { border: none; border-top: 1px solid #374151; margin-bottom: 1.5rem; }
    div.stButton > button:first-child {
        background-color: #2563eb; color: #ffffff; border-radius: 6px; padding: 12px 24px;
        font-weight: 600; font-size: 16px; border: none; width: 100%; transition: all 0.2s; text-transform: uppercase; letter-spacing: 1px;
    }
    div.stButton > button:first-child:hover { background-color: #1d4ed8; }
    .status-approved {
        background: rgba(5, 150, 105, 0.1); border: 1px solid #059669; color: #10b981; 
        padding: 20px; border-radius: 8px; text-align: center; font-size: 22px; font-weight: 700; margin-top: 20px;
    }
    .status-rejected {
        background: rgba(220, 38, 38, 0.1); border: 1px solid #dc2626; color: #ef4444; 
        padding: 20px; border-radius: 8px; text-align: center; font-size: 22px; font-weight: 700; margin-top: 20px;
    }
    .reason-box {
        background-color: rgba(31, 41, 55, 0.9); border-left: 4px solid #ef4444;
        padding: 12px; border-radius: 4px; margin-top: 12px; font-size: 15px; color: #fca5a5;
    }
    </style>
""", unsafe_allow_html=True)

# --- HEADER SECTION ---
st.markdown("""
    <div class="premium-header">
        <h1>LOAN APPROVAL PREDICTOR</h1>
        <p>Comprehensive Risk Assessment & Eligibility System</p>
    </div>
""", unsafe_allow_html=True)

# --- LOAD MODEL ---
try:
    model = joblib.load('MODEL/loan_model.pkl')
    scaler = joblib.load('MODEL/scaler.pkl')
except Exception as e:
    st.error(f"System Error: Application models missing. Details: {e}")
    st.stop()

# --- MAIN DASHBOARD FORM ---
with st.form("pro_loan_form"):
    
    st.markdown("<div class='section-title'>Applicant Demographics</div><hr class='divider'>", unsafe_allow_html=True)
    col1, col2, col3 = st.columns(3)
    with col1: gender = st.selectbox("Gender", ("Male", "Female"))
    with col2: married = st.selectbox("Marital Status", ("Yes", "No"))
    with col3: dependents = st.selectbox("Number of Dependents", ("0", "1", "2", "3+"))

    st.markdown("<div class='section-title'>Financial & Employment Profile</div><hr class='divider'>", unsafe_allow_html=True)
    col4, col5 = st.columns(2)
    with col4: 
        education = st.selectbox("Education Level", ("Graduate", "Not Graduate"))
        app_income = st.number_input("Applicant Monthly Income (Rs.)", min_value=0, value=40000, step=5000)
    with col5: 
        self_employed = st.selectbox("Employment Type", ("Salaried", "Self-Employed"))
        co_app_income = st.number_input("Co-Applicant Monthly Income (Rs.)", min_value=0, value=15000, step=5000)

    st.markdown("<div class='section-title'>Loan Specifications & Credit History</div><hr class='divider'>", unsafe_allow_html=True)
    
    col_purpose, col_existing = st.columns(2)
    with col_purpose: loan_purpose = st.selectbox("Purpose of Loan", ("Home Loan", "Personal Loan", "Vehicle Loan", "Business Loan"))
    with col_existing: existing_loans = st.selectbox("Any Existing Active Loans?", ("No", "Yes"))

    col6, col7 = st.columns(2)
    with col6:
        loan_amount = st.number_input("Requested Loan Amount (in Thousands)", min_value=10, value=250, step=10)
        property_area = st.selectbox("Property Location", ("Urban", "Semiurban", "Rural"))
    with col7:
        loan_amount_term = st.selectbox("Loan Tenure", (120, 180, 240, 360), index=3, format_func=lambda x: f"{x} Days ({x//30} Months)")
        cibil_score = st.slider("CIBIL Credit Score", min_value=300, max_value=900, value=720, step=10)

    st.markdown("<br>", unsafe_allow_html=True)
    submit_button = st.form_submit_button(label="Evaluate Application")

# --- SMART PREDICTION LOGIC ---
if submit_button:
    total_monthly_income = app_income + co_app_income
    estimated_emi = (loan_amount * 1000 * 0.008) 
    base_dti = (estimated_emi / total_monthly_income) * 100 if total_monthly_income > 0 else 100
    dti_ratio = base_dti + 15 if existing_loans == "Yes" else base_dti
    
    g_val = 1 if gender == "Male" else 0
    m_val = 1 if married == "Yes" else 0
    d_val = 3 if dependents == "3+" else int(dependents)
    e_val = 0 if education == "Graduate" else 1
    se_val = 1 if "Self-Employed" in self_employed else 0
    p_val = {"Rural": 0, "Semiurban": 1, "Urban": 2}[property_area]
    c_val = 1.0 if cibil_score >= 650 else 0.0

    features = np.array([[g_val, m_val, d_val, e_val, se_val, loan_amount, loan_amount_term, c_val, p_val, total_monthly_income]])
    
    scaled_features = scaler.transform(features)
    prediction = model.predict(scaled_features)
    probabilities = model.predict_proba(scaled_features)
    
    base_confidence = np.max(probabilities) * 100
    if cibil_score >= 750: final_probability = min(99.0, base_confidence + ((cibil_score - 750) * 0.05))
    elif cibil_score < 600: final_probability = max(15.0, base_confidence - ((600 - cibil_score) * 0.05))
    else: final_probability = base_confidence

    is_approved = (prediction[0] == 1 and cibil_score >= 650 and dti_ratio <= 50)

    # MAGIC TRICK: Agar approve hua, toh sidebar ke calculator ke variables update kar do
    if is_approved:
        st.session_state.calc_amount = int(loan_amount)
        st.session_state.calc_term = int(loan_amount_term // 30)

    st.markdown("<div class='section-title'>Live Application Metrics</div>", unsafe_allow_html=True)
    m1, m2, m3 = st.columns(3)
    m1.metric(label="Total Monthly Income", value=f"Rs. {total_monthly_income:,}")
    m2.metric(label="Est. Monthly EMI", value=f"Rs. {estimated_emi:,.0f}")
    m3.metric(label="Debt-to-Income (DTI) Ratio", value=f"{dti_ratio:.1f}%", delta="High Risk" if dti_ratio > 40 else "Standard", delta_color="inverse")
    
    st.markdown("<div class='section-title'>Application Status</div>", unsafe_allow_html=True)
    
    if is_approved:
        st.markdown(f"""
            <div class='status-approved'>
                STATUS: APPROVED<br>
                <span style='font-size:15px; color:#6ee7b7; font-weight:normal;'>Approval Probability: {final_probability:.1f}% | Profile Category: Standard</span>
            </div>
        """, unsafe_allow_html=True)
        st.balloons()
    else:
        st.markdown(f"""
            <div class='status-rejected'>
                STATUS: REJECTED<br>
                <span style='font-size:15px; color:#fca5a5; font-weight:normal;'>Approval Probability: {final_probability:.1f}% | Profile Category: High Risk</span>
            </div>
        """, unsafe_allow_html=True)
        
        reasons = []
        if cibil_score < 650: reasons.append("Credit Score Deficit: CIBIL score is below the operational threshold of 650.")
        if dti_ratio > 50: reasons.append(f"Excessive DTI Ratio: Calculated DTI is {dti_ratio:.1f}%. Bank policy mandates maximum 50%.")
        if app_income < 15000: reasons.append("Insufficient Income: Base monthly income does not meet minimum requirements.")
        if len(reasons) == 0: reasons.append("Internal Policy Criteria: Application does not align with regional risk parameters.")
        
        st.markdown("<p style='margin-top: 15px; font-weight: bold; color: #e2e8f0;'>Primary reasons for rejection:</p>", unsafe_allow_html=True)
        for reason in reasons:
            st.markdown(f"<div class='reason-box'>- {reason}</div>", unsafe_allow_html=True)

# --- SIDEBAR: EMI CALCULATOR (Auto-updates on Approval) ---
with st.sidebar:
    st.markdown("<h3 style='color: #60a5fa;'>Financial Calculator</h3>", unsafe_allow_html=True)
    st.markdown("Plan your repayment schedule.")
    
    # Ye inputs seedha session_state se jude hain. Form approve hote hi inki value badal jayegi!
    calc_amount = st.number_input("Loan Amount (Thousands)", min_value=10, step=10, key="calc_amount")
    calc_term = st.number_input("Term (Months)", min_value=12, step=12, key="calc_term")
    interest_rate = st.slider("Interest Rate (% p.a.)", 5.0, 15.0, 8.5, step=0.1)
    
    if calc_amount > 0 and calc_term > 0:
        p = calc_amount * 1000
        r = (interest_rate / 100) / 12
        n = calc_term
        emi = (p * r * ((1 + r)**n)) / (((1 + r)**n) - 1)
        st.success(f"Estimated EMI: Rs. {emi:,.2f} / month")
        
    st.markdown("---")
    st.markdown("<p style='color: #9ca3af; font-size: 14px;'>Note: Keep your Debt-to-Income (DTI) ratio below 40% for optimal approval chances.</p>", unsafe_allow_html=True)
