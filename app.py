import streamlit as st
import pandas as pd
import numpy as np
from joblib import load
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

# ─── PAGE CONFIG ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="ChurnSense AI",
    page_icon="🔮",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─── GLOBAL CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>

@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;700;800&family=Space+Grotesk:wght@300;400;500&display=swap');

html, body, [class*="css"]{
    font-family:'Space Grotesk',sans-serif;
}

.stApp{
background:
radial-gradient(circle at top left,#4f46e5 0%,transparent 25%),
radial-gradient(circle at bottom right,#7c3aed 0%,transparent 25%),
linear-gradient(135deg,#050816,#0d1027,#111827);
background-size:400% 400%;
animation: gradientMove 15s ease infinite;
}

@keyframes gradientMove{
0%{background-position:0% 50%;}
50%{background-position:100% 50%;}
100%{background-position:0% 50%;}
}

header[data-testid="stHeader"]{
background:transparent;
}

#MainMenu, footer{
visibility:hidden;
}

[data-testid="stSidebar"]{
background:rgba(12,15,35,.75);
backdrop-filter:blur(25px);
border-right:1px solid rgba(255,255,255,.08);
}

.glass{
background:rgba(255,255,255,.05);
backdrop-filter:blur(18px);
border:1px solid rgba(255,255,255,.08);
border-radius:24px;
padding:22px;
box-shadow:
0 8px 32px rgba(0,0,0,.35),
0 0 30px rgba(99,102,241,.15);
transition:.4s;
}

.glass:hover{
transform:translateY(-8px);
box-shadow:
0 15px 45px rgba(99,102,241,.25),
0 0 50px rgba(139,92,246,.25);
}

.metric-card{
background:linear-gradient(
135deg,
rgba(99,102,241,.12),
rgba(139,92,246,.06)
);

backdrop-filter:blur(20px);
border-radius:24px;
padding:28px;
border:1px solid rgba(255,255,255,.08);

transition:.4s;
}

.metric-card:hover{
transform:scale(1.03);
box-shadow:0 0 35px rgba(99,102,241,.35);
}

.stButton>button{

height:65px;

background:
linear-gradient(
135deg,
#6366f1,
#7c3aed,
#9333ea
)!important;

background-size:300%;

animation:move 4s infinite;

border:none!important;
border-radius:18px!important;

font-size:18px!important;
font-weight:800!important;

box-shadow:
0 0 20px rgba(99,102,241,.4),
0 0 50px rgba(99,102,241,.25);

transition:.4s!important;
}

@keyframes move{
0%{background-position:0%}
100%{background-position:100%}
}

.stButton button:hover{
transform:translateY(-4px)!important;
box-shadow:
0 0 50px rgba(139,92,246,.7);
}

h1{
font-size:3rem!important;
font-weight:800!important;
color:white!important;
letter-spacing:-2px;
}

.stTabs [data-baseweb="tab"]{
background:rgba(255,255,255,.04);
border-radius:14px;
padding:10px;
margin-right:6px;
transition:.3s;
}

.stTabs [aria-selected="true"]{
background:linear-gradient(
90deg,
rgba(99,102,241,.4),
rgba(139,92,246,.3)
)!important;
box-shadow:0 0 25px rgba(99,102,241,.35);
}

</style>
""",unsafe_allow_html=True)

# ─── LOAD MODEL & DATA ─────────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    return load("logistic.pkl")

@st.cache_data
def load_data():
    df = pd.read_csv("customer_churn_prediction_dataset.csv")
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce').fillna(0)
    return df

model = load_model()
df = load_data()

FEATURE_COLS = list(model.feature_names_in_)

CATEGORICAL_COLS = {
    'gender': ['Male', 'Female'],
    'Partner': ['Yes', 'No'],
    'Dependents': ['Yes', 'No'],
    'PhoneService': ['Yes', 'No'],
    'MultipleLines': ['Yes', 'No', 'No phone service'],
    'InternetService': ['DSL', 'Fiber optic', 'No'],
    'OnlineSecurity': ['Yes', 'No', 'No internet service'],
    'OnlineBackup': ['Yes', 'No', 'No internet service'],
    'DeviceProtection': ['Yes', 'No', 'No internet service'],
    'TechSupport': ['Yes', 'No', 'No internet service'],
    'StreamingTV': ['Yes', 'No', 'No internet service'],
    'StreamingMovies': ['Yes', 'No', 'No internet service'],
    'Contract': ['Month-to-month', 'One year', 'Two year'],
    'PaperlessBilling': ['Yes', 'No'],
    'PaymentMethod': ['Electronic check', 'Mailed check', 'Bank transfer', 'Credit card'],
}


def build_feature_vector(inputs):
    row = {}
    row['SeniorCitizen'] = inputs['SeniorCitizen']
    row['tenure'] = inputs['tenure']
    row['MonthlyCharges'] = inputs['MonthlyCharges']
    row['TotalCharges'] = inputs['TotalCharges']
    for col, options in CATEGORICAL_COLS.items():
        val = inputs[col]
        for opt in options:
            col_name = f"{col}_{opt}"
            row[col_name] = 1 if val == opt else 0
    fv = pd.DataFrame([row])
    for fc in FEATURE_COLS:
        if fc not in fv.columns:
            fv[fc] = 0
    return fv[FEATURE_COLS]


def gauge_chart(prob):
    color = "#22c55e" if prob < 0.35 else "#f59e0b" if prob < 0.65 else "#ef4444"
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=prob * 100,
        number={'suffix': '%', 'font': {'size': 44, 'color': 'white', 'family': 'Syne'}},
        gauge={
            'axis': {'range': [0, 100], 'tickcolor': 'rgba(255,255,255,0.2)', 'tickfont': {'color': 'rgba(255,255,255,0.4)', 'size': 11}},
            'bar': {'color': color, 'thickness': 0.25},
            'bgcolor': 'rgba(0,0,0,0)',
            'borderwidth': 0,
            'steps': [
                {'range': [0, 35], 'color': 'rgba(34,197,94,0.08)'},
                {'range': [35, 65], 'color': 'rgba(245,158,11,0.08)'},
                {'range': [65, 100], 'color': 'rgba(239,68,68,0.08)'},
            ],
            'threshold': {'line': {'color': color, 'width': 4}, 'thickness': 0.85, 'value': prob * 100}
        }
    ))
    fig.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                      margin=dict(t=30, b=0, l=30, r=30), height=260, font={'color': 'white'})
    return fig


def feature_importance_chart():
    coefs = model.coef_[0]
    feat_imp = pd.DataFrame({'feature': FEATURE_COLS, 'coef': coefs})
    feat_imp['abs'] = feat_imp['coef'].abs()
    top = feat_imp.nlargest(15, 'abs').sort_values('coef')
    colors = ['#ef4444' if c > 0 else '#22c55e' for c in top['coef']]
    fig = go.Figure(go.Bar(x=top['coef'], y=top['feature'], orientation='h',
                           marker_color=colors, marker_line_width=0))
    fig.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                      xaxis=dict(title='Coefficient Weight', gridcolor='rgba(99,102,241,0.1)',
                                 color='rgba(255,255,255,0.4)', zerolinecolor='rgba(99,102,241,0.3)'),
                      yaxis=dict(color='rgba(255,255,255,0.7)', tickfont={'size': 11}),
                      margin=dict(t=10, b=40, l=10, r=10), height=420,
                      font={'color': 'white', 'family': 'DM Sans'})
    return fig


def churn_distribution_chart():
    churn_counts = df['Churn'].value_counts()
    fig = go.Figure(go.Pie(
        labels=['Retained', 'Churned'], values=churn_counts.values, hole=0.65,
        marker=dict(colors=['#6366f1', '#ef4444'], line=dict(width=0)),
        textfont={'color': 'white', 'size': 13, 'family': 'DM Sans'},
    ))
    fig.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                      showlegend=True, legend=dict(font=dict(color='rgba(255,255,255,0.7)', size=13)),
                      margin=dict(t=10, b=10, l=10, r=10), height=300)
    return fig


def tenure_churn_chart():
    temp = df.copy()
    temp['Churn_bin'] = (temp['Churn'] == 'Yes').astype(int)
    temp['tenure_group'] = pd.cut(temp['tenure'], bins=[0,12,24,36,48,60,72],
                                  labels=['0-12', '13-24', '25-36', '37-48', '49-60', '61-72'])
    grp = temp.groupby('tenure_group', observed=True)['Churn_bin'].mean().reset_index()
    fig = go.Figure(go.Bar(
        x=grp['tenure_group'], y=grp['Churn_bin'] * 100,
        marker=dict(color=grp['Churn_bin'] * 100,
                    colorscale=[[0, '#6366f1'], [0.5, '#8b5cf6'], [1, '#ef4444']], line_width=0),
        hovertemplate='%{x} months: %{y:.1f}% churn<extra></extra>',
    ))
    fig.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                      xaxis=dict(title='Tenure (months)', color='rgba(255,255,255,0.5)', gridcolor='rgba(99,102,241,0.1)'),
                      yaxis=dict(title='Churn Rate (%)', color='rgba(255,255,255,0.5)', gridcolor='rgba(99,102,241,0.1)'),
                      margin=dict(t=10, b=50, l=10, r=10), height=300,
                      font={'color': 'white', 'family': 'DM Sans'})
    return fig


# ─── HEADER ────────────────────────────────────────────────────────────────────
st.markdown("""
<div class='glass'>

<div style='display:flex;align-items:center;gap:20px;'>

<div style='
width:85px;
height:85px;
border-radius:24px;
background:linear-gradient(135deg,#6366f1,#8b5cf6);
display:flex;
align-items:center;
justify-content:center;
font-size:42px;
box-shadow:0 0 40px rgba(99,102,241,.6);
animation:pulse 2s infinite;
'>
🔮
</div>

<div>

<h1>
ChurnSense AI
</h1>

<div style='color:#9ca3af'>
Predict • Analyze • Retain Customers
</div>

</div>
</div>

</div>
""",unsafe_allow_html=True)

# ─── STATS BAR ─────────────────────────────────────────────────────────────────
churn_rate = (df['Churn'] == 'Yes').mean()
avg_tenure = df['tenure'].mean()
avg_monthly = df['MonthlyCharges'].mean()
total_customers = len(df)

c1, c2, c3, c4 = st.columns(4)
for col, icon, label, val in zip(
    [c1, c2, c3, c4],
    ['👥', '📉', '⏳', '💰'],
    ['Total Customers', 'Churn Rate', 'Avg Tenure', 'Avg Monthly Charge'],
    [f"{total_customers:,}", f"{churn_rate:.1%}", f"{avg_tenure:.0f} mo", f"${avg_monthly:.0f}"]
):
    col.markdown(f"""
    <div class="metric-card">
        <div style="font-size:28px;margin-bottom:4px;">{icon}</div>
        <div style="font-size:11px;color:#818cf8;text-transform:uppercase;letter-spacing:1px;margin-bottom:4px;">{label}</div>
        <div style="font-family:'Syne',sans-serif;font-size:26px;font-weight:700;color:white;">{val}</div>
    </div>
    """, unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ─── TABS ──────────────────────────────────────────────────────────────────────
tab1, tab2 = st.tabs(["🎯  Predict Churn", "📊  Dataset Insights"])

# ══════════════════════════════════════════════════
# TAB 1 — PREDICTION
# ══════════════════════════════════════════════════
with tab1:
    left, right = st.columns([1, 1.1], gap="large")

    with left:
        st.markdown('<div style="font-size:13px;color:#a5b4fc;text-transform:uppercase;letter-spacing:1px;margin-bottom:1.2rem;">📋 Customer Profile</div>', unsafe_allow_html=True)

        with st.expander("👤 Demographics", expanded=True):
            d1, d2 = st.columns(2)
            gender = d1.selectbox("Gender", ['Male', 'Female'])
            senior = d2.radio("Senior Citizen", ['No', 'Yes'], horizontal=True)
            senior_val = 1 if senior == 'Yes' else 0
            p1, p2 = st.columns(2)
            partner = p1.selectbox("Partner", ['Yes', 'No'])
            dependents = p2.selectbox("Dependents", ['Yes', 'No'])

        with st.expander("🗂️ Account Details", expanded=True):
            tenure = st.slider("Tenure (months)", 0, 72, 24)
            a1, a2 = st.columns(2)
            contract = a1.selectbox("Contract", ['Month-to-month', 'One year', 'Two year'])
            billing = a2.selectbox("Paperless Billing", ['Yes', 'No'])
            payment = st.selectbox("Payment Method", ['Electronic check', 'Mailed check', 'Bank transfer', 'Credit card'])
            ch1, ch2 = st.columns(2)
            monthly = ch1.number_input("Monthly Charges ($)", 0.0, 200.0, 65.0, step=0.5)
            total = ch2.number_input("Total Charges ($)", 0.0, 10000.0, float(tenure * monthly), step=10.0)

        with st.expander("📡 Services", expanded=True):
            s1, s2 = st.columns(2)
            phone = s1.selectbox("Phone Service", ['Yes', 'No'])
            multiline = s2.selectbox("Multiple Lines", ['Yes', 'No', 'No phone service'])
            internet = st.selectbox("Internet Service", ['DSL', 'Fiber optic', 'No'])
            s3, s4 = st.columns(2)
            online_sec = s3.selectbox("Online Security", ['Yes', 'No', 'No internet service'])
            online_bkp = s4.selectbox("Online Backup", ['Yes', 'No', 'No internet service'])
            s5, s6 = st.columns(2)
            device = s5.selectbox("Device Protection", ['Yes', 'No', 'No internet service'])
            tech = s6.selectbox("Tech Support", ['Yes', 'No', 'No internet service'])
            s7, s8 = st.columns(2)
            tv = s7.selectbox("Streaming TV", ['Yes', 'No', 'No internet service'])
            movies = s8.selectbox("Streaming Movies", ['Yes', 'No', 'No internet service'])

        st.markdown("<br>", unsafe_allow_html=True)
        predict_btn = st.button("🔮  Analyze Churn Risk", use_container_width=True)

    with right:
        st.markdown('<div style="font-size:13px;color:#a5b4fc;text-transform:uppercase;letter-spacing:1px;margin-bottom:1.2rem;">🔍 Prediction Results</div>', unsafe_allow_html=True)

        if predict_btn:
            inputs = {
                'SeniorCitizen': senior_val, 'tenure': tenure,
                'MonthlyCharges': monthly, 'TotalCharges': total,
                'gender': gender, 'Partner': partner, 'Dependents': dependents,
                'PhoneService': phone, 'MultipleLines': multiline,
                'InternetService': internet, 'OnlineSecurity': online_sec,
                'OnlineBackup': online_bkp, 'DeviceProtection': device,
                'TechSupport': tech, 'StreamingTV': tv, 'StreamingMovies': movies,
                'Contract': contract, 'PaperlessBilling': billing, 'PaymentMethod': payment,
            }
            fv = build_feature_vector(inputs)
            prob = model.predict_proba(fv)[0][1]

            if prob < 0.35:
                risk_label, risk_color, risk_bg, risk_emoji = "LOW RISK", "#22c55e", "rgba(34,197,94,0.1)", "✅"
            elif prob < 0.65:
                risk_label, risk_color, risk_bg, risk_emoji = "MEDIUM RISK", "#f59e0b", "rgba(245,158,11,0.1)", "⚠️"
            else:
                risk_label, risk_color, risk_bg, risk_emoji = "HIGH RISK", "#ef4444", "rgba(239,68,68,0.1)", "🚨"

            st.markdown(f"""
            <div style="background:{risk_bg};border:1.5px solid {risk_color};border-radius:16px;
                        padding:20px;margin-bottom:16px;text-align:center;">
                <div style="font-size:36px;margin-bottom:4px;">{risk_emoji}</div>
                <div style="font-family:'Syne',sans-serif;font-size:22px;font-weight:800;
                            color:{risk_color};letter-spacing:2px;">{risk_label}</div>
                <div style="color:rgba(255,255,255,0.5);font-size:13px;margin-top:4px;">
                    Churn probability: <span style="color:white;font-weight:600;">{prob:.1%}</span>
                </div>
            </div>
            """, unsafe_allow_html=True)

            st.plotly_chart(gauge_chart(prob), use_container_width=True, config={'displayModeBar': False})

            # Risk drivers
            coefs = model.coef_[0]
            fv_arr = fv.values[0]
            contributions = pd.DataFrame({'feature': FEATURE_COLS, 'contribution': coefs * fv_arr})
            top_risks = contributions.nlargest(5, 'contribution')
            top_protect = contributions.nsmallest(3, 'contribution')

            st.markdown('<div style="font-size:12px;color:#a5b4fc;text-transform:uppercase;letter-spacing:1px;margin-bottom:12px;">⚡ Key Risk Drivers</div>', unsafe_allow_html=True)
            for _, row in top_risks.iterrows():
                if row['contribution'] > 0.01:
                    bar_pct = min(abs(row['contribution']) / contributions['contribution'].abs().max() * 100, 100)
                    st.markdown(f"""
                    <div style="margin-bottom:10px;">
                        <div style="display:flex;justify-content:space-between;font-size:12px;color:rgba(255,255,255,0.7);margin-bottom:4px;">
                            <span>{row['feature'].replace('_', ' ')}</span>
                            <span style="color:#ef4444;">+{row['contribution']:.3f}</span>
                        </div>
                        <div style="background:rgba(255,255,255,0.05);border-radius:4px;height:6px;">
                            <div style="width:{bar_pct:.0f}%;height:100%;background:linear-gradient(90deg,#ef4444,#f87171);border-radius:4px;"></div>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)

            st.markdown('<div style="font-size:12px;color:#a5b4fc;text-transform:uppercase;letter-spacing:1px;margin:16px 0 12px;">🛡️ Retention Factors</div>', unsafe_allow_html=True)
            for _, row in top_protect.iterrows():
                if row['contribution'] < -0.01:
                    bar_pct = min(abs(row['contribution']) / contributions['contribution'].abs().max() * 100, 100)
                    st.markdown(f"""
                    <div style="margin-bottom:10px;">
                        <div style="display:flex;justify-content:space-between;font-size:12px;color:rgba(255,255,255,0.7);margin-bottom:4px;">
                            <span>{row['feature'].replace('_', ' ')}</span>
                            <span style="color:#22c55e;">{row['contribution']:.3f}</span>
                        </div>
                        <div style="background:rgba(255,255,255,0.05);border-radius:4px;height:6px;">
                            <div style="width:{bar_pct:.0f}%;height:100%;background:linear-gradient(90deg,#22c55e,#4ade80);border-radius:4px;"></div>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)

            if prob >= 0.65:
                recs = ["Offer loyalty discount or contract upgrade", "Assign dedicated account manager", "Proactive outreach within 48 hours"]
            elif prob >= 0.35:
                recs = ["Send personalized retention email", "Highlight unused premium features", "Monitor engagement closely"]
            else:
                recs = ["Continue standard engagement", "Consider upsell opportunities", "Schedule quarterly check-in"]

            st.markdown(f"""
            <div style="background:rgba(99,102,241,0.08);border:1px solid rgba(99,102,241,0.2);
                        border-radius:12px;padding:16px;margin-top:12px;">
                <div style="font-size:12px;color:#a5b4fc;text-transform:uppercase;letter-spacing:1px;margin-bottom:10px;">💡 Recommended Actions</div>
                {"".join(f'<div style="display:flex;gap:8px;margin-bottom:6px;font-size:13px;color:rgba(255,255,255,0.8);"><span>›</span><span>{r}</span></div>' for r in recs)}
            </div>
            """, unsafe_allow_html=True)

        else:
            st.markdown("""
            <div style="background:rgba(99,102,241,0.05);border:1px dashed rgba(99,102,241,0.25);
                        border-radius:16px;padding:60px 30px;text-align:center;margin-top:30px;">
                <div style="font-size:56px;margin-bottom:16px;opacity:0.4;">🔮</div>
                <div style="font-family:'Syne',sans-serif;font-size:18px;color:rgba(255,255,255,0.5);">
                    Fill in the customer profile<br>and hit <strong style="color:#818cf8;">Analyze Churn Risk</strong>
                </div>
            </div>
            <div style="margin-top:24px;background:rgba(99,102,241,0.06);border-radius:12px;padding:20px;">
                <div style="font-size:12px;color:#a5b4fc;letter-spacing:1px;text-transform:uppercase;margin-bottom:12px;">⚙️ Model Info</div>
                <div style="display:grid;grid-template-columns:1fr 1fr;gap:10px;">
                    <div style="font-size:12px;color:rgba(255,255,255,0.5);">Algorithm</div>
                    <div style="font-size:12px;color:white;">Logistic Regression</div>
                    <div style="font-size:12px;color:rgba(255,255,255,0.5);">Features</div>
                    <div style="font-size:12px;color:white;">45 encoded inputs</div>
                    <div style="font-size:12px;color:rgba(255,255,255,0.5);">Output</div>
                    <div style="font-size:12px;color:white;">Churn probability [0–1]</div>
                </div>
            </div>
            """, unsafe_allow_html=True)

# ══════════════════════════════════════════════════
# TAB 2 — INSIGHTS
# ══════════════════════════════════════════════════
with tab2:
    i1, i2 = st.columns(2)
    with i1:
        st.markdown('<div style="font-size:13px;color:#a5b4fc;text-transform:uppercase;letter-spacing:1px;margin-bottom:8px;">📊 Churn Distribution</div>', unsafe_allow_html=True)
        st.plotly_chart(churn_distribution_chart(), use_container_width=True, config={'displayModeBar': False})
    with i2:
        st.markdown('<div style="font-size:13px;color:#a5b4fc;text-transform:uppercase;letter-spacing:1px;margin-bottom:8px;">⏳ Churn Rate by Tenure</div>', unsafe_allow_html=True)
        st.plotly_chart(tenure_churn_chart(), use_container_width=True, config={'displayModeBar': False})

    st.markdown('<div style="font-size:13px;color:#a5b4fc;text-transform:uppercase;letter-spacing:1px;margin:16px 0 8px;">🏋️ Feature Importances (Top 15)</div>', unsafe_allow_html=True)
    st.plotly_chart(feature_importance_chart(), use_container_width=True, config={'displayModeBar': False})

    st.markdown('<div style="font-size:13px;color:#a5b4fc;text-transform:uppercase;letter-spacing:1px;margin:16px 0 8px;">📃 Churn by Contract Type</div>', unsafe_allow_html=True)
    contract_churn = df.groupby('Contract')['Churn'].apply(lambda x: (x=='Yes').mean() * 100).reset_index()
    contract_churn.columns = ['Contract', 'ChurnPct']
    fig_contract = go.Figure(go.Bar(
        x=contract_churn['Contract'], y=contract_churn['ChurnPct'],
        marker=dict(color=['#ef4444', '#f59e0b', '#22c55e'], line_width=0),
        text=[f"{v:.1f}%" for v in contract_churn['ChurnPct']],
        textposition='outside', textfont={'color': 'white', 'size': 13, 'family': 'Syne'},
    ))
    fig_contract.update_layout(
        paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
        xaxis=dict(color='rgba(255,255,255,0.5)', gridcolor='rgba(99,102,241,0.1)'),
        yaxis=dict(title='Churn Rate (%)', color='rgba(255,255,255,0.5)', gridcolor='rgba(99,102,241,0.1)'),
        height=300, margin=dict(t=30, b=10, l=10, r=10), font={'color': 'white', 'family': 'DM Sans'},
    )
    st.plotly_chart(fig_contract, use_container_width=True, config={'displayModeBar': False})

# ─── FOOTER ────────────────────────────────────────────────────────────────────
st.markdown("""
<div style="text-align:center;padding:32px 0 16px 0;color:rgba(255,255,255,0.2);
            font-size:12px;border-top:1px solid rgba(99,102,241,0.1);margin-top:40px;">
    ChurnSense AI · Logistic Regression Model · Customer Intelligence Platform
</div>
""", unsafe_allow_html=True)