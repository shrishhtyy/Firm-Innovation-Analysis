import streamlit as st
import numpy as np
import pandas as pd
from xgboost import XGBClassifier
from sklearn.preprocessing import LabelEncoder
from imblearn.over_sampling import SMOTE

st.set_page_config(page_title="Firm Innovation Analysis", page_icon="📊", layout="centered")

@st.cache_resource
def train_model():
    np.random.seed(42)
    n = 9000
    firm_size = np.random.randint(5, 500, n)
    firm_age = np.random.randint(2, 52, n)
    has_rd = np.random.choice([0,1], n, p=[0.75, 0.25])
    is_exporter = np.random.choice([0,1], n, p=[0.70, 0.30])
    has_training = np.random.choice([0,1], n, p=[0.60, 0.40])
    competition = np.random.choice([0,1,2,3,4,5], n, p=[0.05,0.15,0.25,0.30,0.15,0.10])
    foreign_owned = np.random.choice([0,1], n, p=[0.85, 0.15])
    quality_cert = np.random.choice([0,1], n, p=[0.70, 0.30])
    finance_obstacle = np.random.choice([0,1,2,3,4], n, p=[0.20,0.25,0.25,0.20,0.10])
    sector_labels = np.random.choice(['Manufacturing','Retail','Services'], n, p=[0.45,0.25,0.30])
    le = LabelEncoder()
    sector = le.fit_transform(sector_labels)

    score = (-4.0 + has_rd*4.0 + (firm_size>100)*2.5 + is_exporter*2.2 + has_training*1.5 +
             quality_cert*1.2 + foreign_owned*0.8 + (competition>=3)*0.5 + (finance_obstacle<=1)*0.4 +
             (sector_labels=='Manufacturing').astype(int)*0.3 + has_rd*is_exporter*1.0 + has_rd*(firm_size>100)*0.8)
    prob = 1 / (1 + np.exp(-(score + np.random.normal(0, 0.2, n))))
    innovator = (np.random.random(n) < prob).astype(int)

    df = pd.DataFrame({'firm_size': firm_size, 'firm_age': firm_age, 'is_exporter': is_exporter,
        'has_rd': has_rd, 'has_training': has_training, 'competition': competition,
        'foreign_owned': foreign_owned, 'quality_cert': quality_cert,
        'finance_obstacle': finance_obstacle, 'sector': sector})

    smote = SMOTE(random_state=42)
    X_sm, y_sm = smote.fit_resample(df, innovator)

    xgb = XGBClassifier(n_estimators=200, random_state=42, use_label_encoder=False, eval_metric='logloss')
    xgb.fit(X_sm, y_sm)

    importance = pd.Series(xgb.feature_importances_, index=df.columns).sort_values(ascending=False)
    return xgb, le, importance

model, le, importance = train_model()

st.title("📊 Firm Innovation Analysis")
st.markdown("**Will this firm innovate?** Enter firm details below and find out.")
st.divider()

col1, col2 = st.columns(2)
with col1:
    firm_size = st.slider("Number of Employees", 5, 500, 50)
    firm_age = st.slider("Firm Age (years)", 1, 50, 10)
    competition = st.slider("Number of Competitors", 0, 5, 2)
    finance_obstacle = st.select_slider("Finance Obstacle Level", options=[0,1,2,3,4],
        format_func=lambda x: ["None","Minor","Moderate","Major","Severe"][x])
with col2:
    has_rd = st.radio("Spends on R&D?", ["No","Yes"], horizontal=True)
    is_exporter = st.radio("Exports products?", ["No","Yes"], horizontal=True)
    has_training = st.radio("Provides employee training?", ["No","Yes"], horizontal=True)
    quality_cert = st.radio("Has quality certification?", ["No","Yes"], horizontal=True)
    foreign_owned = st.radio("Foreign ownership?", ["No","Yes"], horizontal=True)
    sector = st.selectbox("Sector", ["Manufacturing","Retail","Services"])

st.divider()

if st.button("🔍 Analyse", use_container_width=True, type="primary"):
    input_data = pd.DataFrame([{
        'firm_size': firm_size, 'firm_age': firm_age,
        'is_exporter': 1 if is_exporter=="Yes" else 0,
        'has_rd': 1 if has_rd=="Yes" else 0,
        'has_training': 1 if has_training=="Yes" else 0,
        'competition': competition,
        'foreign_owned': 1 if foreign_owned=="Yes" else 0,
        'quality_cert': 1 if quality_cert=="Yes" else 0,
        'finance_obstacle': finance_obstacle,
        'sector': le.transform([sector])[0]
    }])

    prediction = model.predict(input_data)[0]
    probability = model.predict_proba(input_data)[0]

    st.divider()
    if prediction == 1:
        st.success("### ✅ This firm is likely an INNOVATOR")
        st.metric("Innovation Probability", f"{probability[1]*100:.1f}%")
    else:
        st.error("### ❌ This firm is likely a NON-INNOVATOR")
        st.metric("Innovation Probability", f"{probability[1]*100:.1f}%")

    st.divider()
    st.subheader("What drives innovation?")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(8, 4))
    importance.sort_values().plot.barh(ax=ax, color='#3498db', edgecolor='black')
    ax.set_title('What Predicts Firm Innovation?', fontweight='bold')
    ax.set_xlabel('Importance')
    plt.tight_layout()
    st.pyplot(fig)

    st.markdown(f"""
    **Top 3 factors that drive innovation:**
    1. **{importance.index[0]}**: strongest factor
    2. **{importance.index[1]}**: second strongest
    3. **{importance.index[2]}**: third strongest
    """)
