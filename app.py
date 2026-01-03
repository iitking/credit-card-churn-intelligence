"""
Streamlit Web Dashboard for Credit Card Churn Intelligence.
Features real-time customer risk simulation, batch portfolio scoring,
interactive EDA, and financial ROI threshold tuning.
"""

import sys
import json
from pathlib import Path
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

# Ensure project root is in python path
ROOT_DIR = Path(__file__).resolve().parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from config.config import (
    DATA_PATH,
    METADATA_PATH,
    METRICS_SUMMARY_PATH,
    ORDINAL_CATEGORIES,
    NOMINAL_COLS,
    DEFAULT_DECISION_THRESHOLD,
    CUSTOMER_LIFETIME_VALUE,
    RETENTION_CAMPAIGN_COST,
    RETENTION_SUCCESS_RATE
)
from src.predict import ChurnPredictor
from src.data_loader import load_raw_data, clean_data

# Page configuration
st.set_page_config(
    page_title="Credit Card Churn Intelligence",
    page_icon="💳",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling
st.markdown("""
<style>
    .main-title {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1E293B;
        margin-bottom: 0.2rem;
    }
    .sub-title {
        font-size: 1.05rem;
        color: #64748B;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background-color: #F8FAFC;
        border-radius: 10px;
        padding: 1.2rem;
        border-left: 5px solid #3B82F6;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05);
    }
    .metric-value {
        font-size: 1.8rem;
        font-weight: 700;
        color: #0F172A;
    }
    .metric-label {
        font-size: 0.85rem;
        color: #64748B;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 48px;
        white-space: pre-wrap;
        border-radius: 6px 6px 0px 0px;
        padding-top: 10px;
        padding-bottom: 10px;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def get_predictor():
    """Cache the trained model predictor."""
    return ChurnPredictor()


@st.cache_data
def load_dataset():
    """Cache the raw dataset."""
    return load_raw_data(DATA_PATH)


# Header
st.markdown('<div class="main-title">💳 Credit Card Churn Intelligence Platform</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">AI-Powered Early Churn Warning, Explainability & Dynamic Customer Retention Engine</div>', unsafe_allow_html=True)

try:
    predictor = get_predictor()
    model_loaded = True
except Exception as e:
    model_loaded = False
    st.error(f"⚠️ Model not initialized yet: {e}. Please run training first or use training script.")

# Tabs
tab1, tab2, tab3, tab4 = st.tabs([
    "🎯 Customer Risk Simulator",
    "📊 Portfolio Batch Analytics",
    "📈 Exploratory Insights",
    "⚙️ Model Performance & ROI"
])

# ==========================================
# TAB 1: Real-time Customer Risk Simulator
# ==========================================
with tab1:
    st.subheader("Individual Customer Risk Assessment")
    st.markdown("Select a preset scenario or customize customer attributes to evaluate real-time churn probability.")

    preset = st.selectbox(
        "Load Customer Profile Preset:",
        ["Custom Input", "Loyal / High-Usage Customer", "At-Risk / Inactive Customer", "Severe Friction / Churn Imminent"]
    )

    # Preset values
    defaults = {
        "Customer_Age": 45,
        "Gender": "F",
        "Dependent_count": 2,
        "Education_Level": "Graduate",
        "Marital_Status": "Married",
        "Income_Category": "$60K - $80K",
        "Card_Category": "Blue",
        "Months_on_book": 36,
        "Total_Relationship_Count": 4,
        "Months_Inactive_12_mon": 2,
        "Contacts_Count_12_mon": 2,
        "Credit_Limit": 8500.0,
        "Total_Revolving_Bal": 1400.0,
        "Avg_Open_To_Buy": 7100.0,
        "Total_Amt_Chng_Q4_Q1": 0.78,
        "Total_Trans_Amt": 4500.0,
        "Total_Trans_Ct": 65,
        "Total_Ct_Chng_Q4_Q1": 0.72,
        "Avg_Utilization_Ratio": 0.165
    }

    if preset == "Loyal / High-Usage Customer":
        defaults.update({
            "Months_Inactive_12_mon": 1,
            "Contacts_Count_12_mon": 1,
            "Total_Revolving_Bal": 2200.0,
            "Total_Trans_Amt": 7800.0,
            "Total_Trans_Ct": 88,
            "Total_Relationship_Count": 5,
            "Total_Ct_Chng_Q4_Q1": 0.95,
            "Avg_Utilization_Ratio": 0.35
        })
    elif preset == "At-Risk / Inactive Customer":
        defaults.update({
            "Months_Inactive_12_mon": 3,
            "Contacts_Count_12_mon": 3,
            "Total_Revolving_Bal": 450.0,
            "Total_Trans_Amt": 1800.0,
            "Total_Trans_Ct": 38,
            "Total_Relationship_Count": 2,
            "Total_Ct_Chng_Q4_Q1": 0.45,
            "Total_Amt_Chng_Q4_Q1": 0.52,
            "Avg_Utilization_Ratio": 0.05
        })
    elif preset == "Severe Friction / Churn Imminent":
        defaults.update({
            "Months_Inactive_12_mon": 4,
            "Contacts_Count_12_mon": 5,
            "Total_Revolving_Bal": 0.0,
            "Total_Trans_Amt": 980.0,
            "Total_Trans_Ct": 20,
            "Total_Relationship_Count": 1,
            "Total_Ct_Chng_Q4_Q1": 0.30,
            "Total_Amt_Chng_Q4_Q1": 0.35,
            "Avg_Utilization_Ratio": 0.00
        })

    with st.form("customer_risk_form"):
        col1, col2, col3 = st.columns(3)

        with col1:
            st.markdown("##### 👤 Demographics & Profile")
            age = st.number_input("Customer Age", min_value=18, max_value=100, value=int(defaults["Customer_Age"]))
            gender = st.selectbox("Gender", ["F", "M"], index=0 if defaults["Gender"] == "F" else 1)
            dependents = st.number_input("Dependent Count", min_value=0, max_value=10, value=int(defaults["Dependent_count"]))
            education = st.selectbox("Education Level", ORDINAL_CATEGORIES["Education_Level"], index=ORDINAL_CATEGORIES["Education_Level"].index(defaults["Education_Level"]))
            marital = st.selectbox("Marital Status", ["Married", "Single", "Divorced", "Unknown"], index=0 if defaults["Marital_Status"] == "Married" else 1)
            income = st.selectbox("Income Category", ORDINAL_CATEGORIES["Income_Category"], index=ORDINAL_CATEGORIES["Income_Category"].index(defaults["Income_Category"]))
            card = st.selectbox("Card Category", ORDINAL_CATEGORIES["Card_Category"], index=ORDINAL_CATEGORIES["Card_Category"].index(defaults["Card_Category"]))

        with col2:
            st.markdown("##### 🏦 Account Relationship")
            months_on_book = st.number_input("Months on Book (Tenure)", min_value=1, max_value=120, value=int(defaults["Months_on_book"]))
            total_relationships = st.number_input("Total Products Held", min_value=1, max_value=10, value=int(defaults["Total_Relationship_Count"]))
            months_inactive = st.number_input("Months Inactive (Last 12m)", min_value=0, max_value=12, value=int(defaults["Months_Inactive_12_mon"]))
            contacts_count = st.number_input("Support Contacts (Last 12m)", min_value=0, max_value=12, value=int(defaults["Contacts_Count_12_mon"]))
            credit_limit = st.number_input("Credit Limit ($)", min_value=500.0, max_value=50000.0, value=float(defaults["Credit_Limit"]), step=500.0)
            revolving_bal = st.number_input("Total Revolving Balance ($)", min_value=0.0, max_value=50000.0, value=float(defaults["Total_Revolving_Bal"]), step=100.0)

        with col3:
            st.markdown("##### 💳 Spending & Activity")
            avg_open_to_buy = credit_limit - revolving_bal
            st.metric("Avg Open To Buy ($)", f"${avg_open_to_buy:,.2f}")
            trans_amt = st.number_input("Total Transaction Amount ($)", min_value=100.0, max_value=50000.0, value=float(defaults["Total_Trans_Amt"]), step=200.0)
            trans_ct = st.number_input("Total Transaction Count", min_value=1, max_value=300, value=int(defaults["Total_Trans_Ct"]))
            amt_chng_q4_q1 = st.slider("Amt Change Q4 vs Q1", min_value=0.0, max_value=3.5, value=float(defaults["Total_Amt_Chng_Q4_Q1"]), step=0.05)
            ct_chng_q4_q1 = st.slider("Count Change Q4 vs Q1", min_value=0.0, max_value=3.5, value=float(defaults["Total_Ct_Chng_Q4_Q1"]), step=0.05)
            utilization = revolving_bal / max(credit_limit, 1.0)
            st.metric("Utilization Ratio", f"{utilization*100:.1f}%")

        submit_btn = st.form_submit_button("⚡ Run Churn Intelligence Prediction", use_container_width=True)

    if submit_btn and model_loaded:
        cust_payload = {
            "Customer_Age": age,
            "Gender": gender,
            "Dependent_count": dependents,
            "Education_Level": education,
            "Marital_Status": marital,
            "Income_Category": income,
            "Card_Category": card,
            "Months_on_book": months_on_book,
            "Total_Relationship_Count": total_relationships,
            "Months_Inactive_12_mon": months_inactive,
            "Contacts_Count_12_mon": contacts_count,
            "Credit_Limit": credit_limit,
            "Total_Revolving_Bal": revolving_bal,
            "Avg_Open_To_Buy": avg_open_to_buy,
            "Total_Amt_Chng_Q4_Q1": amt_chng_q4_q1,
            "Total_Trans_Amt": trans_amt,
            "Total_Trans_Ct": trans_ct,
            "Total_Ct_Chng_Q4_Q1": ct_chng_q4_q1,
            "Avg_Utilization_Ratio": utilization
        }

        res = predictor.predict_single(cust_payload)

        st.markdown("---")
        st.subheader("🔍 Prediction Results & Retention Diagnosis")

        res_col1, res_col2, res_col3 = st.columns([1.2, 1, 1])

        with res_col1:
            prob = res["churn_probability"]
            # Gauge chart
            fig_gauge = go.Figure(go.Indicator(
                mode="gauge+number",
                value=prob * 100,
                number={'suffix': "%", 'font': {'size': 32}},
                title={'text': "Churn Probability", 'font': {'size': 18}},
                gauge={
                    'axis': {'range': [0, 100]},
                    'bar': {'color': "#EF4444" if prob >= 0.50 else "#F59E0B" if prob >= 0.25 else "#10B981"},
                    'steps': [
                        {'range': [0, 25], 'color': "#DCFCE7"},
                        {'range': [25, 50], 'color': "#FEF3C7"},
                        {'range': [50, 75], 'color': "#FEE2E2"},
                        {'range': [75, 100], 'color': "#FECACA"},
                    ],
                    'threshold': {
                        'line': {'color': "black", 'width': 3},
                        'thickness': 0.8,
                        'value': res["decision_threshold"] * 100
                    }
                }
            ))
            fig_gauge.update_layout(height=260, margin=dict(l=20, r=20, t=30, b=10))
            st.plotly_chart(fig_gauge, use_container_width=True)

        with res_col2:
            st.markdown("##### Status & Tier")
            tier_info = res["risk_tier"]
            tier_badge = "🔴" if tier_info["tier_code"] in ["HIGH", "CRITICAL"] else "🟡" if tier_info["tier_code"] == "MODERATE" else "🟢"
            st.markdown(f"### {tier_badge} {tier_info['label']}")
            st.info(f"**Predicted Action:** {'⚠️ Flag for Retention Intervention' if res['is_churn'] else '✅ Retained Account'}")
            st.caption(f"Decision Threshold: `{res['decision_threshold']:.2f}`")

        with res_col3:
            st.markdown("##### 💵 Financial Impact")
            expected_loss = prob * CUSTOMER_LIFETIME_VALUE
            st.metric("Expected Lifetime Value at Risk", f"${expected_loss:,.2f}")
            st.metric("Recommended Campaign Cost", f"${RETENTION_CAMPAIGN_COST:,.2f}")
            net_roi = ((CUSTOMER_LIFETIME_VALUE * RETENTION_SUCCESS_RATE) - RETENTION_CAMPAIGN_COST) if res["is_churn"] else 0.0
            st.metric("Expected Net Recovery Value", f"${net_roi:,.2f}")

        # Diagnosis Breakdown
        col_risk, col_rec = st.columns(2)
        with col_risk:
            st.markdown("##### 🚨 Key Churn Risk Drivers")
            if res["key_risk_drivers"]:
                for rf in res["key_risk_drivers"]:
                    st.error(f"• {rf}")
            else:
                st.success("• No acute risk flags detected in usage profile.")

            if res["positive_signals"]:
                st.markdown("##### ✨ Positive Customer Signals")
                for ps in res["positive_signals"]:
                    st.success(f"• {ps}")

        with col_rec:
            st.markdown("##### 🎯 Prescribed Retention Interventions")
            st.warning(f"**Protocol:** {res['urgency_level']}")
            for action in res["prescribed_retention_actions"]:
                st.markdown(f"- 📌 {action}")

# ==========================================
# TAB 2: Portfolio Batch Analytics
# ==========================================
with tab2:
    st.subheader("Customer Portfolio Batch Scoring")
    st.markdown("Score multiple accounts simultaneously to generate executive retention worklists.")

    batch_source = st.radio("Choose Data Source:", ["Use Benchmark Dataset Sample (500 Customers)", "Upload Custom CSV File"], horizontal=True)

    df_batch = None
    if batch_source == "Upload Custom CSV File":
        uploaded_file = st.file_uploader("Upload customer CSV", type=["csv"])
        if uploaded_file is not None:
            df_batch = pd.read_csv(uploaded_file)
    else:
        raw_df = load_dataset()
        df_batch = raw_df.sample(n=min(500, len(raw_df)), random_state=42)

    if df_batch is not None and model_loaded:
        st.write(f"Loaded **{len(df_batch)}** customer records.")
        if st.button("🚀 Score Portfolio Batch", use_container_width=True):
            with st.spinner("Scoring customer portfolio with ML pipeline..."):
                scored_df = predictor.predict_batch(df_batch)

            # Summary Metrics
            total_customers = len(scored_df)
            total_churn_pred = (scored_df["Predicted_Attrition"] == "Attrited Customer").sum()
            churn_pct = (total_churn_pred / total_customers) * 100
            total_at_risk_clv = scored_df["Churn_Probability"].sum() * CUSTOMER_LIFETIME_VALUE

            mcol1, mcol2, mcol3, mcol4 = st.columns(4)
            mcol1.metric("Total Customers", f"{total_customers:,}")
            mcol2.metric("Predicted Churners", f"{total_churn_pred:,}")
            mcol3.metric("Projected Churn Rate", f"{churn_pct:.1f}%")
            mcol4.metric("Total Revenue at Risk", f"${total_at_risk_clv:,.2f}")

            # Risk Distribution Chart
            tier_counts = scored_df["Risk_Tier"].value_counts().reset_index()
            tier_counts.columns = ["Risk_Tier", "Count"]

            fig_bar = px.bar(
                tier_counts,
                x="Risk_Tier",
                y="Count",
                color="Risk_Tier",
                color_discrete_map={
                    "Low Risk": "#10B981",
                    "Moderate Risk": "#F59E0B",
                    "High Risk": "#F97316",
                    "Critical Risk": "#EF4444"
                },
                title="Customer Distribution by Churn Risk Tier"
            )
            st.plotly_chart(fig_bar, use_container_width=True)

            # Interactive Filterable Table
            st.markdown("##### 📋 Actionable Customer Worklist")
            risk_filter = st.multiselect("Filter by Risk Tier:", ["Critical Risk", "High Risk", "Moderate Risk", "Low Risk"], default=["Critical Risk", "High Risk"])
            filtered_df = scored_df[scored_df["Risk_Tier"].isin(risk_filter)]

            st.dataframe(
                filtered_df[[
                    col for col in ["CLIENTNUM", "Churn_Probability", "Risk_Tier", "Customer_Age", "Gender",
                                   "Credit_Limit", "Total_Trans_Ct", "Total_Trans_Amt", "Months_Inactive_12_mon", "Contacts_Count_12_mon"]
                    if col in filtered_df.columns
                ]],
                use_container_width=True
            )

            # Download CSV
            csv_data = scored_df.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="📥 Download Scored Customer Retention Worklist (CSV)",
                data=csv_data,
                file_name="scored_churn_portfolio.csv",
                mime="text/csv",
                use_container_width=True
            )

# ==========================================
# TAB 3: Exploratory Insights (EDA)
# ==========================================
with tab3:
    st.subheader("Data Intelligence & Churn Drivers EDA")
    raw_df = load_dataset()

    eda_col1, eda_col2 = st.columns(2)

    with eda_col1:
        # Churn Rate Breakdown
        target_counts = raw_df["Attrition_Flag"].value_counts().reset_index()
        target_counts.columns = ["Status", "Count"]
        fig_pie = px.pie(
            target_counts,
            values="Count",
            names="Status",
            title="Overall Customer Attrition Distribution",
            color="Status",
            color_discrete_map={"Existing Customer": "#3B82F6", "Attrited Customer": "#EF4444"},
            hole=0.45
        )
        st.plotly_chart(fig_pie, use_container_width=True)

        # Total Transactions vs Churn
        fig_trans = px.histogram(
            raw_df,
            x="Total_Trans_Ct",
            color="Attrition_Flag",
            nbins=40,
            barmode="overlay",
            title="Transaction Count Distribution (Churners cluster < 50 transactions)",
            color_discrete_map={"Existing Customer": "#3B82F6", "Attrited Customer": "#EF4444"}
        )
        st.plotly_chart(fig_trans, use_container_width=True)

    with eda_col2:
        # Inactivity vs Churn
        fig_inactive = px.histogram(
            raw_df,
            x="Months_Inactive_12_mon",
            color="Attrition_Flag",
            title="Months Inactive in Last 12 Months",
            color_discrete_map={"Existing Customer": "#3B82F6", "Attrited Customer": "#EF4444"}
        )
        st.plotly_chart(fig_inactive, use_container_width=True)

        # Revolving balance vs Utilization
        fig_revolving = px.box(
            raw_df,
            x="Attrition_Flag",
            y="Total_Revolving_Bal",
            color="Attrition_Flag",
            title="Total Revolving Balance (Attrited customers exhibit low/zero revolving balance)",
            color_discrete_map={"Existing Customer": "#3B82F6", "Attrited Customer": "#EF4444"}
        )
        st.plotly_chart(fig_revolving, use_container_width=True)

# ==========================================
# TAB 4: Model Performance & Business ROI
# ==========================================
with tab4:
    st.subheader("Model Performance & Financial ROI Calibration")

    # Load metadata if exists
    meta = {}
    if METADATA_PATH.exists():
        with open(METADATA_PATH, "r") as f:
            meta = json.load(f)

    if meta:
        metrics = meta.get("metrics_at_optimal_threshold", {})
        cm = metrics.get("confusion_matrix", {})
        bi = metrics.get("business_impact", {})

        pcol1, pcol2, pcol3, pcol4, pcol5 = st.columns(5)
        pcol1.metric("Accuracy", f"{metrics.get('accuracy', 0)*100:.2f}%")
        pcol2.metric("Precision", f"{metrics.get('precision', 0)*100:.2f}%")
        pcol3.metric("Recall", f"{metrics.get('recall', 0)*100:.2f}%")
        pcol4.metric("F1 Score", f"{metrics.get('f1', 0)*100:.2f}%")
        pcol5.metric("ROC-AUC", f"{metrics.get('roc_auc', 0):.4f}")

        st.markdown("---")
        c1, c2 = st.columns([1.2, 1])

        with c1:
            st.markdown("##### 🌲 Top Predictive Features")
            top_feats = meta.get("top_features", [])
            if top_feats:
                feat_df = pd.DataFrame(top_feats).head(10)
                fig_feat = px.bar(
                    feat_df,
                    x="importance",
                    y="feature",
                    orientation="h",
                    title="Top 10 Feature Importances (%)",
                    labels={"importance": "Importance (%)", "feature": "Feature"},
                    color="importance",
                    color_continuous_scale="Blues"
                )
                fig_feat.update_layout(yaxis={'categoryorder': 'total ascending'})
                st.plotly_chart(fig_feat, use_container_width=True)

        with c2:
            st.markdown("##### 💼 Financial ROI Analysis")
            st.write(f"• **Net Saved Revenue:** `${bi.get('net_benefit_usd', 0):,.2f}`")
            st.write(f"• **Gross Saved Churn:** `${bi.get('saved_revenue_usd', 0):,.2f}`")
            st.write(f"• **Total Retention Costs:** `${bi.get('retention_costs_usd', 0):,.2f}`")
            st.write(f"• **Campaign ROI Ratio:** `{bi.get('roi_ratio', 0):.2f}x`")
            st.write(f"• **Optimal Threshold:** `{meta.get('optimal_threshold', 0.35):.2f}`")

            # Confusion Matrix Visualization
            if cm:
                cm_data = [[cm['true_negatives'], cm['false_positives']],
                           [cm['false_negatives'], cm['true_positives']]]
                fig_cm = px.imshow(
                    cm_data,
                    text_auto=True,
                    labels=dict(x="Predicted Label", y="True Label", color="Count"),
                    x=["Retained (0)", "Churned (1)"],
                    y=["Retained (0)", "Churned (1)"],
                    color_continuous_scale="Blues",
                    title="Confusion Matrix"
                )
                st.plotly_chart(fig_cm, use_container_width=True)

    # Candidate Models Benchmark Table
    if METRICS_SUMMARY_PATH.exists():
        st.markdown("##### 📊 Benchmark Model Comparison")
        with open(METRICS_SUMMARY_PATH, "r") as f:
            bench_data = json.load(f)
        bench_df = pd.DataFrame(bench_data)
        st.dataframe(bench_df, use_container_width=True)

st.markdown("---")
st.caption("Credit Card Churn Intelligence Platform | Production AI System | Powered by Antigravity")
