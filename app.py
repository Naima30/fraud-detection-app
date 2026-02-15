# ======================================================
# Fraud Detection Dashboard – Final Presentation Version
# ======================================================

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, confusion_matrix, classification_report
)

sns.set_style("whitegrid")

DEFAULT_DATA_PATH = "finguard_transaction_data_P2.csv"

# ======================================================
# UI Styling
# ======================================================
def apply_css():
    css = """
    <style>

    .stApp {
        background-color: #0e1117;
        color: #ffffff;
    }

    /* Hero Banner */
    .hero {
        background: linear-gradient(135deg, #1c1f26, #243b55);
        padding: 40px;
        border-radius: 12px;
        margin-bottom: 25px;
        text-align: center;
        color: white;
    }

    .hero h1 {
        font-size: 38px;
        font-weight: 700;
        margin-bottom: 10px;
        color: white !important;
    }

    .hero p {
        font-size: 16px;
        color: #cfd8dc !important;
    }

    /* Cards */
    .card {
        background: #1c1f26;
        padding: 18px;
        border-radius: 12px;
        margin-bottom: 18px;
        box-shadow: 0 4px 12px rgba(0,0,0,0.5);
    }

    /* Sidebar */
    section[data-testid="stSidebar"] {
        background-color: #1c1f26;
    }

    /* Tables */
    .stDataFrame {
        background-color: #1c1f26;
        color: white;
    }

    </style>
    """
    st.markdown(css, unsafe_allow_html=True)

# ======================================================
# Load Dataset
# ======================================================
@st.cache_data
def load_dataset(uploaded_file):
    if uploaded_file:
        return pd.read_csv(uploaded_file)

    p = Path(DEFAULT_DATA_PATH)
    if p.exists():
        return pd.read_csv(p)

    return None

def auto_encode_features(X):
    return pd.get_dummies(X, drop_first=True)

# ======================================================
# Overview Page
# ======================================================
def overview_page(df):
    st.markdown('<div class="card">', unsafe_allow_html=True)

    st.subheader("Dataset Overview")
    st.dataframe(df.head(), use_container_width=True)

    c1, c2, c3 = st.columns(3)
    c1.metric("Rows", df.shape[0])
    c2.metric("Columns", df.shape[1])

    if "Is_Fraud" in df.columns:
        c3.metric("Fraud Cases", int(df["Is_Fraud"].sum()))

    st.markdown('</div>', unsafe_allow_html=True)

# ======================================================
# EDA Page
# ======================================================
def eda_page(df):
    st.subheader("Exploratory Data Analysis")

    numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
    cat_cols = df.select_dtypes(include="object").columns.tolist()

    # Distribution + Boxplot
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("Distribution & Outliers")

    if numeric_cols:
        col = st.selectbox("Select Numeric Feature", numeric_cols)
        fig, axes = plt.subplots(1,2, figsize=(12,4))
        sns.histplot(df[col], kde=True, ax=axes[0])
        sns.boxplot(y=df[col], ax=axes[1])
        st.pyplot(fig)

    st.markdown('</div>', unsafe_allow_html=True)

    # Scatter
    if len(numeric_cols) >= 2:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.subheader("Scatter Plot")

        x = st.selectbox("X axis", numeric_cols, key="x")
        y = st.selectbox("Y axis", numeric_cols, key="y")

        fig, ax = plt.subplots()
        sns.scatterplot(x=df[x], y=df[y], ax=ax)
        st.pyplot(fig)

        st.markdown('</div>', unsafe_allow_html=True)

def eda_page(df):

    st.subheader("Exploratory Data Analysis")

    numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
    cat_cols = df.select_dtypes(include=['object','category']).columns.tolist()

    # Bar Chart
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("Bar Chart")

    if len(cat_cols) == 0:
        st.warning("No categorical columns available for bar chart.")
    else:
        cat = st.selectbox("Select Categorical Feature", cat_cols)

        counts = df[cat].value_counts().head(10)

        fig, ax = plt.subplots(figsize=(8,4))
        counts.plot(kind="bar", ax=ax)

        ax.set_title(f"Top categories in {cat}")
        st.pyplot(fig)

    st.markdown('</div>', unsafe_allow_html=True)


    # Pie Chart
    if cat_cols:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.subheader("Pie Chart")

        cat2 = st.selectbox("Pie Feature", cat_cols, key="pie")
        fig, ax = plt.subplots()
        df[cat2].value_counts().head(6).plot.pie(autopct="%1.1f%%", ax=ax)
        ax.set_ylabel("")
        st.pyplot(fig)

        st.markdown('</div>', unsafe_allow_html=True)
        

# ======================================================
# Modeling Page
# ======================================================
def prepare_features(df, target="Is_Fraud"):
    df = df.copy()

    # Drop ID-like columns automatically
    drop_cols = []
    for col in df.columns:
        if df[col].nunique() > 1000:
            drop_cols.append(col)

    df = df.drop(columns=drop_cols)

    # Separate X and y
    X = df.drop(target, axis=1)
    y = df[target]

    # Encode categorical safely
    X = pd.get_dummies(X, drop_first=True)

    # Limit features to avoid memory explosion
    if X.shape[1] > 200:
        X = X.iloc[:, :200]

    return X, y

def modeling_page(df):
    st.subheader("Model Training & Comparison")

    if "Is_Fraud" not in df.columns:
        st.error("Dataset must contain 'Is_Fraud'")
        return

    features = st.multiselect(
        "Select Features",
        [c for c in df.columns if c != "Is_Fraud"],
        default=[c for c in df.columns if c != "Is_Fraud"][:5]
    )

    if not features:
        return

    if st.button("Train Models"):

        X, y = prepare_features(df[features + ["Is_Fraud"]])

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        models = {
            "Logistic Regression": LogisticRegression(max_iter=2000),
            "Random Forest": RandomForestClassifier(n_estimators=200)
        }

        results = []

        for name, model in models.items():
            model.fit(X_train, y_train)
            preds = model.predict(X_test)

            acc = accuracy_score(y_test, preds)
            prec = precision_score(y_test, preds)
            rec = recall_score(y_test, preds)
            f1 = f1_score(y_test, preds)

            results.append([name, acc, prec, rec, f1])

            st.subheader(name)
            fig, ax = plt.subplots()
            sns.heatmap(confusion_matrix(y_test, preds), annot=True, fmt="d", ax=ax)
            st.pyplot(fig)

            st.text(classification_report(y_test, preds))

            st.session_state[f"model_{name}"] = model

        st.dataframe(pd.DataFrame(results,
                     columns=["Model","Accuracy","Precision","Recall","F1"]))

# ======================================================
# Risk Predictor Page
# ======================================================
def risk_predictor_page(df):
    st.subheader("Fraud Risk Predictor")

    available_models = {
        k.replace("model_",""): st.session_state[k]
        for k in st.session_state if k.startswith("model_")
    }

    if not available_models:
        st.warning("Train a model first.")
        return

    model_choice = st.selectbox("Select Model", list(available_models.keys()))
    model = available_models[model_choice]

    numeric_features = [
        c for c in df.columns
        if c != "Is_Fraud" and pd.api.types.is_numeric_dtype(df[c])
    ]

    inputs = {}
    cols = st.columns(2)

    for i, col in enumerate(numeric_features):
        inputs[col] = cols[i % 2].number_input(col, float(df[col].median()))

    if st.button("Analyze Fraud Risk"):
        Xnew = pd.DataFrame([inputs])
        Xnew = auto_encode_features(Xnew)

        for col in model.feature_names_in_:
            if col not in Xnew.columns:
                Xnew[col] = 0
        Xnew = Xnew[model.feature_names_in_]

        prob = model.predict_proba(Xnew)[0][1] * 100

        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=prob,
            gauge={'axis': {'range':[0,100]}}
        ))

        st.plotly_chart(fig, use_container_width=True)

        if prob < 30:
            st.success("Low Risk Transaction")
        elif prob < 60:
            st.warning("Medium Risk Transaction")
        else:
            st.error("High Fraud Risk")

# ======================================================
# About Page
# ======================================================
def about_page():
    st.subheader("About")
    st.write("""
Fraud Detection Dashboard built using:
- Machine Learning
- Streamlit
- Data Visualization
- Risk Scoring
""")

# ======================================================
# Main
# ======================================================
def main():
    apply_css()

    st.markdown("""
    <div class="hero">
        <h1>Fraud Detection Dashboard</h1>
        <p>Machine Learning powered fraud risk analysis</p>
    </div>
    """, unsafe_allow_html=True)

    uploaded_file = st.sidebar.file_uploader("Upload dataset", type=["csv"])
    df = load_dataset(uploaded_file)

    if df is None:
        st.info("Upload dataset to continue.")
        return

    page = st.sidebar.radio(
        "Navigation",
        ["Overview","Full EDA","Model Training","Risk Predictor","About"]
    )

    if page == "Overview":
        overview_page(df)
    elif page == "Full EDA":
        eda_page(df)
    elif page == "Model Training":
        modeling_page(df)
    elif page == "Risk Predictor":
        risk_predictor_page(df)
    else:
        about_page()

if __name__ == "__main__":
    main()