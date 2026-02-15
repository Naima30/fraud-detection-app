# ======================================================
# Fraud Detection Dashboard – Final Version
# ======================================================

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, confusion_matrix, roc_auc_score,
    precision_recall_curve, roc_curve, auc
)

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import SMOTE

import lime
import lime.lime_tabular

sns.set_style("whitegrid")

# ======================================================
# PAGE CONFIG
# ======================================================
st.set_page_config(page_title="Fraud Detection Dashboard", layout="wide")

st.title("💳 Fraud Detection Dashboard")

# ======================================================
# LOAD DATA
# ======================================================
uploaded_file = st.sidebar.file_uploader("Upload dataset", type=["csv"])

if uploaded_file is None:
    st.info("Upload dataset to continue.")
    st.stop()

df = pd.read_csv(uploaded_file)
df.columns = df.columns.str.strip()

# ======================================================
# SIDEBAR NAVIGATION
# ======================================================
page = st.sidebar.radio(
    "Navigation",
    ["Overview", "EDA", "Model Training", "Risk Predictor"]
)

# ======================================================
# OVERVIEW PAGE
# ======================================================
if page == "Overview":

    st.subheader("Dataset Overview")
    st.dataframe(df.head(), use_container_width=True)

    col1, col2, col3 = st.columns(3)
    col1.metric("Rows", df.shape[0])
    col2.metric("Columns", df.shape[1])

    if "Is_Fraud" in df.columns:
        col3.metric("Fraud Cases", int(df["Is_Fraud"].sum()))

# ======================================================
# EDA PAGE
# ======================================================
elif page == "EDA":

    st.subheader("Fraud Distribution")

    fig, ax = plt.subplots()
    sns.countplot(x=df["Is_Fraud"], ax=ax)
    st.pyplot(fig)

    st.subheader("Correlation Heatmap")
    numeric_cols = df.select_dtypes(include=np.number)

    fig2, ax2 = plt.subplots(figsize=(8,5))
    sns.heatmap(numeric_cols.corr(), cmap="coolwarm", ax=ax2)
    st.pyplot(fig2)

# ======================================================
# MODEL TRAINING PAGE
# ======================================================
elif page == "Model Training":

    if "Is_Fraud" not in df.columns:
        st.error("Dataset must contain Is_Fraud column")
        st.stop()

    st.subheader("Preparing Data")

    # Drop ID columns
    df_model = df.drop(columns=["Transaction_ID", "Customer_ID"], errors="ignore")

    X = df_model.drop("Is_Fraud", axis=1)
    y = df_model["Is_Fraud"]

    # Encoding categorical
    X = pd.get_dummies(X, drop_first=True)

    # Train Test Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # ========================
    # SMOTE Handling Imbalance
    # ========================
    st.subheader("Handling Imbalance using SMOTE")

    sm = SMOTE(random_state=42)
    X_train_sm, y_train_sm = sm.fit_resample(X_train, y_train)

    fig3, ax3 = plt.subplots()
    sns.countplot(x=y_train_sm, ax=ax3)
    ax3.set_title("After SMOTE")
    st.pyplot(fig3)

    # ========================
    # Train Models
    # ========================
    st.subheader("Training Models")

    models = {
        "Logistic Regression": LogisticRegression(max_iter=2000),
        "Random Forest": RandomForestClassifier(n_estimators=200)
    }

    results = []

    for name, model in models.items():

        model.fit(X_train_sm, y_train_sm)
        preds = model.predict(X_test)
        probs = model.predict_proba(X_test)[:,1]

        acc = accuracy_score(y_test, preds)
        prec = precision_score(y_test, preds)
        rec = recall_score(y_test, preds)
        f1 = f1_score(y_test, preds)
        roc_auc = roc_auc_score(y_test, probs)

        results.append([name, acc, prec, rec, f1, roc_auc])

        st.subheader(name)

        # Confusion Matrix
        fig4, ax4 = plt.subplots()
        sns.heatmap(confusion_matrix(y_test, preds), annot=True, fmt="d", ax=ax4)
        st.pyplot(fig4)

        # ROC Curve
        fpr, tpr, _ = roc_curve(y_test, probs)
        fig5, ax5 = plt.subplots()
        ax5.plot(fpr, tpr, label=f"AUC={roc_auc:.2f}")
        ax5.plot([0,1],[0,1],'--')
        ax5.legend()
        st.pyplot(fig5)

        # Precision Recall Curve
        precision, recall, _ = precision_recall_curve(y_test, probs)
        fig6, ax6 = plt.subplots()
        ax6.plot(recall, precision)
        ax6.set_title("Precision Recall Curve")
        st.pyplot(fig6)

        # Save best model
        st.session_state[f"model_{name}"] = model
        st.session_state["X_train"] = X_train_sm
        st.session_state["X_test"] = X_test
        st.session_state["y_test"] = y_test
        st.session_state["feature_columns"] = X_train.columns

    st.dataframe(pd.DataFrame(results,
        columns=["Model","Accuracy","Precision","Recall","F1","ROC-AUC"]
    ))

# ======================================================
# RISK PREDICTOR PAGE
# ======================================================
elif page == "Risk Predictor":

    available_models = {
        k.replace("model_",""): st.session_state[k]
        for k in st.session_state if k.startswith("model_")
    }

    if not available_models:
        st.warning("Train a model first.")
        st.stop()

    model_choice = st.selectbox("Select Model", list(available_models.keys()))
    model = available_models[model_choice]

    st.subheader("Enter Transaction Details")

    feature_cols = st.session_state["feature_columns"]
    inputs = {}

    cols = st.columns(2)

    for i, col in enumerate(feature_cols[:10]):  # limit UI size
        inputs[col] = cols[i % 2].number_input(col, 0.0)

    if st.button("Predict Fraud Risk"):

        Xnew = pd.DataFrame([inputs])

        # Align columns
        for col in feature_cols:
            if col not in Xnew.columns:
                Xnew[col] = 0

        Xnew = Xnew[feature_cols]

        prob = model.predict_proba(Xnew)[0][1] * 100

        # Gauge chart
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

        # ==========================
        # LIME EXPLAINABILITY
        # ==========================
        st.subheader("Why was this prediction made? (LIME)")

        explainer = lime.lime_tabular.LimeTabularExplainer(
            training_data=np.array(st.session_state["X_train"]),
            feature_names=feature_cols,
            class_names=["Legit","Fraud"],
            mode="classification"
        )

        exp = explainer.explain_instance(
            Xnew.iloc[0].values,
            model.predict_proba,
            num_features=5
        )

        fig_lime = exp.as_pyplot_figure()
        st.pyplot(fig_lime)