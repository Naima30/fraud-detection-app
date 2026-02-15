import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    precision_recall_curve,
    confusion_matrix,
    roc_curve,
    auc
)

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE


# ---------------------------------------------------
# PAGE CONFIG
# ---------------------------------------------------
st.set_page_config(page_title="Fraud Risk Intelligence System", layout="wide")

st.title("💳 Fraud Risk Intelligence Dashboard")


# ---------------------------------------------------
# SESSION STATE INIT
# ---------------------------------------------------
if "df" not in st.session_state:
    st.session_state.df = None

if "model" not in st.session_state:
    st.session_state.model = None

if "X_columns" not in st.session_state:
    st.session_state.X_columns = None


# ---------------------------------------------------
# SIDEBAR NAVIGATION
# ---------------------------------------------------
menu = [
    "1. Load Dataset",
    "2. Data Exploration",
    "3. Data Cleaning",
    "4. Feature Engineering",
    "5. Model Training",
    "6. Model Evaluation",
    "7. Risk Prediction",
    "8. Business Dashboard"
]

choice = st.sidebar.radio("Navigation", menu)


# ---------------------------------------------------
# MODULE 1: LOAD DATA
# ---------------------------------------------------
if choice == "1. Load Dataset":

    st.header("Dataset Loader")

    uploaded_file = st.file_uploader("Upload CSV", type=["csv"])

    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        st.session_state.df = df
        st.success("Dataset Loaded Successfully")

        st.subheader("Preview")
        st.dataframe(df.head())

        st.subheader("Shape")
        st.write(df.shape)

        st.subheader("Missing Values")
        st.write(df.isnull().sum())


# ---------------------------------------------------
# MODULE 2: DATA EXPLORATION
# ---------------------------------------------------
elif choice == "2. Data Exploration":

    if st.session_state.df is None:
        st.warning("Upload dataset first")
    else:
        df = st.session_state.df

        st.header("Exploratory Data Analysis")

        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Fraud Distribution")
            st.bar_chart(df["Is_Fraud"].value_counts())

        with col2:
            st.subheader("Transaction Amount Distribution")
            fig = px.histogram(df, x="Transaction_Amount", nbins=50)
            st.plotly_chart(fig)

        st.subheader("Correlation Heatmap")

        numeric_df = df.select_dtypes(include=np.number)
        corr = numeric_df.corr()

        fig, ax = plt.subplots(figsize=(10, 6))
        sns.heatmap(corr, annot=False, cmap="coolwarm", ax=ax)
        st.pyplot(fig)

        st.subheader("Fraud vs Transaction Amount")
        fig2 = px.box(df, x="Is_Fraud", y="Transaction_Amount")
        st.plotly_chart(fig2)


# ---------------------------------------------------
# MODULE 3: DATA CLEANING
# ---------------------------------------------------
elif choice == "3. Data Cleaning":

    if st.session_state.df is None:
        st.warning("Upload dataset first")
    else:
        df = st.session_state.df.copy()
        st.header("Data Cleaning")

        st.subheader("Missing Values")
        st.write(df.isnull().sum())

        method = st.selectbox("Imputation Method", ["None", "Fill Median"])

        if method == "Fill Median":
            for col in df.select_dtypes(include=np.number).columns:
                df[col].fillna(df[col].median(), inplace=True)
            st.success("Missing values filled")

        st.subheader("Outlier Handling")

        col_select = st.selectbox("Select Column", df.select_dtypes(include=np.number).columns)

        Q1 = df[col_select].quantile(0.25)
        Q3 = df[col_select].quantile(0.75)
        IQR = Q3 - Q1

        lower = Q1 - 1.5 * IQR
        upper = Q3 + 1.5 * IQR

        st.write("Outliers:", df[(df[col_select] < lower) | (df[col_select] > upper)].shape[0])

        if st.button("Cap Outliers"):
            df[col_select] = np.clip(df[col_select], lower, upper)
            st.success("Outliers capped")

        if st.button("Save Cleaned Data"):
            st.session_state.df = df
            st.success("Saved")


# ---------------------------------------------------
# MODULE 4: FEATURE ENGINEERING
# ---------------------------------------------------
elif choice == "4. Feature Engineering":

    if st.session_state.df is None:
        st.warning("Upload dataset first")
    else:
        df = st.session_state.df.copy()
        st.header("Feature Engineering")

        if st.checkbox("Create Balance Delta"):
            if "Account_Balance_Pre" in df.columns and "Account_Balance_Post" in df.columns:
                df["Balance_Delta"] = df["Account_Balance_Pre"] - df["Account_Balance_Post"]
                st.success("Feature Created")

        if st.checkbox("Create Spend Ratio"):
            if "Average_Monthly_Spend" in df.columns:
                df["Spend_Ratio"] = df["Transaction_Amount"] / (df["Average_Monthly_Spend"] + 1)

        st.subheader("Encoding Categorical Features")

        if st.button("Encode Categories"):
            for col in df.select_dtypes(include="object").columns:
                df[col] = LabelEncoder().fit_transform(df[col])
            st.session_state.df = df
            st.success("Encoding Done")

        st.dataframe(df.head())


# ---------------------------------------------------
# MODULE 5: MODEL TRAINING
# ---------------------------------------------------
elif choice == "5. Model Training":

    if st.session_state.df is None:
        st.warning("Upload dataset first")
    else:
        df = st.session_state.df.copy()

        st.header("Model Training")

        target = "Is_Fraud"

        X = df.drop(columns=[target])
        y = df[target]

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        smote = SMOTE()
        X_res, y_res = smote.fit_resample(X_scaled, y)

        X_train, X_test, y_train, y_test = train_test_split(
            X_res, y_res, test_size=0.2, random_state=42
        )

        algo = st.selectbox(
            "Choose Algorithm",
            ["Random Forest", "Logistic Regression", "XGBoost"]
        )

        if st.button("Train Model"):

            if algo == "Random Forest":
                model = RandomForestClassifier(n_estimators=200)

            elif algo == "Logistic Regression":
                model = LogisticRegression(max_iter=500)

            else:
                model = XGBClassifier(eval_metric="logloss")

            model.fit(X_train, y_train)

            st.session_state.model = model
            st.session_state.X_columns = X.columns
            st.session_state.X_test = X_test
            st.session_state.y_test = y_test

            st.success("Model Trained Successfully")


# ---------------------------------------------------
# MODULE 6: MODEL EVALUATION
# ---------------------------------------------------
elif choice == "6. Model Evaluation":

    if st.session_state.model is None:
        st.warning("Train model first")
    else:
        model = st.session_state.model
        X_test = st.session_state.X_test
        y_test = st.session_state.y_test

        st.header("Model Evaluation")

        y_probs = model.predict_proba(X_test)[:, 1]
        y_pred = (y_probs > 0.5).astype(int)

        acc = accuracy_score(y_test, y_pred)
        roc_auc = roc_auc_score(y_test, y_probs)

        st.metric("Accuracy", f"{acc:.3f}")
        st.metric("ROC-AUC", f"{roc_auc:.3f}")

        st.subheader("Confusion Matrix")

        cm = confusion_matrix(y_test, y_pred)
        fig, ax = plt.subplots()
        sns.heatmap(cm, annot=True, fmt="d", ax=ax)
        st.pyplot(fig)

        st.subheader("ROC Curve")

        fpr, tpr, _ = roc_curve(y_test, y_probs)
        fig2 = px.line(x=fpr, y=tpr)
        st.plotly_chart(fig2)


# ---------------------------------------------------
# MODULE 7: RISK PREDICTION (ADVANCED UI)
# ---------------------------------------------------
elif choice == "7. Risk Prediction":

    if st.session_state.model is None:
        st.warning("Train model first")
    else:
        st.header("🔎 Live Fraud Risk Analyzer")

        model = st.session_state.model
        columns = st.session_state.X_columns

        st.subheader("Enter Transaction Details")

        col1, col2 = st.columns(2)

        user_input = {}

        # Numeric Inputs
        numeric_fields = [
            "Transaction_Amount",
            "Account_Balance_Pre",
            "Account_Balance_Post",
            "Login_Attempts",
            "Customer_Age",
            "Average_Monthly_Spend"
        ]

        # Categorical Inputs
        categorical_fields = {
            "Is_International": [0, 1],
            "Transaction_Type": ["UPI", "Debit Card", "Credit Card", "Net Banking"],
            "Device_Type": ["Mobile-Android", "Mobile-iOS", "Desktop", "Tablet"],
            "Merchant_Category": ["Travel", "Dining", "Electronics", "Gaming"],
        }

        # Numeric UI
        with col1:
            for field in numeric_fields:
                if field in columns:
                    user_input[field] = st.number_input(field, value=0.0)

        # Categorical UI
        with col2:
            for field, options in categorical_fields.items():
                if field in columns:
                    user_input[field] = st.selectbox(field, options)

        # Prediction
        if st.button("Predict Risk Score"):

            input_df = pd.DataFrame([user_input])

            # Handle encoding like training stage
            input_df = pd.get_dummies(input_df)

            # Align columns
            input_df = input_df.reindex(columns=columns, fill_value=0)

            prob = model.predict_proba(input_df)[0][1]

            st.subheader("Risk Assessment")

            colA, colB = st.columns(2)

            with colA:
                st.metric("Fraud Probability", f"{prob:.2f}")

            with colB:
                if prob > 0.7:
                    st.error("🔴 High Risk Transaction")
                elif prob > 0.4:
                    st.warning("🟡 Medium Risk Transaction")
                else:
                    st.success("🟢 Low Risk Transaction")

            # Interpretation
            st.info(f"""
            Interpretation:
            - Model estimates a **{prob*100:.1f}% probability** that this transaction is fraudulent.
            - High values of transaction amount, login attempts, or international transactions typically increase risk.
            """)


# ---------------------------------------------------
# MODULE 8: BUSINESS DASHBOARD
# ---------------------------------------------------
elif choice == "8. Business Dashboard":

    if st.session_state.model is None:
        st.warning("Train model first")
    else:
        st.header("Executive Dashboard")

        y_test = st.session_state.y_test
        X_test = st.session_state.X_test
        model = st.session_state.model

        probs = model.predict_proba(X_test)[:, 1]
        preds = (probs > 0.5).astype(int)

        cm = confusion_matrix(y_test, preds)

        tp = cm[1,1]
        fn = cm[1,0]
        fp = cm[0,1]

        avg_loss = 1200
        review_cost = 50

        savings = tp * avg_loss
        risk = fn * avg_loss
        ops = fp * review_cost

        col1, col2, col3 = st.columns(3)
        col1.metric("Fraud Caught", tp)
        col2.metric("Loss Prevented", f"${savings}")
        col3.metric("Operational Cost", f"${ops}")

        df_plot = pd.DataFrame({
            "Metric": ["Saved", "Risk", "Ops Cost"],
            "Value": [savings, risk, ops]
        })

        fig = px.bar(df_plot, x="Metric", y="Value")
        st.plotly_chart(fig)