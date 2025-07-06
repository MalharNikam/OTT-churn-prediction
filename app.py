import streamlit as st
import pandas as pd
import pickle
import shap
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load model
model = pickle.load(open("churn_model.pkl", "rb"))

# Load dataset
df = pd.read_csv("data/ott_churn_dataset.csv")
df.columns = df.columns.str.strip()

# Ensure consistency in naming for plotting
if 'Devices' in df.columns:
    df = df.rename(columns={'Devices': 'NumOfDevices'})

# SHAP setup
X_sample = df.drop(['CustomerID', 'Churn'], axis=1)
y_sample = df['Churn'].map({'No': 0, 'Yes': 1})

preprocessor = model.named_steps['preprocessor']
cat_features = preprocessor.transformers_[0][2]
num_features = ['Age', 'MonthlyCharges', 'WatchTimePerDay', 'DaysSinceLastLogin', 'NumOfComplaints', 'NumOfDevices']
ohe = preprocessor.named_transformers_['cat']
ohe_features = ohe.get_feature_names_out(cat_features)
processed_feature_names = list(ohe_features) + num_features

# Use raw estimator for SHAP
rf_model = model.named_steps['classifier'].calibrated_classifiers_[0].estimator
explainer = shap.TreeExplainer(rf_model)

feature_labels = {
    'Age': 'Young Age',
    'MonthlyCharges': 'High Charges',
    'WatchTimePerDay': 'Low Watch Time',
    'DaysSinceLastLogin': 'Long Inactivity',
    'NumOfComplaints': 'High Complaints',
    'NumOfDevices': 'Fewer Devices'
}

st.set_page_config(page_title="OTT Churn Dashboard", page_icon="📺", layout="wide")
st.title("📺 OTT Customer Churn Dashboard")

page = st.sidebar.radio("Navigate", ["🏠 Home", "🧮 Predict Churn", "📊 Churn Factors", "🎯 Retention Strategies"])

# --- Home Page ---
if page == "🏠 Home":
    st.header("✨ Welcome to the OTT Churn Prediction App")
    st.write("Predict OTT churn, explain the reasons, and explore retention strategies.")
    st.image("bg.jpeg", use_container_width=True)

# --- Predict Churn Page ---
elif page == "🧮 Predict Churn":
    st.header("🧮 Predict Customer Churn")

    with st.form("prediction_form"):
        customer_id = st.text_input("Customer ID")
        gender = st.selectbox("Gender", df["Gender"].unique())
        age = st.number_input("Age", 10, 100, 30)
        subscription = st.selectbox("Subscription Type", df["SubscriptionType"].unique())
        charges = st.number_input("Monthly Charges", 0.0, 2000.0, 500.0)
        genre = st.selectbox("Preferred Genre", df["PreferredGenre"].unique())
        num_devices = st.number_input("Number of Devices", 1, 10, 2)
        watch_time = st.number_input("Watch Time Per Day", 0.0, 24.0, 2.5)
        last_login = st.number_input("Days Since Last Login", 0, 90, 10)
        complaints = st.number_input("Number of Complaints", 0, 10, 1)
        submit = st.form_submit_button("🚀 Predict")

    if submit:
        input_df = pd.DataFrame([{
            "Gender": gender,
            "Age": age,
            "SubscriptionType": subscription,
            "MonthlyCharges": charges,
            "PreferredGenre": genre,
            "NumOfDevices": num_devices,
            "WatchTimePerDay": watch_time,
            "DaysSinceLastLogin": last_login,
            "NumOfComplaints": complaints
        }])

        prediction = model.predict(input_df)[0]
        prob_array = model.predict_proba(input_df)[0]
        churn_index = list(model.classes_).index(1)
        churn_prob = prob_array[churn_index]

        if customer_id:
            st.info(f"Customer ID: {customer_id}")

        if prediction == 1:
            st.error(f"❌ High chance of churn! (Probability: {churn_prob:.2f})")

            processed_input = preprocessor.transform(input_df)
            shap_values = explainer.shap_values(processed_input)
            contribs = shap_values[churn_index][0] if isinstance(shap_values, list) else shap_values[0]

            top_indices = sorted(
                range(len(contribs)),
                key=lambda i: np.linalg.norm(contribs[i]),
                reverse=True
            )[:3]

            st.markdown("### 🔍 Top Reasons for Churn:")
            for i in top_indices:
                if i < len(processed_feature_names):
                    raw = processed_feature_names[i]
                    label = next((v for k, v in feature_labels.items() if k in raw), raw)
                else:
                    label = f"Unknown Feature #{i}"
                st.markdown(f"- {label}")

            st.markdown("### 💡 Suggested Actions")
            st.markdown("- 🎁 Personalized discount offer")
            st.markdown("- 📺 Recommend content in preferred genre")
            st.markdown("- ☎️ Reach out for feedback")

        else:
            st.success(f"✅ Customer is likely to stay. (Probability: {1 - churn_prob:.2f})")
            st.markdown("### 🌟 Engagement Suggestions")
            st.markdown("- 🏆 Offer loyalty benefits")
            st.markdown("- 🎬 Recommend binge-worthy titles")
            st.markdown("- 📈 Encourage add-on features")

# --- Churn Factors Dashboard ---
elif page == "📊 Churn Factors":
    st.header("📊 Churn Factor Visualizations")

    fig1, ax1 = plt.subplots(figsize=(10, 6))
    sns.countplot(x="SubscriptionType", hue="Churn", data=df, ax=ax1)
    ax1.set_title("Subscription Type vs Churn")
    st.pyplot(fig1)

    fig2, ax2 = plt.subplots(figsize=(10, 6))
    sns.countplot(x="NumOfDevices", hue="Churn", data=df, ax=ax2)
    ax2.set_title("Devices vs Churn")
    st.pyplot(fig2)

    fig3, ax3 = plt.subplots(figsize=(10, 6))
    sns.boxplot(x="Churn", y="WatchTimePerDay", data=df, ax=ax3)
    ax3.set_title("Watch Time vs Churn")
    st.pyplot(fig3)

    fig4, ax4 = plt.subplots(figsize=(10, 6))
    sns.boxplot(x="Churn", y="MonthlyCharges", data=df, ax=ax4)
    ax4.set_title("Monthly Charges vs Churn")
    st.pyplot(fig4)

# --- Retention Strategies ---
elif page == "🎯 Retention Strategies":
    st.header("🎯 Retention Strategies")

    tab1, tab2 = st.tabs(["Proactive", "Reactive"])

    with tab1:
        st.subheader("📈 Proactive Measures")
        st.markdown("- 💡 Reward long watch-time users")
        st.markdown("- 🔮 Recommend genre-matched content")
        st.markdown("- 📨 Send onboarding emails to re-engage")

    with tab2:
        st.subheader("🛠️ Reactive Interventions")
        st.markdown("- 🆓 Provide premium content temporarily")
        st.markdown("- 💳 Offer pause/flex billing options")
        st.markdown("- 📞 Conduct feedback interviews")

    st.markdown("---")
    st.subheader("💡 Bonus Tactics")
    st.markdown("""
    - 🎬 Smart playlists based on habits  
    - 🎉 Limited-time offers for lapsed users  
    - 🔔 Push notifications for trending shows
    """)