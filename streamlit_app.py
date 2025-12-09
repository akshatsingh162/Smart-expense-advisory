import streamlit as st
import pandas as pd
import joblib

# -----------------------#
# Page Config
# -----------------------#
st.set_page_config(
    page_title="Smart Expense Dashboard",
    page_icon="💸",
    layout="wide"
)

# -----------------------#
# Load Model & Vectorizer
# -----------------------#
@st.cache_resource
def load_components():
    model = joblib.load("expense_model_v2.pkl")
    tfidf = joblib.load("tfidf_vectorizer_v2.pkl")
    return model, tfidf

model, tfidf = load_components()

# -----------------------#
# Sidebar
# -----------------------#
st.sidebar.title("📊 Expense Dashboard")
st.sidebar.markdown("---")

page = st.sidebar.radio(
    "Navigate",
    ["Upload File", "Predictions", "Analytics Dashboard", "About"],
    label_visibility="collapsed"
)

st.sidebar.markdown("---")
st.sidebar.caption("💸 Smart Expense Classifier – Powered by ML")

# -----------------------#
# File Upload (Shared)
# -----------------------#
if "df" not in st.session_state:
    st.session_state.df = None

if page == "Upload File":
    st.title("📤 Upload Your Bank Statement")

    uploaded_file = st.file_uploader("Upload CSV", type=["csv"])

    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        st.session_state.df = df

        st.success("File uploaded successfully!")
        st.write(df)

        if "Description" not in df.columns:
            st.error("CSV must contain a 'Description' column.")

# -----------------------#
# Predictions Page
# -----------------------#
elif page == "Predictions":
    st.title("🔮 Classification Results")

    if st.session_state.df is None:
        st.warning("Please upload a CSV file first.")
        st.stop()

    df = st.session_state.df.copy()

    # TF-IDF transform
    X = tfidf.transform(df["Description"].astype(str))

    # Predict categories
    df["Predicted Category"] = model.predict(X)

    st.session_state.df = df  # save predictions

    st.dataframe(df, use_container_width=True)

    # Download
    csv = df.to_csv(index=False).encode("utf-8")
    st.download_button(
        "⬇️ Download Classified CSV",
        csv,
        "classified_expenses.csv",
        "text/csv",
        use_container_width=True
    )

# -----------------------#
# Analytics Dashboard
# -----------------------#
elif page == "Analytics Dashboard":
    st.title("📊 Analytics Dashboard")

    if st.session_state.df is None or "Predicted Category" not in st.session_state.df:
        st.warning("Please upload a file and generate predictions first.")
        st.stop()

    df = st.session_state.df

    # Category Summary
    st.subheader("📌 Category Distribution")
    category_counts = df["Predicted Category"].value_counts()

    col1, col2 = st.columns(2)

    with col1:
        st.bar_chart(category_counts)

    with col2:
        st.write("### 🥧 Category Pie Chart")
        st.pyplot(category_counts.plot.pie(autopct="%1.1f%%").figure)

    # Filter Section
    st.markdown("---")
    st.subheader("🔍 Explore by Category")

    selected = st.multiselect(
        "Select categories",
        df["Predicted Category"].unique()
    )

    if selected:
        filtered_df = df[df["Predicted Category"].isin(selected)]
        st.dataframe(filtered_df, use_container_width=True)

        st.info(f"Showing {len(filtered_df)} matching transactions")

    # Insights
    st.markdown("---")
    st.subheader("💡 AI Insights")

    most_common = category_counts.idxmax()

    st.success(f"Your most frequent expense category is **{most_common}**.")

    st.write("""
    - You might want to review recurring high-frequency categories  
    - Consider setting expense limits for certain groups  
    - Trends can help identify unnecessary spends  
    """)

# -----------------------#
# About
# -----------------------#
elif page == "About":
    st.title("ℹ️ About This App")

    st.markdown("""
    ### 💸 Smart Expense Advisor  
    A machine-learning powered app that classifies your expenses automatically
    and gives real-time insights.

    **Features:**  
    - ML-based category prediction  
    - Interactive analytics dashboard  
    - Pie charts, bar charts, filters  
    - CSV export  
    - Clean & responsive UI  
    """)

    st.markdown("---")
    st.caption("Built with ❤️ using Streamlit")
