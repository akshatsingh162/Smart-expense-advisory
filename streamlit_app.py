import streamlit as st
import pandas as pd
import joblib

# Load your model & vectorizer
model = joblib.load("expense_model.pkl")
tfidf = joblib.load("tfidf_vectorizer.pkl")

st.title("💸 Expense Classifier")

uploaded_file = st.file_uploader("Upload Bank Statement (CSV)", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)

    st.subheader("📄 Uploaded Data")
    st.write(df)

    if "Description" not in df.columns:
        st.error("CSV must contain a 'Description' column")
    else:
        # Preprocess text using TF-IDF
        X = tfidf.transform(df["Description"].astype(str))
        
        # Predict categories
        df["Predicted Category"] = model.predict(X)

        st.subheader("🔮 Predictions")
        st.write(df)

        # Download button
        csv = df.to_csv(index=False).encode("utf-8")
        st.download_button(
            "Download Results",
            csv,
            "classified_expenses.csv",
            "text/csv"
        )
