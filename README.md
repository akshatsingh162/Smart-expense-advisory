# 🧠 Smart Expense Advisory

A machine-learning powered Streamlit app that analyzes your expense descriptions, classifies them into categories, and provides personalized spending advice.

## 📌 Overview

Smart Expense Advisory is designed to help users gain insights into their spending patterns.
By entering a simple text description of an expense (e.g., "Dinner at KFC", "Uber ride to airport"), the app:

1. Classifies the expense into a category using a trained ML model

2. Analyzes the pattern of expenses

3. Provides meaningful advice to help users budget better

The project uses TF-IDF vectorization for text processing and a machine-learning classification model trained on labeled expense data.

## 🚀 Features

✔️ Classifies textual expenses into categories (Food, Travel, Shopping, Utilities, etc.)

✔️ ML-based prediction using saved .pkl model files

✔️ Real-time advisory suggestions

✔️ Simple and interactive Streamlit UI

✔️ Can be deployed easily on Streamlit Cloud

## 🏗️ Project Structure

```graphnql
Smart-expense-advisory/
│
├── expense_model_v2.pkl          # Trained ML classification model
├── tfidf_vectorizer_v2.pkl       # TF-IDF vectorizer used for text preprocessing
├── streamlit_app.py              # Main web application script
└── requirements.txt              # Required dependencies
```

Pull requests are welcome. For major changes, please open an issue first
to discuss what you would like to change.

Please make sure to update tests as appropriate.

## 🧰 Tech Stack

1. Python
2. Streamlit
3. Scikit-learn
4. Pandas / NumPy
5. TF-IDF Vectorization

## 🧠 How It Works
* User enters an expense description
* App loads the TF-IDF vectorizer
* The text is converted into feature vectors

* The ML model predicts the expense category

* App displays:

  1. Predicted category

  2. Insights

  3. Advisory messages
