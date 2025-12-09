# streamlit_app.py
import streamlit as st
import pandas as pd
import joblib
import plotly.express as px
import io
from datetime import datetime
from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.lib.units import mm
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet

# ----------------------------
# Page config & theme toggle
# ----------------------------
st.set_page_config(page_title="Smart Expense Dashboard", page_icon="💸", layout="wide")

# small CSS for dark/light toggle (Streamlit theme still remains but this helps small tweaks)
DARK_CSS = """
<style>
body { background-color:#0e1117; color: #e6eef8; }
.stButton>button { background-color:#2b3037; color: #fff; }
</style>
"""
LIGHT_CSS = """
<style>
body { background-color: #ffffff; color: #0f1724; }
</style>
"""

# ----------------------------
# Load model + vectorizer
# ----------------------------
@st.cache_resource
def load_models():
    model = joblib.load("expense_model_v2.pkl")
    tfidf = joblib.load("tfidf_vectorizer_v2.pkl")
    return model, tfidf

try:
    model, tfidf = load_models()
except Exception as e:
    st.error("Failed to load model/vectorizer. Make sure expense_model_v2.pkl and tfidf_vectorizer_v2.pkl exist in repo root.")
    st.stop()

# ----------------------------
# Helpers
# ----------------------------
def parse_date_safe(val):
    # tries to parse various date formats and returns datetime.date or None
    try:
        return pd.to_datetime(val, dayfirst=True, errors='coerce').date()
    except:
        return None

def make_pdf_report(df, summary, title="Expense Report"):
    """
    Create a simple PDF in-memory using ReportLab and return bytes.
    Contains title, generated time, summary table and top 20 transactions.
    """
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=A4, rightMargin=12*mm, leftMargin=12*mm, topMargin=12*mm)
    styles = getSampleStyleSheet()
    story = []

    # Title
    story.append(Paragraph(f"<b>{title}</b>", styles["Title"]))
    story.append(Spacer(1, 6))

    # Generated time
    story.append(Paragraph(f"Generated: {datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC')}", styles["Normal"]))
    story.append(Spacer(1, 12))

    # Summary (category counts)
    story.append(Paragraph("<b>Category Summary</b>", styles["Heading3"]))
    data = [["Category", "Count"]]
    for k, v in summary.items():
        data.append([str(k), str(v)])
    t = Table(data, colWidths=[90*mm, 30*mm])
    t.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.HexColor('#dbeafe')),
        ('GRID', (0,0), (-1,-1), 0.25, colors.grey),
        ('ALIGN', (1,1), (-1,-1), 'CENTER'),
    ]))
    story.append(t)
    story.append(Spacer(1, 12))

    # Top transactions table (limit to 20)
    story.append(Paragraph("<b>Top Transactions (sample)</b>", styles["Heading3"]))
    txn_data = [["Date", "Description", "Amount", "Category"]]
    for _, r in df.head(20).iterrows():
        txn_data.append([
            str(r.get("Date") or r.get("txn_date") or ""),
            (r.get("Description") or str(r.get("description") or "") )[:60],
            str(r.get("Amount") or r.get("amount") or ""),
            str(r.get("Predicted Category") or r.get("category") or "")
        ])
    tt = Table(txn_data, colWidths=[25*mm, 85*mm, 25*mm, 30*mm])
    tt.setStyle(TableStyle([
        ('GRID', (0,0), (-1,-1), 0.25, colors.grey),
        ('VALIGN', (0,0), (-1,-1), 'TOP'),
    ]))
    story.append(tt)

    doc.build(story)
    pdf_bytes = buffer.getvalue()
    buffer.close()
    return pdf_bytes

# ----------------------------
# Sidebar
# ----------------------------
st.sidebar.title("Smart Expense Advisor")
st.sidebar.caption("Upload → Classify → Analyze")
theme_choice = st.sidebar.radio("Theme", ["Light", "Dark"])
if theme_choice == "Dark":
    st.markdown(DARK_CSS, unsafe_allow_html=True)
else:
    st.markdown(LIGHT_CSS, unsafe_allow_html=True)

st.sidebar.markdown("---")
st.sidebar.header("Actions")
action = st.sidebar.selectbox("Choose", ["Upload & Predict", "Analytics", "Download Report", "About"])

# ----------------------------
# Main: Upload & Predict
# ----------------------------
if action == "Upload & Predict":
    st.header("📤 Upload Bank Statement (CSV)")
    st.markdown("CSV must contain a text column (e.g., `Description`, `description`, `Narration`) and optionally date/amount columns.")
    uploaded = st.file_uploader("Upload CSV file", type=["csv"], accept_multiple_files=False)

    if uploaded:
        df = pd.read_csv(uploaded)
        st.subheader("Preview")
        st.dataframe(df.head(6))

        # Try to detect text column
        text_cols = [c for c in df.columns if c.lower() in ("description", "desc", "narration", "remarks", "remarks", "transaction", "particulars")]
        if not text_cols:
            # fallback: ask user to pick
            st.warning("Could not auto-detect a description column. Please select the column containing transaction text.")
            chosen = st.selectbox("Select text column", list(df.columns))
            text_col = chosen
        else:
            text_col = text_cols[0]
            st.info(f"Auto-detected text column: **{text_col}** (change from sidebar to override)")

        # optional column mapping in sidebar
        with st.expander("Column mapping & options"):
            st.write("If the automatic choices are wrong, select columns manually.")
            date_col = st.selectbox("Date column (optional)", [None] + list(df.columns), index=0)
            amount_col = st.selectbox("Amount column (optional)", [None] + list(df.columns), index=0)
            text_col = st.selectbox("Text/Description column", [text_col] + [c for c in df.columns if c != text_col], index=0)
            sample_rows = st.slider("Rows to preview", 3, 20, 6)

        if st.button("Run classification"):
            # fill and clean
            df[text_col] = df[text_col].astype(str).fillna("")
            # predict
            X = tfidf.transform(df[text_col])
            preds = model.predict(X)
            df["Predicted Category"] = preds

            # try unify date/amount column names if present
            if date_col:
                df["Date"] = df[date_col].apply(parse_date_safe)
            if amount_col:
                df["Amount"] = df[amount_col]

            st.success("✅ Classification complete")
            st.dataframe(df.head(sample_rows), use_container_width=True)

            # save to session state
            st.session_state["classified_df"] = df

# ----------------------------
# Analytics page
# ----------------------------
elif action == "Analytics":
    st.header("📊 Analytics")
    df = st.session_state.get("classified_df", None)
    if df is None:
        st.warning("No classified data in session. Upload and run classification first.")
        st.stop()

    # Ensure predicted column exists
    if "Predicted Category" not in df.columns:
        st.error("No Predicted Category found — run classification first.")
        st.stop()

    # filter controls
    st.sidebar.markdown("---")
    st.sidebar.subheader("Filters")
    categories = list(df["Predicted Category"].unique())
    selected_cats = st.sidebar.multiselect("Categories", categories, default=categories)
    date_from = st.sidebar.date_input("From", value=df.get("Date").dropna().min() if "Date" in df else pd.to_datetime("2023-01-01").date())
    date_to = st.sidebar.date_input("To", value=df.get("Date").dropna().max() if "Date" in df else pd.to_datetime("today").date())

    # apply filters
    dff = df[df["Predicted Category"].isin(selected_cats)]
    if "Date" in dff.columns:
        dff = dff[(dff["Date"].notnull()) & (dff["Date"] >= date_from) & (dff["Date"] <= date_to)]

    st.subheader("Category Distribution")
    cat_counts = dff["Predicted Category"].value_counts()
    fig = px.pie(names=cat_counts.index, values=cat_counts.values, hole=0.35, title="Category Share")
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Monthly Spend (if Amount present)")
    if "Amount" in dff.columns:
        # ensure numeric
        dff["Amount"] = pd.to_numeric(dff["Amount"], errors="coerce")
        dff = dff.dropna(subset=["Amount"])
        # if date present
        if "Date" in dff.columns:
            monthly = dff.groupby(pd.Grouper(key="Date", freq="M"))["Amount"].sum().reset_index()
            monthly["month"] = monthly["Date"].dt.to_period("M").astype(str)
            fig2 = px.line(monthly, x="month", y="Amount", markers=True, title="Monthly Total Spend")
            st.plotly_chart(fig2, use_container_width=True)
        else:
            st.info("Date column not found - cannot produce monthly trend.")
    else:
        st.info("Amount column not found - upload a file with an amount column to see spend trends.")

    st.subheader("Top Merchants / Items")
    text_field = None
    # try merchant-like fields
    txt_candidates = [c for c in df.columns if c.lower() in ("merchant", "description", "narration", "remarks", "particulars")]
    if txt_candidates:
        text_field = txt_candidates[0]
        top_merchants = dff[text_field].value_counts().head(10)
        st.bar_chart(top_merchants)

# ----------------------------
# Download Report
# ----------------------------
elif action == "Download Report":
    st.header("📥 Export / Report")
    df = st.session_state.get("classified_df", None)
    if df is None:
        st.warning("No classified data available. Run classification first.")
        st.stop()

    # CSV / Excel downloads
    csv_bytes = df.to_csv(index=False).encode("utf-8")
    st.download_button("Download CSV", csv_bytes, "classified_expenses.csv", "text/csv")

    try:
        import openpyxl
        excel_buffer = io.BytesIO()
        df.to_excel(excel_buffer, index=False, engine="openpyxl")
        st.download_button("Download Excel", excel_buffer.getvalue(), "classified_expenses.xlsx", "application/vnd.ms-excel")
    except Exception:
        st.info("Excel export not available (missing openpyxl). CSV still available.")

    # PDF summary
    st.markdown("### PDF Summary")
    summary = df["Predicted Category"].value_counts().to_dict()
    if st.button("Generate PDF Report"):
        pdf_bytes = make_pdf_report(df, summary, title="Smart Expense Report")
        st.download_button("Download PDF Report", pdf_bytes, "expense_report.pdf", "application/pdf")

# ----------------------------
# About
# ----------------------------
else:
    st.header("About Smart Expense Advisor")
    st.markdown("""
    **Smart Expense Advisor** — classify expenses, visualize spending and export reports.
    - Upload CSVs
    - ML-based categorization
    - Interactive charts and download options
    """)
    st.markdown("---")
    st.caption("Built with Streamlit • Joblib • scikit-learn • Plotly")
