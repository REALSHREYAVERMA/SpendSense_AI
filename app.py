
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest

# Page settings
st.set_page_config(
    page_title="SpendSense AI",
    page_icon="💰",
    layout="wide"
)

# Custom styling
# Custom styling
st.markdown("""
<style>
.stApp {
    background-color: #0e1117;
}

h1, h2, h3, p, label {
    color: white;
}

[data-testid="stMetricValue"] {
    color: #7CFCB2;
}

</style>
""", unsafe_allow_html=True)

# Project title
st.title("💰 SpendSense AI")
st.subheader("Your Smart Expense & Subscription Detector")
st.write("Track spending, identify recurring payments and detect unusual expenses.")

# Sidebar
st.sidebar.title("Dashboard Menu")
st.sidebar.write("Upload your transaction data")

uploaded_file = st.sidebar.file_uploader(
    "Upload CSV file",
    type=["csv"]
)

# Load data
if uploaded_file is not None:
    original_df = pd.read_csv(uploaded_file)
else:
    original_df = pd.read_csv("transactions.csv")

required_columns = ["Date", "Description", "Amount", "Category"]

if not all(column in original_df.columns for column in required_columns):
    st.error("Your CSV must contain Date, Description, Amount and Category columns.")
    st.stop()

original_df["Date"] = pd.to_datetime(original_df["Date"], errors="coerce")
original_df["Amount"] = pd.to_numeric(original_df["Amount"], errors="coerce")
original_df = original_df.dropna(subset=required_columns)

# Remove duplicate transactions for main analysis
df = original_df.drop_duplicates(
    subset=["Date", "Description", "Amount"]
).copy()

# Summary
st.markdown("## 📊 Spending Overview")

col1, col2, col3 = st.columns(3)

with col1:
    st.metric("Total Spending", f"₹{df['Amount'].sum():,.0f}")

with col2:
    st.metric("Total Transactions", len(df))

with col3:
    st.metric("Unique Categories", df["Category"].nunique())

st.divider()

# Charts
st.markdown("## 📈 Expense Analysis")

col1, col2 = st.columns(2)

with col1:
    st.markdown("### Monthly Spending")
    monthly = df.groupby(df["Date"].dt.to_period("M"))["Amount"].sum()
    monthly.index = monthly.index.astype(str)
    st.bar_chart(monthly)

with col2:
    st.markdown("### Category-wise Spending")
    category = df.groupby("Category")["Amount"].sum()
    st.bar_chart(category)

# Financial Insights
st.markdown("## 💡 Smart Financial Insights")

category_totals = df.groupby("Category")["Amount"].sum()
top_category = category_totals.idxmax()
top_category_amount = category_totals.max()

monthly_totals = df.groupby(
    df["Date"].dt.to_period("M")
)["Amount"].sum()

highest_month = monthly_totals.idxmax()
highest_month_amount = monthly_totals.max()

average_transaction = df["Amount"].mean()

col1, col2, col3 = st.columns(3)

with col1:
    st.metric("Highest Spending Category", top_category)

with col2:
    st.metric("Highest Spending Month", str(highest_month))

with col3:
    st.metric("Average Transaction", f"₹{average_transaction:,.0f}")

st.info(
    f"Your highest spending category is {top_category}, "
    f"with total spending of ₹{top_category_amount:,.0f}."
)

st.info(
    f"Your highest spending month is {highest_month}, "
    f"with spending of ₹{highest_month_amount:,.0f}."
)

st.divider()

# Recurring payments
st.markdown("## 🔁 Recurring Payment Detection")

recurring = df.groupby("Description").filter(
    lambda x: x["Date"].dt.to_period("M").nunique() >= 2
)

if not recurring.empty:
    recurring_summary = recurring.groupby("Description").agg(
        Payments=("Amount", "count"),
        Average_Amount=("Amount", "mean")
    ).reset_index()

    st.dataframe(recurring_summary, use_container_width=True)
else:
    st.info("No recurring payments found.")

# Price increase detection
st.markdown("## 📈 Price Increase Alerts")

increases = []

for name, group in df.groupby("Description"):
    group = group.sort_values("Date")
    old_price = group["Amount"].iloc[0]
    new_price = group["Amount"].iloc[-1]

    if new_price > old_price:
        increases.append({
            "Description": name,
            "Previous Amount": old_price,
            "Latest Amount": new_price,
            "Increase": new_price - old_price
        })

if increases:
    st.dataframe(pd.DataFrame(increases), use_container_width=True)
else:
    st.success("No price increases detected.")

# Duplicate detection
st.markdown("## 🔍 Duplicate Transaction Detection")

duplicates = original_df[
    original_df.duplicated(
        subset=["Date", "Description", "Amount"],
        keep=False
    )
]

if not duplicates.empty:
    st.warning(f"{len(duplicates)} duplicate transaction rows found.")
    st.dataframe(duplicates, use_container_width=True)
else:
    st.success("No duplicate transactions found.")

# Machine learning

    

# Machine Learning: Anomaly Detection
st.markdown("## 🤖 AI Unusual Expense Detection")

if len(df) >= 5:
    ml_df = df.copy()

    # Convert categories into separate numerical columns
    category_encoded = pd.get_dummies(
        ml_df["Category"],
        dtype=int
    )

    # Combine amount and category information
    features = pd.concat(
        [ml_df[["Amount"]], category_encoded],
        axis=1
    )

    # Train the Isolation Forest model
    model = IsolationForest(
        contamination=0.1,
        random_state=42
    )

    ml_df["Anomaly"] = model.fit_predict(features)

    # Select unusual transactions
    unusual = ml_df[ml_df["Anomaly"] == -1].copy()

    if not unusual.empty:
        st.warning("The model identified statistically unusual transactions.")

        unusual["Reason"] = unusual.apply(
            lambda row: (
                f"₹{row['Amount']:,.0f} transaction in "
                f"{row['Category']} was flagged by the model."
            ),
            axis=1
        )

        st.dataframe(
            unusual[["Date", "Description", "Amount", "Category", "Reason"]],
            width="stretch"
        )

        st.info(
            "Isolation Forest detects transactions that differ from "
            "the patterns in the dataset. A flagged transaction is "
            "not necessarily fraudulent."
        )
    else:
        st.success("No unusual transactions detected.")

else:
    st.info("Add at least 5 transactions for anomaly detection.")



    

# Transaction history

# Interactive Transaction History
st.divider()
st.markdown("## 📋 Transaction History")

col1, col2 = st.columns(2)

with col1:
    selected_category = st.selectbox(
        "Filter by Category",
        ["All Categories"] + sorted(df["Category"].unique().tolist())
    )

with col2:
    month_options = sorted(
        df["Date"].dt.strftime("%B %Y").unique().tolist()
    )

    selected_month = st.selectbox(
        "Filter by Month",
        ["All Months"] + month_options
    )

filtered_df = df.copy()

if selected_category != "All Categories":
    filtered_df = filtered_df[
        filtered_df["Category"] == selected_category
    ]

if selected_month != "All Months":
    filtered_df = filtered_df[
        filtered_df["Date"].dt.strftime("%B %Y") == selected_month
    ]

st.write("Filtered Transactions:", len(filtered_df))

st.dataframe(
    filtered_df.sort_values("Date", ascending=False),
    width="stretch"
)

# Download report
csv_data = filtered_df.to_csv(index=False).encode("utf-8")

st.download_button(
    label="📥 Download Expense Report",
    data=csv_data,
    file_name="SpendSense_Expense_Report.csv",
    mime="text/csv"
)

st.caption("SpendSense AI | Smart Expense Analysis Project")




