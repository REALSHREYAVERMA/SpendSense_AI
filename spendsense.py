import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest

# Load dataset
df = pd.read_csv("transactions.csv")

# Convert date column
df["Date"] = pd.to_datetime(df["Date"])

# Display dataset information
print("SPENDSENSE AI")
print("--------------------")

print("\nTransaction Data:")
print(df)

print("\nDataset Information:")
print(df.info())

print("\nTotal Transactions:", len(df))
print("Total Spending: ₹", df["Amount"].sum())

# DATA PREPROCESSING

print("\nDATA PREPROCESSING")
print("--------------------")

# Check missing values
print("\nMissing Values:")
print(df.isnull().sum())

# Check duplicate transactions
duplicates = df.duplicated(
    subset=["Date", "Description", "Amount"]
)

print("\nDuplicate Transactions:")
print(df[duplicates])

# Remove duplicate transactions
df = df.drop_duplicates(
    subset=["Date", "Description", "Amount"]
)

# Reset index
df = df.reset_index(drop=True)

print("\nData After Cleaning:")
print(df)

print("\nTotal Transactions After Cleaning:", len(df))


# SUBSCRIPTION DETECTION

print("\nSUBSCRIPTION DETECTION")
print("----------------------")

subscriptions = df.groupby("Description").filter(
    lambda x: x["Date"].dt.month.nunique() >= 2
)

subscription_summary = subscriptions.groupby("Description").agg(
    Total_Payments=("Amount", "count"),
    Average_Amount=("Amount", "mean")
)

print(subscription_summary)

# PRICE INCREASE DETECTION

print("\nPRICE INCREASE DETECTION")
print("------------------------")

for name, group in df.groupby("Description"):

    group = group.sort_values("Date")

    old_price = group["Amount"].iloc[0]
    new_price = group["Amount"].iloc[-1]

    if new_price > old_price:

        increase = new_price - old_price

        print(name, "Price Increased!")
        print("Old Price:", old_price)
        print("New Price:", new_price)
        print("Increase:", increase)

        # DUPLICATE TRANSACTION DETECTION

print("\nDUPLICATE TRANSACTION DETECTION")
print("--------------------------------")

original_df = pd.read_csv("transactions.csv")

duplicates = original_df[
    original_df.duplicated(
        subset=["Date", "Description", "Amount"],
        keep=False
    )
]

print("Duplicate Transactions Found:")
print(duplicates)

# MACHINE LEARNING - ANOMALY DETECTION

print("\nMACHINE LEARNING: UNUSUAL EXPENSE DETECTION")
print("--------------------------------------------")

model = IsolationForest(
    contamination=0.1,
    random_state=42
)

df["Anomaly"] = model.fit_predict(df[["Amount"]])

df["Result"] = df["Anomaly"].map({
    1: "Normal",
    -1: "Unusual Expense"
})

print(df[["Description", "Amount", "Result"]])

print("\nUnusual Transactions:")
print(df[df["Anomaly"] == -1][["Description", "Amount"]])

# MONTHLY EXPENSE ANALYSIS

print("\nMONTHLY EXPENSE ANALYSIS")
print("------------------------")

df["Month"] = df["Date"].dt.strftime("%B")

monthly_expenses = df.groupby("Month")["Amount"].sum()

print(monthly_expenses)

# Create graph
monthly_expenses.plot(kind="bar")

plt.title("Monthly Expense Analysis")
plt.xlabel("Month")
plt.ylabel("Total Spending (₹)")

plt.tight_layout()
plt.show()

# CATEGORY-WISE EXPENSE ANALYSIS

print("\nCATEGORY-WISE EXPENSE ANALYSIS")

category_expenses = df.groupby("Category")["Amount"].sum()

print(category_expenses)

category_expenses.plot(kind="bar")

plt.title("Spending by Category")
plt.xlabel("Category")
plt.ylabel("Total Amount (₹)")

plt.tight_layout()
plt.show()

print("\nCATEGORY-WISE EXPENSE ANALYSIS")
print("-------------------------------")

category_expenses = df.groupby("Category")["Amount"].sum()

print(category_expenses)

category_expenses.plot(kind="bar")
plt.title("Spending by Category")
plt.xlabel("Category")
plt.ylabel("Total Spending (₹)")
plt.tight_layout()
plt.show()




