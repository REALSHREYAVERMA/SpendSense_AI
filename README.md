
# 💰 SpendSense AI – Hidden Subscription & Expense Detector

## 📌 About the Project

SpendSense AI is a Python-based expense analysis and anomaly detection project. It helps users understand their spending habits, identify recurring payments, detect possible price increases, find duplicate transactions, and flag unusual expenses using Machine Learning.

## ✨ Features

- **Expense Analysis:** Calculates total spending and analyzes expenses by month and category.
- **Recurring Payment Detection:** Identifies merchants that appear across multiple months.
- **Price Increase Alerts:** Compares previous and latest transaction amounts.
- **Duplicate Detection:** Finds repeated transactions.
- **AI Anomaly Detection:** Uses the Isolation Forest algorithm to identify statistically unusual transactions.
- **Interactive Dashboard:** Displays insights, charts, filters and transaction history.
- **CSV Upload:** Allows users to upload their own transaction data.
- **Expense Report:** Allows users to download filtered transaction records.

## 🛠️ Technologies Used

- Python
- Pandas
- NumPy
- Matplotlib
- Scikit-learn
- Streamlit

## 🤖 Machine Learning Algorithm

This project uses the Isolation Forest algorithm, an unsupervised Machine Learning technique for anomaly detection.

It identifies transactions that differ from patterns in the dataset. A flagged transaction is not necessarily fraudulent.

## 📂 Project Files

- `app.py` – Streamlit dashboard
- `spendsense.py` – Data analysis and visualizations
- `transactions.csv` – Sample transaction dataset
- `README.md` – Project documentation

## ▶️ How to Run the Project

1. Install the required libraries:

   `pip install pandas numpy matplotlib scikit-learn streamlit`

2. Open the project folder in VS Code.

3. Run the dashboard:

   `python -m streamlit run app.py`

4. Open the local URL displayed in the terminal.

## 📊 Dataset Format

The CSV file should contain these columns:

- Date
- Description
- Amount
- Category

## 🎯 Project Objective

The objective of SpendSense AI is to make personal expense tracking more informative by combining data analysis, visualization and Machine Learning.

