gr# saas-churn-analytics-platform
Description
End to end Saas churn analytics project using python ,SQL, visualization and machine learning to predict churn and improve customer retention.
SaaS-Customer-Churn-Prediction/
│
├── README.md
│
├── docs/
│   ├── Industry_Ready_SaaS_Churn_Project.docx
│   ├── Project_Report.pdf
│   └── Presentation.pptx
│
├── data/
│   ├── raw/
│   │   └── saas_customer_data.csv
│   │
│   └── processed/
│       └── cleaned_saas_data.csv
│
├── notebooks/
│   ├── 01_Data_Exploration.ipynb
│   ├── 02_Metrics_Calculation.ipynb
│   ├── 03_Data_Visualization.ipynb
│   └── 04_Churn_Prediction_Model.ipynb
│
├── src/
│   ├── data_preprocessing.py
│   ├── metrics.py
│   ├── churn_model.py
│   └── retention_strategies.py
│
├── sql/
│   ├── churn_analysis.sql
│   ├── mrr_calculation.sql
│   └── high_risk_customers.sql
│
├── visuals/
│   ├── churn_distribution.png
│   ├── mrr_trend.png
│   ├── engagement_vs_churn.png
│   ├── cltv_distribution.png
│   └── support_tickets_vs_churn.png
│
├── requirements.txt
│
└── .gitignore
"""
SaaS Customer Churn Prediction & Retention Strategy
Methodology, KPIs, and Business Insights (Python Format)
Author: Harsh Verma
"""

# =========================
# 1. IMPORT LIBRARIES
# =========================
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression


# =========================
# 2. DATA LOADING
# =========================
def load_data(path: str) -> pd.DataFrame:
    """
    Load SaaS customer dataset
    """
    df = pd.read_csv(path)
    return df


# =========================
# 3. DATA CLEANING
# =========================
def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Handle missing values and duplicates
    """
    df = df.drop_duplicates()
    df = df.fillna(0)
    return df


# =========================
# 4. KPI / METRICS FUNCTIONS
# =========================
def customer_churn_rate(df: pd.DataFrame) -> float:
    """
    Churn Rate = Lost Customers / Total Customers
    """
    return (df[df["churn"] == 1].shape[0] / df.shape[0]) * 100


def revenue_churn_rate(df: pd.DataFrame) -> float:
    """
    Revenue Churn (MRR Churn)
    """
    lost_mrr = df[df["churn"] == 1]["monthly_revenue"].sum()
    total_mrr = df["monthly_revenue"].sum()
    return (lost_mrr / total_mrr) * 100


def retention_rate(df: pd.DataFrame) -> float:
    """
    Retention Rate
    """
    retained = df[df["churn"] == 0].shape[0]
    total = df.shape[0]
    return (retained / total) * 100


def monthly_recurring_revenue(df: pd.DataFrame) -> float:
    """
    Monthly Recurring Revenue (MRR)
    """
    return df["monthly_revenue"].sum()


def cltv(df: pd.DataFrame) -> float:
    """
    Customer Lifetime Value (CLTV)
    """
    arpu = df["monthly_revenue"].mean()
    churn = customer_churn_rate(df)
    return arpu / churn if churn != 0 else 0


def cac(total_marketing_cost: float, new_customers: int) -> float:
    """
    Customer Acquisition Cost (CAC)
    """
    return total_marketing_cost / new_customers


# =========================
# 5. CHURN PREDICTION MODEL
# =========================
def churn_prediction_model(df: pd.DataFrame):
    """
    Logistic Regression Churn Prediction Model
    """
    X = df[["login_frequency", "support_tickets", "tenure"]]
    y = df["churn"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = LogisticRegression()
    model.fit(X_train, y_train)

    return model


# =========================
# 6. BUSINESS INSIGHTS
# =========================
def generate_insights(df: pd.DataFrame) -> dict:
    """
    Generate key churn insights
    """
    insights = {}

    insights["low_engagement_churn"] = (
        df[df["login_frequency"] < 5]["churn"].mean()
    )

    insights["high_support_ticket_churn"] = (
        df[df["support_tickets"] > 5]["churn"].mean()
    )

    insights["long_tenure_churn"] = (
        df[df["tenure"] > 24]["churn"].mean()
    )

    return insights


# =========================
# 7. PIPELINE EXECUTION
# =========================
if __name__ == "__main__":
    df = load_data("saas_data.csv")
    df = clean_data(df)

    print("Customer Churn Rate:", customer_churn_rate(df))
    print("Revenue Churn Rate:", revenue_churn_rate(df))
    print("Retention Rate:", retention_rate(df))
    print("MRR:", monthly_recurring_revenue(df))
    print("CLTV:", cltv(df))

    model = churn_prediction_model(df)

    insights = generate_insights(df)
    print("Business Insights:", insights)
| KPI                | Purpose                         |
| ------------------ | ------------------------------- |
| Churn Rate         | Measures customer loss          |
| MRR Churn          | Measures revenue loss           |
| Retention Rate     | Measures loyalty                |
| MRR                | Measures revenue stability      |
| CLTV               | Measures long-term value        |
| CAC                | Measures acquisition efficiency |
| Engagement Metrics | Predict churn                   |
| NPS & CSAT         | Measure satisfaction            |
DATA SET:

| Customer ID | Plan       | Monthly Revenue | Tenure (Months) | Login Frequency | Support Tickets | Churn |
| ----------- | ---------- | --------------- | --------------- | --------------- | --------------- | ----- |
| 1001        | Basic      | 20              | 2               | 3               | 4               | 1     |
| 1002        | Pro        | 50              | 12              | 15              | 1               | 0     |
| 1003        | Pro        | 50              | 8               | 10              | 2               | 0     |
| 1004        | Enterprise | 120             | 24              | 25              | 0               | 0     |
| 1005        | Basic      | 20              | 3               | 2               | 5               | 1     |
| 1006        | Pro        | 50              | 10              | 12              | 1               | 0     |
| 1007        | Basic      | 20              | 1               | 1               | 6               | 1     |
| 1008        | Enterprise | 120             | 36              | 30              | 0               | 0     |
| 1009        | Pro        | 50              | 15              | 18              | 1               | 0     |
| 1010        | Basic      | 20              | 4               | 4               | 4               | 1     |
| 1011        | Pro        | 50              | 18              | 20              | 1               | 0     |
| 1012        | Enterprise | 120             | 30              | 28              | 0               | 0     |
| 1013        | Basic      | 20              | 5               | 6               | 3               | 1     |
| 1014        | Pro        | 50              | 14              | 14              | 2               | 0     |
| 1015        | Basic      | 20              | 6               | 5               | 3               | 1     |
| 1016        | Enterprise | 120             |                 |                 |                 |       |

Graphical representation


Customer churn distribution
<img width="640" height="480" alt="Image" src="https://github.com/user-attachments/assets/3b64e8b1-2e76-432b-8469-fec85e11dc90" />

MRR trend Analysis
<img width="640" height="480" alt="Image" src="https://github.com/user-attachments/assets/afac96b6-1616-43f2-814d-4b4e34bc0cc1" />

User Engagement vs churn
<img width="640" height="480" alt="Image" src="https://github.com/user-attachments/assets/be214b88-a4d2-49e4-8e4c-87b677da1fd4" />

Customer Lifetime Value Distribution
<img width="640" height="480" alt="Image" src="https://github.com/user-attachments/assets/836df174-5cfe-42cc-aa93-01101dfd6e51" />

Support Tickets vs Churn
<img width="640" height="480" alt="Image" src="https://github.com/user-attachments/assets/96191e7a-8810-4637-93de-e96d2cb3d590" />





