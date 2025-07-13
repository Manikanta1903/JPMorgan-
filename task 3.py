import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score

# Load your data
df = pd.read_csv("Task 3 and 4_Loan_Data.csv")

# Features and target
features = [
    'credit_lines_outstanding', 'loan_amt_outstanding',
    'total_debt_outstanding', 'income',
    'years_employed', 'fico_score'
]
target = 'default'

X = df[features]
y = df[target]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale features for Logistic Regression
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train Logistic Regression
logreg = LogisticRegression()
logreg.fit(X_train_scaled, y_train)
logreg_probs = logreg.predict_proba(X_test_scaled)[:, 1]

# Train Random Forest
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)
rf5_probs = rf.predict_proba(X_test)[:, 1]

# Evaluate models
logreg_auc = roc_auc_score(y_test, logreg_probs)
rf_auc = roc_auc_score(y_test, rf5_probs)

print("Logistic Regression AUC:", logreg_auc)
print("Random Forest AUC:", rf_auc)



def predict_expected_loss(model, scaler, input_data):
    """
    input_data: dict with keys matching the feature names
    """
    import numpy as np

    features = [
        'credit_lines_outstanding', 'loan_amt_outstanding',
        'total_debt_outstanding', 'income',
        'years_employed', 'fico_score'
    ]

    X_new = pd.DataFrame([input_data])[features]
    X_scaled = scaler.transform(X_new)

    pd_default = model.predict_proba(X_scaled)[0][1]
    expected_loss = pd_default * input_data['loan_amt_outstanding'] * 0.90

    return {
        'PD': pd_default,
        'Expected_Loss': expected_loss
    }

