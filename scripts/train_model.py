# Step 1: Import libraries
import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from xgboost import XGBClassifier
import joblib
import shap
import matplotlib.pyplot as plt

# Step 2: Load dataset
data_path = os.path.join("data", "cleaned_churn_data.csv")
df = pd.read_csv(data_path)

# Step 3: Handle missing and categorical data
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
df.dropna(inplace=True)

# Encode 'Churn' column as binary target
df['Churn'] = df['Churn'].map({'Yes': 1, 'No': 0})

# Identify categorical columns
cat_cols = df.select_dtypes(include=['object']).columns
num_cols = df.select_dtypes(include=['int64', 'float64']).columns

# Label Encode categorical columns
le = LabelEncoder()
for col in cat_cols:
    df[col] = le.fit_transform(df[col])

# Step 4: Split data
# Drop unnecessary columns before training
X = df.drop(['Churn', 'customerID', 'Churn_Encoded'], axis=1, errors='ignore')


y = df['Churn']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Step 5: Feature scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Save the scaler and feature columns
os.makedirs("models", exist_ok=True)
joblib.dump(scaler, "models/scaler.joblib")
joblib.dump(X.columns.tolist(), "models/feature_columns.joblib")
print("✅ Scaler and feature columns saved successfully.")


# Step 6: Train models
print("Training Logistic Regression...")
log_reg = LogisticRegression(max_iter=1000)
log_reg.fit(X_train_scaled, y_train)


print("Training XGBoost Classifier...")
xgb = XGBClassifier(use_label_encoder=False, eval_metric='logloss')
xgb.fit(X_train_scaled, y_train)

# Step 7: Evaluate models
models = {'Logistic Regression': log_reg, 'XGBoost': xgb}

for name, model in models.items():
    y_pred = model.predict(X_test_scaled)
    acc = accuracy_score(y_test, y_pred)
    print(f"\nModel: {name}")
    print("Accuracy:", round(acc, 4))
    print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
    print("Classification Report:\n", classification_report(y_test, y_pred))

# Step 8: Choose best model (save XGBoost if better)
best_model = xgb  # assuming XGBoost performs better
os.makedirs("models", exist_ok=True)
joblib.dump(best_model, "models/xgb_churn_model.joblib")
print("\n✅ Model saved successfully as models/xgb_churn_model.joblib")

# Step 9: Explainability using SHAP
explainer = shap.TreeExplainer(best_model)
shap_values = explainer.shap_values(X_test_scaled)

# Plot SHAP summary
plt.figure()
shap.summary_plot(shap_values, X_test, show=False)
plt.title("SHAP Summary Plot for XGBoost")
plt.tight_layout()
plt.savefig("models/shap_summary.png")
print("✅ SHAP summary plot saved as models/shap_summary.png")
