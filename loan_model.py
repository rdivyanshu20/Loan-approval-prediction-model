import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import joblib

# 1. Generate Synthetic Loan Data
np.random.seed(42)
data_size = 1000

data = {
    'ApplicantIncome': np.random.randint(20000, 150000, data_size),
    'CoapplicantIncome': np.random.randint(0, 50000, data_size),
    'LoanAmount': np.random.randint(50000, 500000, data_size),
    'Credit_History': np.random.choice([0, 1], data_size, p=[0.2, 0.8]),
    'Dependents': np.random.randint(0, 4, data_size),
    'Property_Area': np.random.choice(['Urban', 'Semiurban', 'Rural'], data_size)
}

df = pd.DataFrame(data)

# Target Variable Logic: Good credit + decent income = approval
df['Loan_Status'] = np.where(
    (df['Credit_History'] == 1) & (df['ApplicantIncome'] > 40000), 1, 0
)

# 2. Data Preprocessing
# Convert categorical variable (Property_Area) into dummy/indicator variables
df = pd.get_dummies(df, columns=['Property_Area'], drop_first=True)

X = df.drop('Loan_Status', axis=1)
y = df['Loan_Status']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 3. Model Training
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# 4. Evaluation
predictions = model.predict(X_test)
print(f"Model Accuracy: {accuracy_score(y_test, predictions):.2f}")
print("\nClassification Report:\n", classification_report(y_test, predictions))

# 5. Export the Model
joblib.dump(model, 'loan_approval_model.pkl')
print("Success: Model saved as 'loan_approval_model.pkl'")
