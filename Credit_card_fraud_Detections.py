#Credit Card Fraud Detection
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix

# Sample Dataset
data = {
    'amount': [100, 2000, 150, 3000, 500, 7000, 50, 9000],
    'time': [1, 2, 1, 3, 2, 4, 1, 5],
    'location': [0, 1, 0, 1, 0, 1, 0, 1],  # 0 = local, 1 = foreign
    'fraud': [0, 1, 0, 1, 0, 1, 0, 1]
}

df = pd.DataFrame(data)

# Features & Label
X = df[['amount', 'time', 'location']]
y = df['fraud']

# Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Model
model = LogisticRegression()
model.fit(X_train, y_train)

# Prediction
amount = float(input("Enter transaction amount: "))
time = int(input("Enter time (1-5): "))
location = int(input("Location (0=local, 1=foreign): "))

pred = model.predict([[amount, time, location]])

if pred == 1:
    print("⚠️ Fraud Transaction Detected")
else:
    print("✅ Genuine Transaction")

# Accuracy
y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))

# Confusion Matrix
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
