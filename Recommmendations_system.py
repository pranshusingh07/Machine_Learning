#Student Performance Prediction
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

# Dataset: [study_hours, attendance]
X = np.array([
    [2, 50],
    [3, 60],
    [5, 80],
    [1, 30],
    [4, 70],
    [6, 90]
])

# Output: 0 = Fail, 1 = Pass
y = np.array([0, 0, 1, 0, 1, 1])

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Model
model = LogisticRegression()
model.fit(X_train, y_train)

# Prediction
hours = float(input("Enter Study Hours: "))
attendance = float(input("Enter Attendance (%): "))

prediction = model.predict([[hours, attendance]])

if prediction == 1:
    print("Student will PASS ✅")
else:
    print("Student will FAIL ❌")
