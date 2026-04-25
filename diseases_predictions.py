#Disease Prediction using 
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

# Dataset: [fever, cough, fatigue]
# 0 = No, 1 = Yes
X = np.array([
    [1, 1, 1],
    [1, 1, 0],
    [0, 1, 1],
    [0, 0, 0],
    [1, 0, 1],
    [0, 1, 0]
])

# Output: 1 = Disease, 0 = No Disease
y = np.array([1, 1, 1, 0, 1, 0])

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Model
model = DecisionTreeClassifier()
model.fit(X_train, y_train)

# User Input
fever = int(input("Fever (1/0): "))
cough = int(input("Cough (1/0): "))
fatigue = int(input("Fatigue (1/0): "))

prediction = model.predict([[fever, cough, fatigue]])

if prediction == 1:
    print("⚠️ Patient may have disease")
else:
    print("✅ Patient is healthy")
