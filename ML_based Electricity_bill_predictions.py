#ML Based Electricity Bill Prediction
import numpy as np
from sklearn.linear_model import LinearRegression

# Dataset: Units vs Bill
X = np.array([[50],[100],[150],[200],[250],[300]])
y = np.array([75,150,300,500,700,900])

# Model
model = LinearRegression()
model.fit(X, y)

# Prediction
units = float(input("Enter units: "))
predicted_bill = model.predict([[units]])

print("Predicted Electricity Bill = ₹", predicted_bill[0])
