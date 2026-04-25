#Weather Prediction System
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

# Dataset: Day vs Temperature
data = {
    'day': [1,2,3,4,5,6,7],
    'temp': [30,32,31,35,36,38,40]
}

df = pd.DataFrame(data)

# Input & Output
X = df[['day']]
y = df['temp']

# Model
model = LinearRegression()
model.fit(X, y)

# Prediction
day = int(input("Enter future day: "))
pred = model.predict([[day]])

print(f"Predicted Temperature for day {day}: {pred[0]:.2f}°C")

# Graph
plt.scatter(X, y)
plt.plot(X, model.predict(X))
plt.xlabel("Day")
plt.ylabel("Temperature")
plt.title("Weather Prediction")
plt.show()
