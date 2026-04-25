#Product Price Prediction
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

# Dataset
data = {
    'rating': [4.5, 4.0, 3.5, 5.0, 4.2],
    'reviews': [200, 150, 100, 300, 250],
    'discount': [10, 20, 30, 5, 15],
    'price': [2000, 1500, 1200, 3000, 2500]
}

df = pd.DataFrame(data)

# Input & Output
X = df[['rating', 'reviews', 'discount']]
y = df['price']

# Model
model = LinearRegression()
model.fit(X, y)

# User Input
rating = float(input("Enter product rating (1-5): "))
reviews = int(input("Enter number of reviews: "))
discount = float(input("Enter discount (%): "))

# Prediction
pred = model.predict([[rating, reviews, discount]])

print(f"Estimated Product Price: ₹ {pred[0]:.2f}")
