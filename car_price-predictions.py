#Car Price Prediction
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

# Dataset
data = {
    'year': [2015, 2016, 2017, 2018, 2019],
    'km_driven': [50000, 40000, 30000, 20000, 10000],
    'price': [300000, 350000, 400000, 450000, 500000]
}

df = pd.DataFrame(data)

# Input (features) & Output
X = df[['year', 'km_driven']]
y = df['price']

# Model
model = LinearRegression()
model.fit(X, y)

# User Input
year = int(input("Enter car year: "))
km = int(input("Enter km driven: "))

# Prediction
pred = model.predict([[year, km]])

print(f"Estimated Car Price: ₹ {pred[0]:.2f}")
