#Anomaly Detection
from sklearn.ensemble import IsolationForest
import numpy as np

X = np.array([[1],[2],[3],[100]])

model = IsolationForest()
model.fit(X)

print(model.predict(X))  # -1 = anomaly
