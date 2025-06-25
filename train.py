import numpy as np
from sklearn.tree import DecisionTreeClassifier
import joblib

X = np.array([[1], [3], [6], [10], [14], [18], [25], [30]])
y = np.array([0, 0, 1, 1, 1, 2, 2, 2])  # 0=low, 1=medium, 2=high

model = DecisionTreeClassifier()
model.fit(X, y)

joblib.dump(model, 'ad_selector_model.pkl')
print("Model saved as ad_selector_model.pkl")
