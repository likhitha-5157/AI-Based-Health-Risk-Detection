import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import pickle

# Load dataset
data = pd.read_csv("C:/Users/Dell/Downloads/health_prediction_project/diabetes.csv")

# Handle missing values
data = data.fillna(data.mean())

# Split data
X = data.drop("Outcome", axis=1)
y = data["Outcome"]

# Train test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train model
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# Test model
pred = model.predict(X_test)
accuracy = accuracy_score(y_test, pred)

print("Model Accuracy:", accuracy)

# Save model
pickle.dump(model, open("model.pkl", "wb"))