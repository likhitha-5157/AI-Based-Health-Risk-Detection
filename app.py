from flask import Flask, render_template, request
import pickle
import numpy as np
import os

# Create Flask app
app = Flask(__name__)

# Load trained model
model = pickle.load(open("model.pkl", "rb"))

# Home route
@app.route("/")
def home():
    return render_template("index.html")

# Prediction route
@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Get input values from form
        input_features = [float(x) for x in request.form.values()]
        
        # Convert to numpy array
        final_features = np.array([input_features])
        
        # Prediction
        prediction = model.predict(final_features)

        # Output result
        if prediction[0] == 1:
            output = "High Risk of Diabetes"
        else:
            output = "Low Risk of Diabetes"

        return render_template("index.html", prediction_text=output)

    except Exception as e:
        return render_template("index.html", prediction_text="Error: " + str(e))


# Run app (IMPORTANT for Render)
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))  # dynamic port for deployment
    app.run(host="0.0.0.0", port=port)
