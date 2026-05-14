
from flask import Flask, render_template, request
import joblib
import numpy as np
from utils.helper import get_career_info

app = Flask(__name__)

# Load model and label encoder
model = joblib.load("model/model.pkl")
label_encoder = joblib.load("model/label_encoder.pkl")

feature_names = [
    "python_skill",
    "java_skill",
    "cpp_skill",
    "javascript_skill",
    "html_css_skill",
    "sql_skill",
    "machine_learning_interest",
    "web_development_interest",
    "data_analysis_interest",
    "cybersecurity_interest",
    "statistics_skill",
    "problem_solving",
    "communication_skill",
    "creativity",
    "logical_thinking",
    "teamwork",
    "cloud_computing_interest",
    "networking_skill",
    "debugging_skill",
    "aptitude_score"
]

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict")
def predict_page():
    return render_template("predict.html")

@app.route("/dashboard")
def dashboard():
    return render_template("dashboard.html")

@app.route("/result", methods=["POST"])
def result():
    try:
        input_data = []

        for feature in feature_names:
            value = request.form.get(feature)

            if value is None or value == "":
                return render_template(
                    "result.html",
                    error="Please fill all input fields."
                )

            input_data.append(float(value))

        input_array = np.array([input_data])

        probabilities = model.predict_proba(input_array)[0]
        prediction = model.predict(input_array)[0]

        predicted_career = label_encoder.inverse_transform([prediction])[0]
        confidence = round(max(probabilities) * 100, 2)

        career_info = get_career_info(predicted_career)

        return render_template(
            "result.html",
            prediction=predicted_career,
            confidence=confidence,
            career_info=career_info
        )

    except Exception as e:
        return render_template(
            "result.html",
            error=f"Error occurred: {str(e)}"
        )

if __name__ == "__main__":
    app.run(debug=True)
