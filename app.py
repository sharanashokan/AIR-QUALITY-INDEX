from flask import Flask, render_template, request
import joblib

app = Flask(__name__)
model = joblib.load("air_quality_model.joblib")

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Retrieve input values from the form
    so2 = float(request.form['so2'])
    no2 = float(request.form['no2'])
    rspm = float(request.form['rspm'])
    spm = float(request.form['spm'])

    # Reshape the input data into a 2D array
    input_data = [[so2, no2, rspm, spm]]

    # Perform prediction using the loaded model
    prediction = model.predict(input_data)[0]
    
    # Assign prediction range and corresponding tag
    if prediction <= 50:
        prediction_range = "Good"
    elif prediction <= 100:
        prediction_range = "Satisfactory"
    elif prediction <= 200:
        prediction_range = "Moderate"
    elif prediction <= 300:
        prediction_range = "Poor"
    elif prediction <= 400:
        prediction_range = "Very Poor"
    else:
        prediction_range = "Severe"

    return render_template('index.html', prediction=prediction, prediction_range=prediction_range)

if __name__ == '__main__':
    app.run()
