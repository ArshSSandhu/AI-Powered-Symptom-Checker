from flask import Flask, request, jsonify
import joblib
import os

app = Flask(__name__)  # Initialize the Flask app

#Loading : a) model b) vectorizor c) label encoder


model = joblib.load(os.path.join(os.path.dirname(__file__), 'trained_model.pkl'))

#to load them, we need to go one level up
vectorizer = joblib.load(os.path.join(os.path.dirname(__file__), '..', 'vectorizer.pkl'))
label_encoder = joblib.load(os.path.join(os.path.dirname(__file__), '..', 'label_encoder.pkl'))

# Home endpoint
@app.route('/')
def home():
    return "Welcome to the AI-Powered Symptom Checker!"

# Test endpoint
@app.route('/test')
def test():
    return jsonify({"message": "The /test endpoint is working!"})

# Check endpoint
@app.route('/check', methods=['POST'])
def check_symptoms():
    # Get symptoms from POST request - backend processes this
    data = request.get_json()  # Reads data
    
    input_text = data.get('symptoms', '')

    #vectorize and predict

    vectorized_input = vectorizer.transform([input_text])
    predicted_label = model.predict(vectorized_input)[0]
    predicted_disease = label_encoder.inverse_transform([predicted_label])[0]

    return jsonify({"predicted_disease": predicted_disease})


# Run the app
if __name__ == '__main__':
    app.run(debug=True)  # Starts Flask app
