from flask import Flask, request, jsonify

app = Flask(__name__)  # Initialize the Flask app

# Home endpoint
@app.route('/')
def home():
    return "Welcome to the AI-Powered Symptom Checker!"

# Check endpoint
@app.route('/check', methods=['POST'])
def check_symptoms():
    # Get symptoms from POST request - backend processes this
    data = request.get_json()  # Reads data
    symptoms = data.get('symptoms', [])

    # Placeholder response to be replaced by ML predictions
    diagnosis = "Diagnosis results based on symptoms: " + ", ".join(symptoms)

    # Return response in JSON format
    return jsonify({"diagnosis": diagnosis})

# Test endpoint
@app.route('/test')
def test():
    return jsonify({"message": "The /test endpoint is working!"})

# Run the app
if __name__ == '__main__':
    app.run(debug=True)  # Starts Flask app
