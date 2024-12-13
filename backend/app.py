from flask import Flask, request, jsonify

app = Flask(__name__)  # Initialize the Flask app

@app.route('/')
def home():
    return "Welcome to the AI-Powered Symptom Checker!"

if __name__ == '__main__':
    app.run(debug=True)  # Run the app

### check endpoint

@app.route('/check' ,methods=['POST'])
def check_symptoms():
    
    #get symptoms from post request - backend processes this
    data = request.get_json()
    symptoms = data.get('symptoms', [])

    #placeholders response that are to be replaced by ML predictions 
    diagnosis = "Diagnosis results based on symptoms: " + ", ".join(symptoms)

    #return response as JSON format

    return jsonify({"diagnosis": diagnosis})