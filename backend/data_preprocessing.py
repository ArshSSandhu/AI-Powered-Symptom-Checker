# Import necessary libraries
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import CountVectorizer
import pandas as pd  # Handles CSV and data frames
import numpy as np  # Numerical operations
import joblib  # For saving the encoder/vectorizer

# ---------------- STEP 1: LOAD DATA ----------------
# Load dataset
data = pd.read_csv('./data/Symptom2Disease.csv')

# Drop the Unnamed: 0 column (unnecessary index column)
data = data.drop(['Unnamed: 0'], axis=1)

# ---------------- STEP 2: LABEL ENCODING ----------------
# Encode the 'label' column (diagnosis)
le = LabelEncoder()
data['label'] = le.fit_transform(data['label'])

# Save the encoder for decoding predictions later
joblib.dump(le, 'label_encoder.pkl')  # Saves the label encoder to a file

# Print the transformed labels to check
print("Encoded Labels:", data['label'].unique())  # Displays numeric labels

# ---------------- STEP 3: TEXT VECTORIZATION ----------------
# Initialize the CountVectorizer
vectorizer = CountVectorizer(stop_words='english')

# Transform the 'text' column into numerical vectors
symptom_features = vectorizer.fit_transform(data['text'])

# Convert the result into a DataFrame
symptoms_df = pd.DataFrame(symptom_features.toarray(), columns=vectorizer.get_feature_names_out())

# Combine the vectorized symptoms with the encoded labels
processed_data = pd.concat([symptoms_df, data['label']], axis=1)

# ---------------- STEP 4: SAVE PROCESSED DATA ----------------

# Save the preprocessed dataset
processed_data.to_csv('./data/processed_data.csv', index=False)

# Save the vectorizer for later use (when predicting new inputs)
joblib.dump(vectorizer, 'vectorizer.pkl')

# ---------------- STEP 5: DISPLAY RESULTS ----------------

# Display the shape of the final dataset
print("Shape of Processed Data:", processed_data.shape)

# Check the first 5 rows to verify data
print(processed_data.head())
