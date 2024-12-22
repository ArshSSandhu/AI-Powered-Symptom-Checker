from sklearn.preprocessing import LabelEncoder
import joblib  # For saving the encoder

import pandas as pd  #handles CSV and data frames
import numpy as np #numerical operations

#load dataset

data = pd.read_csv('./data/Symptom2Disease.csv')

# Drop the Unnamed: 0 column
data = data.drop(['Unnamed: 0'], axis=1)

# Encode the 'label' column
le = LabelEncoder()
data['label'] = le.fit_transform(data['label'])

# Save the encoder for decoding predictions later
joblib.dump(le, 'label_encoder.pkl')  # Saves the label encoder to a file

# Print the transformed labels to check
print(data['label'].unique())  # Displays numeric labels



