# Import necessary libraries
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import pandas as pd
import joblib

# ---------------- STEP 1: LOAD PREPROCESSED DATA ----------------
# Load the processed dataset
data = pd.read_csv('../data/processed_data.csv')

# Separate features (X) and target (y)
X = data.drop('label', axis=1)  # All columns except 'label'
y = data['label']               # Target column (diagnosis)

# Split into training and testing sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ---------------- STEP 2: TRAIN THE MODEL ----------------
# Initialize the Random Forest model
model = RandomForestClassifier(n_estimators=100, random_state=42)

# Train the model
model.fit(X_train, y_train)

# Save the trained model
joblib.dump(model, 'trained_model.pkl')
print("Model trained and saved as 'trained_model.pkl'.")


# ---------------- STEP 3: EVALUATE THE MODEL ----------------
# Make predictions on the test set
y_pred = model.predict(X_test)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print("Model Accuracy:", accuracy)

# Detailed classification report
print("\nClassification Report:\n")
print(classification_report(y_test, y_pred))
