import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import joblib

print("Loading dataset...")
# Load the dataset
df = pd.read_excel('Farmer_Land_Analysis_June2023_Randomized.xlsx')

# Clean the data by dropping rows with missing essential values
df = df.dropna(subset=['Plant_Name', 'Detected_Disease_Name', 'Treatment_Suggestion'])

# Extract unique lists for the frontend dropdowns
plant_options = df['Plant_Name'].unique().tolist()
disease_options = df['Detected_Disease_Name'].unique().tolist()

print("Encoding data...")
# Convert text data into numerical labels
le_plant = LabelEncoder()
le_disease = LabelEncoder()

df['Plant_Num'] = le_plant.fit_transform(df['Plant_Name'])
df['Disease_Num'] = le_disease.fit_transform(df['Detected_Disease_Name'])

# Define features (X) and target (y)
X = df[['Plant_Num', 'Disease_Num']]
y = df['Treatment_Suggestion']

print("Training model...")
# Train the Random Forest Classifier
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X, y)

print("Saving files...")
# Save the trained model, encoders, and dropdown lists
joblib.dump(model, 'agri_robot_model.pkl')
joblib.dump(le_plant, 'plant_encoder.pkl')
joblib.dump(le_disease, 'disease_encoder.pkl')
joblib.dump(plant_options, 'plant_options.pkl')
joblib.dump(disease_options, 'disease_options.pkl')

print("Success! Model trained and all files saved.")