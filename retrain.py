import pandas as pd
import sqlite3
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import joblib

def retrain_model():
    print("Loading original dataset...")
    df_excel = pd.read_excel('Farmer_Land_Analysis_June2023_Randomized.xlsx')
    df_excel = df_excel[['Plant_Name', 'Detected_Disease_Name', 'Treatment_Suggestion']]
    df_excel.columns = ['plant', 'disease', 'treatment']

    print("Loading user feedback from database...")
    try:
        conn = sqlite3.connect('agri_feedback.db')
        df_db = pd.read_sql_query("SELECT plant, disease, verified_treatment as treatment FROM feedback_logs", conn)
        conn.close()
    except Exception as e:
        print("No feedback database found yet. Using Excel data only.")
        df_db = pd.DataFrame(columns=['plant', 'disease', 'treatment'])

    # Combine original data with new user feedback
    combined_df = pd.concat([df_excel, df_db], ignore_index=True)
    combined_df = combined_df.dropna(subset=['plant', 'disease', 'treatment'])

    print(f"Total training rows (Original + New Feedback): {len(combined_df)}")

    # Update dropdown lists
    plant_options = combined_df['plant'].unique().tolist()
    disease_options = combined_df['disease'].unique().tolist()

    # Re-encode and train
    le_plant = LabelEncoder()
    le_disease = LabelEncoder()

    combined_df['Plant_Num'] = le_plant.fit_transform(combined_df['plant'])
    combined_df['Disease_Num'] = le_disease.fit_transform(combined_df['disease'])

    X = combined_df[['Plant_Num', 'Disease_Num']]
    y = combined_df['treatment']

    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X, y)

    # Overwrite the old pickle files with the newly trained, smarter model
    joblib.dump(model, 'agri_robot_model.pkl')
    joblib.dump(le_plant, 'plant_encoder.pkl')
    joblib.dump(le_disease, 'disease_encoder.pkl')
    joblib.dump(plant_options, 'plant_options.pkl')
    joblib.dump(disease_options, 'disease_options.pkl')

    print("Success! Model successfully retrained with user feedback.")

if __name__ == '__main__':
    retrain_model()