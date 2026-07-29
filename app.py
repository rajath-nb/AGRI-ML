import os
import pandas as pd
from dotenv import load_dotenv
import streamlit as st
import joblib
import google.generativeai as genai
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sqlalchemy import create_engine, text

# --- CONFIGURATION ---
st.set_page_config(page_title="AGRIshield Scanner", page_icon="🌱", layout="centered")

load_dotenv('api.env')

# Safe helper to prevent StreamlitSecretNotFoundError
def get_secret(key):
    val = os.getenv(key)
    if val:
        return val
    try:
        if key in st.secrets:
            return st.secrets[key]
    except Exception:
        pass
    return None

API_KEY = get_secret("GEMINI_API_KEY")
DATABASE_URL = get_secret("DATABASE_URL")

if not API_KEY:
    st.error("GEMINI_API_KEY not found! Please set it in Streamlit Cloud Secrets.")
    st.stop()

if not DATABASE_URL:
    st.error("DATABASE_URL not found! Please set it in Streamlit Cloud Secrets.")
    st.stop()

# Format URI scheme and append SSL mode
if DATABASE_URL.startswith("postgres://"):
    DATABASE_URL = DATABASE_URL.replace("postgres://", "postgresql://", 1)

if "sslmode=" not in DATABASE_URL:
    separator = "&" if "?" in DATABASE_URL else "?"
    DATABASE_URL += f"{separator}sslmode=require"

engine = create_engine(
    DATABASE_URL,
    connect_args={"sslmode": "require", "connect_timeout": 10},
    pool_pre_ping=True,
    pool_recycle=300
)

genai.configure(api_key=API_KEY)
llm_model = genai.GenerativeModel('gemini-2.5-flash')

# --- INITIALIZE DATABASE TABLE ---
def init_db():
    try:
        with engine.connect() as conn:
            conn.execute(text('''
                CREATE TABLE IF NOT EXISTS feedback_logs (
                    id SERIAL PRIMARY KEY,
                    plant TEXT,
                    disease TEXT,
                    predicted_treatment TEXT,
                    verified_treatment TEXT,
                    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            '''))
            conn.commit()
    except Exception as e:
        st.warning("Connecting to Cloud Database...")

init_db()

# --- AUTOMATIC RETRAINING FUNCTION ---
def trigger_automatic_retraining():
    try:
        df_excel = pd.read_excel('Farmer_Land_Analysis_June2023_Randomized.xlsx')
        df_excel = df_excel[['Plant_Name', 'Detected_Disease_Name', 'Treatment_Suggestion']]
        df_excel.columns = ['plant', 'disease', 'treatment']

        df_db = pd.read_sql_query("SELECT plant, disease, verified_treatment as treatment FROM feedback_logs", engine)

        combined_df = pd.concat([df_excel, df_db], ignore_index=True)
        combined_df = combined_df.dropna(subset=['plant', 'disease', 'treatment'])

        plant_options = combined_df['plant'].unique().tolist()
        disease_options = combined_df['disease'].unique().tolist()

        le_plant = LabelEncoder()
        le_disease = LabelEncoder()

        combined_df['Plant_Num'] = le_plant.fit_transform(combined_df['plant'])
        combined_df['Disease_Num'] = le_disease.fit_transform(combined_df['disease'])

        X = combined_df[['Plant_Num', 'Disease_Num']]
        y = combined_df['treatment']

        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X, y)

        joblib.dump(model, 'agri_robot_model.pkl')
        joblib.dump(le_plant, 'plant_encoder.pkl')
        joblib.dump(le_disease, 'disease_encoder.pkl')
        joblib.dump(plant_options, 'plant_options.pkl')
        joblib.dump(disease_options, 'disease_options.pkl')
        
        return True
    except Exception as e:
        st.error(f"Retraining error: {e}")
        return False

# --- LOAD MODELS ---
try:
    model = joblib.load('agri_robot_model.pkl')
    le_plant = joblib.load('plant_encoder.pkl')
    le_disease = joblib.load('disease_encoder.pkl')
    plant_options = joblib.load('plant_options.pkl')
    disease_options = joblib.load('disease_options.pkl')
except FileNotFoundError:
    st.error("Model files missing! Ensure .pkl files are pushed to GitHub.")
    st.stop()

# --- UI DESIGN ---
st.title("🌿 AGRIshield Diagnostic Report")
st.write("Get instant, verified chemical-free herbal solutions for crop health.")

input_method = st.radio("Input Method:", ["Select from Options", "Type Manually"])

plant_input = ""
disease_input = ""

if input_method == "Select from Options":
    plant_input = st.selectbox("Select Plant Name:", plant_options)
    disease_input = st.selectbox("Select Detected Disease:", disease_options)
elif input_method == "Type Manually":
    plant_input = st.text_input("Enter Plant Name:")
    disease_input = st.text_input("Enter Detected Disease:")

if 'last_prediction' not in st.session_state:
    st.session_state.last_prediction = "No prediction yet"

if st.button("Generate Condensed Herbal Report", use_container_width=True):
    if plant_input and disease_input:
        try:
            plant_num = le_plant.transform([plant_input])[0]
            disease_num = le_disease.transform([disease_input])[0]
            ml_prediction = model.predict([[plant_num, disease_num]])[0]
            
            st.session_state.last_prediction = ml_prediction
            
            prompt = f"""
            Act as an expert agricultural botanist specializing in organic farming. 
            Target Crop: {plant_input}
            Condition/Disease: {disease_input}
            Local Storehouse Suggestion: {ml_prediction}
            
            Provide a concise, highly scannable, condensed diagnostic report in Markdown using ONLY these bullet points:
            
            * **Diagnosis Summary:** 1 sentence explaining what this condition is.
            * **Root Cause (Why it occurs):** 1-2 bullet points explaining the environmental or pathogen trigger.
            * **Herbal Prescription (100% Chemical-Free):** 2 concise bullet points detailing exact organic/herbal remedies (e.g., neem oil ratio, bio-pesticides, garlic extract) and application method.
            * **Prevention Rule:** 1 quick tip to prevent recurrence.
            
            Keep descriptions punchy, direct, and easy for a farmer to read at a glance. No long paragraphs.
            """
            
            with st.spinner("🔍 Compiling Verified Herbal Report..."):
                response = llm_model.generate_content(prompt)
                
            st.markdown("---")
            st.markdown(response.text)
            
        except ValueError:
            st.warning("Custom entry detected. Running Deep Analysis...")
            st.session_state.last_prediction = "Custom Cloud Analysis"
            fallback_prompt = f"Provide a condensed, 100% chemical-free herbal treatment report for '{plant_input}' suffering from '{disease_input}' using short bullet points."
            response = llm_model.generate_content(fallback_prompt)
            st.markdown("---")
            st.markdown(response.text)
    else:
        st.error("Please fill in both fields.")

# --- FEEDBACK LOOP ---
st.markdown("---")
st.markdown("### 📝 Help Improve AGRIshield (Dynamic Cloud Learning)")
feedback_col1, feedback_col2 = st.columns(2)

with feedback_col1:
    is_correct = st.radio("Was this diagnosis accurate?", ["Select...", "Yes", "No"], key="feedback_radio")

if is_correct == "No":
    corrected_treatment = st.text_input("Enter the verified correct herbal treatment:")
    if st.button("Submit Correction & Retrain Model"):
        if corrected_treatment and plant_input and disease_input:
            with engine.connect() as conn:
                conn.execute(text('''
                    INSERT INTO feedback_logs (plant, disease, predicted_treatment, verified_treatment)
                    VALUES (:plant, :disease, :pred, :verified)
                '''), {
                    "plant": plant_input,
                    "disease": disease_input,
                    "pred": st.session_state.last_prediction,
                    "verified": corrected_treatment
                })
                conn.commit()
            
            with st.spinner("🔄 Logging to Supabase & Retraining Model..."):
                success = trigger_automatic_retraining()
                
            if success:
                st.success("✅ Log saved to cloud database and model updated!")
            else:
                st.error("Feedback logged, but automated retraining encountered an issue.")
        else:
            st.error("Please ensure all fields are filled properly.")

elif is_correct == "Yes":
    if st.button("Confirm Accuracy & Retrain"):
        if plant_input and disease_input:
            with engine.connect() as conn:
                conn.execute(text('''
                    INSERT INTO feedback_logs (plant, disease, predicted_treatment, verified_treatment)
                    VALUES (:plant, :disease, :pred, :verified)
                '''), {
                    "plant": plant_input,
                    "disease": disease_input,
                    "pred": st.session_state.last_prediction,
                    "verified": st.session_state.last_prediction
                })
                conn.commit()
            
            with st.spinner("🔄 Confirmation logged to Supabase & Model updated..."):
                success = trigger_automatic_retraining()
                
            if success:
                st.success("✅ Positive feedback stored permanently in the cloud!")
            else:
                st.error("Log saved, but retraining encountered an issue.")
        else:
            st.error("Please ensure all fields are filled properly.")