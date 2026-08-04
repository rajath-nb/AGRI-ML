import os
import PIL.Image
import pandas as pd
from dotenv import load_dotenv
import streamlit as st
import joblib
import google.generativeai as genai
from sqlalchemy import create_engine, text

# --- CONFIGURATION & PAGE SETUP ---
st.set_page_config(page_title="AGRIshield ML", page_icon="🌿", layout="wide")

st.markdown("""
    <style>
        .block-container { padding-top: 2rem; padding-bottom: 2rem; }
        h1 { color: #2E7D32; font-weight: 700; }
    </style>
""", unsafe_allow_html=True)

load_dotenv('api.env')

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

if not API_KEY or not DATABASE_URL:
    st.error("⚠️ GEMINI_API_KEY or DATABASE_URL not found! Set them in secrets.")
    st.stop()

if DATABASE_URL.startswith("postgres://"):
    DATABASE_URL = DATABASE_URL.replace("postgres://", "postgresql://", 1)

if "sslmode=" not in DATABASE_URL:
    separator = "&" if "?" in DATABASE_URL else "?"
    DATABASE_URL += f"{separator}sslmode=require"

engine = create_engine(
    DATABASE_URL,
    connect_args={"sslmode": "require", "connect_timeout": 15},
    pool_pre_ping=True,
    pool_recycle=300
)

genai.configure(api_key=API_KEY)
llm_model = genai.GenerativeModel('gemini-2.5-flash')

# --- DYNAMIC OPTION TRANSLATION DICTIONARY ---
OPTION_TRANSLATIONS = {
    "Hindi (हिंदी)": {
        "Tomato": "टमाटर (Tomato)", "Grapes": "अंगूर (Grapes)", "Wheat": "गेहूं (Wheat)", "Onion": "प्याज (Onion)",
        "Potato": "आलू (Potato)", "Chilli": "मिर्च (Chilli)", "Corn": "मक्का (Corn)",
        "Late Blight": "पछेती झुलसा (Late Blight)", "Healthy": "स्वास्थ्य (Healthy)", "Downy Mildew": "मृदु रोमिल आसिता (Downy Mildew)",
        "Purple Blotch": "बैंगनी धब्बा (Purple Blotch)", "Early Blight": "अगेती झुलसा (Early Blight)", "Fruit Rot": "फल सड़न (Fruit Rot)",
        "Yellow Rust": "पीला रतुआ (Yellow Rust)", "Gray Leaf Spot": "ग्रे लीफ स्पॉट (Gray Leaf Spot)", "Leaf Curl": "पत्ती मरोड़ (Leaf Curl)",
        "Black Rot": "काला सड़न (Black Rot)", "Leaf Mold": "पत्ती मोल्ड (Leaf Mold)", "Brown Rust": "भूरा रतुआ (Brown Rust)",
        "Blight": "झुलसा (Blight)", "Common Rust": "सामान्य रतुआ (Common Rust)"
    },
    "Tamil (தமிழ்)": {
        "Tomato": "தக்காளி (Tomato)", "Grapes": "திராட்சை (Grapes)", "Wheat": "கோதுமை (Wheat)", "Onion": "வெங்காயம் (Onion)",
        "Potato": "உருளைக்கிழங்கு (Potato)", "Chilli": "மிளகாய் (Chilli)", "Corn": "சோளம் (Corn)",
        "Late Blight": "பிந்தைய கருகல் நோய் (Late Blight)", "Healthy": "ஆரோக்கியமானது (Healthy)", "Downy Mildew": "அடிச்சாம்பல் நோய் (Downy Mildew)",
        "Purple Blotch": "ஊதா நிற புள்ளி நோய் (Purple Blotch)", "Early Blight": "முந்தைய கருகல் நோய் (Early Blight)", "Fruit Rot": "பழ அழுகல் (Fruit Rot)",
        "Yellow Rust": "மஞ்சள் துரு நோய் (Yellow Rust)", "Gray Leaf Spot": "சாம்பல் இலை புள்ளி (Gray Leaf Spot)", "Leaf Curl": "இலை சுருட்டல் நோய் (Leaf Curl)",
        "Black Rot": "கருப்பு அழுகல் (Black Rot)", "Leaf Mold": "இலை பூஞ்சணம் (Leaf Mold)", "Brown Rust": "பழுப்பு துரு நோய் (Brown Rust)",
        "Blight": "கருகல் நோய் (Blight)", "Common Rust": "பொதுவான துரு நோய் (Common Rust)"
    },
    "Kannada (ಕನ್ನಡ)": {
        "Tomato": "ಟೊಮೆಟೊ (Tomato)", "Grapes": "ದ್ರಾಕ್ಷಿ (Grapes)", "Wheat": "ಗೋಧಿ (Wheat)", "Onion": "ಈರುಳ್ಳಿ (Onion)",
        "Potato": "ಆಲೂಗಡ್ಡೆ (Potato)", "Chilli": "ಮೆಣಸಿನಕಾಯಿ (Chilli)", "Corn": "ಮೆಕ್ಕೆಜೋಳ (Corn)",
        "Late Blight": "ತಡವಾದ ಅಂಗಮಾರಿ ರೋಗ (Late Blight)", "Healthy": "ಆರೋಗ್ಯಕರ (Healthy)", "Downy Mildew": "ಬೂದಿ ರೋಗ (Downy Mildew)",
        "Purple Blotch": "ನೇರಳೆ ಮಚ್ಚೆ ರೋಗ (Purple Blotch)", "Early Blight": "ಆರಂಭಿಕ ಅಂಗಮಾರಿ ರೋಗ (Early Blight)", "Fruit Rot": "ಹಣ್ಣು ಕೊಳೆತ ರೋಗ (Fruit Rot)",
        "Yellow Rust": "ಹಳದಿ ತುಕ್ಕು ರೋಗ (Yellow Rust)", "Gray Leaf Spot": "ಬೂದು ಇಲೆ ಮಚ್ಚೆ (Gray Leaf Spot)", "Leaf Curl": "ಎಲೆ ಸುರುಳಿ ರೋಗ (Leaf Curl)",
        "Black Rot": "ಕಪ್ಪು ಕೊಳೆತ (Black Rot)", "Leaf Mold": "ಎಲೆ ಸಿಲಿಂಡರ್ (Leaf Mold)", "Brown Rust": "ಕಂದು ತುಕ್ಕು ರೋಗ (Brown Rust)",
        "Blight": "ಅಂಗಮಾರಿ ರೋಗ (Blight)", "Common Rust": "ಸಾಮಾನ್ಯ ತುಕ್ಕು ರೋಗ (Common Rust)"
    },
    "Telugu (తెలుగు)": {
        "Tomato": "టమోటా (Tomato)", "Grapes": "ద్రాక్ష (Grapes)", "Wheat": "గోధుమ (Wheat)", "Onion": "ఉల్లిపాయ (Onion)",
        "Potato": "బంగాళాదుంప (Potato)", "Chilli": "మిరపకాయ (Chilli)", "Corn": "మొక్కజొన్న (Corn)",
        "Late Blight": "లేట్ బ్లైట్ (Late Blight)", "Healthy": "ఆరోగ్యకరమైనది (Healthy)", "Downy Mildew": "డౌనీ మైల్డ్యూ (Downy Mildew)",
        "Purple Blotch": "పర్పుల్ బ్లాచ్ (Purple Blotch)", "Early Blight": "అర్లీ బ్లైట్ (Early Blight)", "Fruit Rot": "కాయ కుళ్లు తెగులు (Fruit Rot)",
        "Yellow Rust": "పసుపు కుంకుమ తెగులు (Yellow Rust)", "Gray Leaf Spot": "గ్రే లీఫ్ స్పాట్ (Gray Leaf Spot)", "Leaf Curl": "ఆకు ముడత తెగులు (Leaf Curl)",
        "Black Rot": "నల్ల కుళ్లు తెగులు (Black Rot)", "Leaf Mold": "ఆకు బూజు తెగులు (Leaf Mold)", "Brown Rust": "గోధుమ రంగు కుంకుమ తెగులు (Brown Rust)",
        "Blight": "ఆకు మచ్చ తెగులు (Blight)", "Common Rust": "సాధారణ కుంకుమ తెగులు (Common Rust)"
    },
    "Marathi (मराठी)": {
        "Tomato": "टोमॅटो (Tomato)", "Grapes": "द्राक्षे (Grapes)", "Wheat": "गहू (Wheat)", "Onion": "कांदा (Onion)",
        "Potato": "बटाटा (Potato)", "Chilli": "मिरची (Chilli)", "Corn": "मका (Corn)",
        "Late Blight": "उशिरा येणारा करपा (Late Blight)", "Healthy": "निरोगी (Healthy)", "Downy Mildew": "केवडा रोग (Downy Mildew)",
        "Purple Blotch": "जांभळा टिपका (Purple Blotch)", "Early Blight": "लवकर येणारा करपा (Early Blight)", "Fruit Rot": "फळ सड (Fruit Rot)",
        "Yellow Rust": "पिवळा तांबेरा (Yellow Rust)", "Gray Leaf Spot": "राखी ठिपके (Gray Leaf Spot)", "Leaf Curl": "पर्णगुच्छ (Leaf Curl)",
        "Black Rot": "काळी कुज (Black Rot)", "Leaf Mold": "पानावरील बुरशी (Leaf Mold)", "Brown Rust": "तांबेरा (Brown Rust)",
        "Blight": "करपा (Blight)", "Common Rust": "सामान्य तांबेरा (Common Rust)"
    },
    "Bengali (বাংলা)": {
        "Tomato": "টমেটো (Tomato)", "Grapes": "আঙ্গুর (Grapes)", "Wheat": "গম (Wheat)", "Onion": "পেঁয়াজ (Onion)",
        "Potato": "আলু (Potato)", "Chilli": "মরিচ (Chilli)", "Corn": "ভুট্টা (Corn)",
        "Late Blight": "লেট ব্ল্যার্ট (Late Blight)", "Healthy": "স্বাস্থ্যে ভালো (Healthy)", "Downy Mildew": "ডাউনি মিলডিউ (Downy Mildew)",
        "Purple Blotch": "পার্পল ব্লচ (Purple Blotch)", "Early Blight": "আর্লি ব্লাইট (Early Blight)", "Fruit Rot": "ফল পচন (Fruit Rot)",
        "Yellow Rust": "হলুদ মরিচা রোগ (Yellow Rust)", "Gray Leaf Spot": "ধূসর পাতা দাগ (Gray Leaf Spot)", "Leaf Curl": "পাতা কোঁকড়ানো (Leaf Curl)",
        "Black Rot": "কালো পচন (Black Rot)", "Leaf Mold": "পাতার ছত্রাক (Leaf Mold)", "Brown Rust": "বাদামী মরিচা রোগ (Brown Rust)",
        "Blight": "ধসা রোগ (Blight)", "Common Rust": "সাধারণ মরিচা রোগ (Common Rust)"
    },
    "Spanish (Español)": {
        "Tomato": "Tomate (Tomato)", "Grapes": "Uvas (Grapes)", "Wheat": "Trigo (Wheat)", "Onion": "Cebolla (Onion)",
        "Potato": "Patata (Potato)", "Chilli": "Chile (Chilli)", "Corn": "Maíz (Corn)",
        "Late Blight": "Tizón tardío (Late Blight)", "Healthy": "Saludable (Healthy)", "Downy Mildew": "Mildeo velloso (Downy Mildew)",
        "Purple Blotch": "Mancha púrpura (Purple Blotch)", "Early Blight": "Tizón temprano (Early Blight)", "Fruit Rot": "Podredumbre del fruto (Fruit Rot)",
        "Yellow Rust": "Roya amarilla (Yellow Rust)", "Gray Leaf Spot": "Mancha foliar gris (Gray Leaf Spot)", "Leaf Curl": "Enrollamiento de la hoja (Leaf Curl)",
        "Black Rot": "Podredumbre negra (Black Rot)", "Leaf Mold": "Moho de la hoja (Leaf Mold)", "Brown Rust": "Roya parda (Brown Rust)",
        "Blight": "Tizón (Blight)", "Common Rust": "Roya común (Common Rust)"
    }
}

# --- INITIALIZE DATABASE ---
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
                    generated_report TEXT,
                    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );
            '''))
            conn.execute(text('''
                CREATE TABLE IF NOT EXISTS plant_image_logs (
                    id SERIAL PRIMARY KEY,
                    image_name TEXT,
                    plant_species TEXT,
                    detected_deformity TEXT,
                    generated_report TEXT,
                    is_verified BOOLEAN DEFAULT FALSE,
                    verified_label TEXT,
                    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );
            '''))
            conn.commit()
            try:
                conn.execute(text("ALTER TABLE plant_image_logs ADD COLUMN generated_report TEXT;"))
                conn.commit()
            except Exception:
                pass
    except Exception as e:
        st.sidebar.warning(f"Database connection notice: {e}")

init_db()

# --- SESSION STATE INITIALIZATION ---
if 'report_text' not in st.session_state: st.session_state.report_text = None
if 'active_mode' not in st.session_state: st.session_state.active_mode = "Text"
if 'feedback_submitted' not in st.session_state: st.session_state.feedback_submitted = False
if 'active_image' not in st.session_state: st.session_state.active_image = None
if 'active_image_name' not in st.session_state: st.session_state.active_image_name = "uploaded_sample.jpg"

# --- LOAD LOCAL ML MODELS ---
try:
    model = joblib.load('agri_robot_model.pkl')
    le_plant = joblib.load('plant_encoder.pkl')
    le_disease = joblib.load('disease_encoder.pkl')
    plant_options = joblib.load('plant_options.pkl')
    disease_options = joblib.load('disease_options.pkl')
except FileNotFoundError:
    plant_options, disease_options = [], []

# =========================================================
# SIDEBAR UI
# =========================================================
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/1892/1892747.png", width=80)
    st.title("⚙️ Settings")
    
    report_language = st.selectbox(
        "🌐 Report Language",
        ["English", "Hindi (हिंदी)", "Kannada (ಕನ್ನಡ)", "Tamil (தமிழ்)", "Telugu (తెలుగు)", "Marathi (मराठी)", "Bengali (বাংলা)", "Spanish (Español)"]
    )
    
    st.markdown("---")
    st.markdown("### 🌿 About AGRIshield\nAn AI-powered agricultural diagnostic tool providing verified, 100% chemical-free herbal solutions and plant deformity reports.")
    st.markdown("---")
    

# Function to dynamically format option labels in selected language
def format_translated_label(option_key):
    if report_language in OPTION_TRANSLATIONS and option_key in OPTION_TRANSLATIONS[report_language]:
        return OPTION_TRANSLATIONS[report_language][option_key]
    return option_key

# =========================================================
# MAIN HEADER
# =========================================================
st.markdown("<h1 style='text-align: center;'>🛡️ AGRIshield ML Based Web Application</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; font-size: 1.1rem; color: #555;'>Instantly scan and detect crop diseases to receive verified organic treatments.</p>", unsafe_allow_html=True)
st.markdown("---")

# =========================================================
# MODERN TABS LAYOUT
# =========================================================
tab_vision, tab_text = st.tabs(["📷 Vision Scanner (AI)", "📝 Manual Diagnostics (ML)"])

# --- TAB 1: VISION SCANNER ---
with tab_vision:
    st.markdown("### 📸 Image-Based Crop Analysis")
    col_input, col_preview = st.columns([1, 1], gap="large")
    
    with col_input:
        image_source = st.radio("Choose Input Method:", ["📁 Upload File", "📸 Take Photo"], horizontal=True)
        active_file = None
        if image_source == "📁 Upload File":
            active_file = st.file_uploader("Upload Crop Photo (Leaf, Stem, Fruit, or Whole Plant):", type=["jpg", "jpeg", "png"], label_visibility="collapsed")
        else:
            active_file = st.camera_input("Take a photo:")

        if active_file is not None:
            try:
                st.session_state.active_image = PIL.Image.open(active_file)
                st.session_state.active_image_name = getattr(active_file, 'name', 'camera_capture.jpg')
            except Exception as img_err:
                st.error(f"Error loading image: {img_err}")

    with col_preview:
        if st.session_state.active_image is not None:
            st.image(st.session_state.active_image, caption="Current Sample Ready for Analysis", use_container_width=True)
            
            if st.button("🔬 Analyze Image & Generate Report", type="primary", use_container_width=True):
                st.session_state.feedback_submitted = False
                st.session_state.active_mode = "Image"
                
                with st.spinner(f"Scanning visuals and translating to {report_language}..."):
                    prompt = f"""
                    Act as an expert plant pathologist and organic botanist. Carefully analyze the uploaded crop/plant sample.
                    IMPORTANT: Generate the ENTIRE diagnostic report strictly in the language: {report_language}.
                    
                    Provide a condensed, scannable report using Markdown with these exact bullet points:
                    * **Identified Crop:** <Species Name>
                    * **Detected Deformity / Disease:** <Condition Name>
                    * **Diagnosis Summary:** 1 sentence explaining the symptoms.
                    * **Herbal Prescription (100% Organic):** 2 precise bullet points detailing exact organic remedies.
                    * **Prevention Rule:** 1 quick soil or farming tip.
                    """
                    try:
                        response = llm_model.generate_content([prompt, st.session_state.active_image])
                        st.session_state.report_text = response.text
                    except Exception as e:
                        st.error(f"Image analysis error: {e}")
        else:
            st.info("👈 Upload or capture a photo to begin the AI scan.")

# --- TAB 2: MANUAL DIAGNOSTICS ---
with tab_text:
    st.markdown("### 📋 Text-Based Crop Analysis")
    input_method = st.radio("Input Preference:", ["Select from Options", "Type Manually"], horizontal=True)
    
    col_plant, col_disease = st.columns(2)
    plant_input, disease_input = "", ""

    with col_plant:
        if input_method == "Select from Options" and plant_options:
            plant_input = st.selectbox(
                "Select Plant Name:", 
                plant_options, 
                format_func=format_translated_label
            )
        else:
            plant_input = st.text_input("Enter Plant Name:")
            
    with col_disease:
        if input_method == "Select from Options" and disease_options:
            disease_input = st.selectbox(
                "Select Detected Disease:", 
                disease_options, 
                format_func=format_translated_label
            )
        else:
            disease_input = st.text_input("Enter Detected Disease:")

    st.markdown("<br>", unsafe_allow_html=True)
    if st.button("Generate Herbal Report", type="primary", use_container_width=True):
        if plant_input and disease_input:
            st.session_state.feedback_submitted = False
            st.session_state.active_mode = "Text"
            st.session_state.active_plant = plant_input
            st.session_state.active_disease = disease_input
            
            with st.spinner(f"Compiling Herbal Report in {report_language}..."):
                prompt = f"""
                Act as an expert organic agricultural botanist.
                Target Crop: {plant_input}
                Condition/Disease: {disease_input}
                
                IMPORTANT: Generate the ENTIRE diagnostic report strictly in the language: {report_language}.
                
                Provide a condensed report using Markdown:
                * **Diagnosis Summary:** 1 sentence.
                * **Herbal Prescription (100% Organic):** 2 concise bullet points.
                * **Prevention Rule:** 1 quick tip.
                """
                response = llm_model.generate_content(prompt)
                st.session_state.report_text = response.text
        else:
            st.warning("Please specify both the Plant and the Disease to proceed.")

# =========================================================
# REPORT DISPLAY & FEEDBACK SYSTEM
# =========================================================
if st.session_state.report_text:
    st.markdown("<br>", unsafe_allow_html=True)
    
    report_card = st.container(border=True)
    with report_card:
        st.markdown(f"### 📑 Official Diagnostic Report")
        st.markdown(st.session_state.report_text)
        
        st.divider()
        
        if not st.session_state.feedback_submitted:
            st.markdown("#### 🔄 Help Improve Our AI")
            st.write("Was this diagnosis and treatment accurate?")
            col_yes, col_no, col_empty = st.columns([1, 1, 4])
            
            with col_yes:
                if st.button("✅ Yes, it's correct", use_container_width=True):
                    try:
                        with engine.connect() as conn:
                            if st.session_state.active_mode == "Image":
                                conn.execute(text('''
                                    INSERT INTO plant_image_logs (image_name, generated_report, is_verified, verified_label)
                                    VALUES (:img_name, :report, TRUE, 'Confirmed Correct')
                                '''), {"img_name": st.session_state.active_image_name, "report": st.session_state.report_text})
                            else:
                                conn.execute(text('''
                                    INSERT INTO feedback_logs (plant, disease, generated_report)
                                    VALUES (:plant, :disease, :report)
                                '''), {"plant": st.session_state.get('active_plant', 'Unknown'), "disease": st.session_state.get('active_disease', 'Unknown'), "report": st.session_state.report_text})
                            conn.commit()
                        st.session_state.feedback_submitted = True
                        st.rerun()
                    except Exception as db_err:
                        st.error(f"Database error: {db_err}")

            with col_no:
                with st.popover("❌ No, I have a correction"):
                    correction = st.text_input("Enter the correct diagnosis or herbal treatment:")
                    if st.button("Submit Correction", type="primary"):
                        if correction:
                            try:
                                with engine.connect() as conn:
                                    if st.session_state.active_mode == "Image":
                                        conn.execute(text('''
                                            INSERT INTO plant_image_logs (image_name, generated_report, is_verified, verified_label)
                                            VALUES (:img_name, :report, FALSE, :verified)
                                        '''), {"img_name": st.session_state.active_image_name, "report": st.session_state.report_text, "verified": correction})
                                    else:
                                        conn.execute(text('''
                                            INSERT INTO feedback_logs (plant, disease, verified_treatment, generated_report)
                                            VALUES (:plant, :disease, :verified, :report)
                                        '''), {"plant": st.session_state.get('active_plant', 'Unknown'), "disease": st.session_state.get('active_disease', 'Unknown'), "verified": correction, "report": st.session_state.report_text})
                                    conn.commit()
                                st.session_state.feedback_submitted = True
                                st.rerun()
                            except Exception as db_err:
                                st.error(f"Database error: {db_err}")
                        else:
                            st.warning("Please type a correction before submitting.")
        else:
            st.success("✅ Thank you! Your feedback has been securely logged to Supabase to train future models.")
