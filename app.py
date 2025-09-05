import streamlit as st
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
import traceback

# -----------------------------
# Load Model
# -----------------------------
@st.cache_resource
def load_model():
    try:
        repo = "google/flan-t5-base"
        tokenizer = AutoTokenizer.from_pretrained(repo)
        model = AutoModelForSeq2SeqLM.from_pretrained(repo)
        pipe = pipeline("text2text-generation", model=model, tokenizer=tokenizer)
        return pipe
    except Exception:
        st.error("Failed to load model from Hugging Face.")
        st.error(f"```\n{traceback.format_exc()}\n```")
        return None


# -----------------------------
# Disease → Medicines dictionary
# -----------------------------
medicine_dict = {
    "flu": ["Paracetamol 500mg – 1 tablet every 6–8 hours", "Cetirizine 10mg – once daily", "Plenty of fluids"],
    "cold": ["Paracetamol 500mg – 1 tablet every 8 hours", "Antihistamine (Levocetirizine) – once daily", "Steam inhalation – 2 times daily"],
    "pneumonia": ["Amoxicillin 500mg – 1 tablet 3 times daily (doctor prescribed)", "Paracetamol – for fever", "Rest and fluids"],
    "malaria": ["Chloroquine (dose per doctor)", "Paracetamol – every 8 hours", "ORS solution – multiple times"],
    "dengue": ["Paracetamol – every 6 hours (avoid aspirin/ibuprofen)", "ORS / coconut water", "Hospital admission if severe"],
    "typhoid": ["Cefixime 200mg – twice daily (doctor prescribed)", "Paracetamol – for fever", "Soft diet + hydration"],
    "covid-19": ["Paracetamol – every 8 hours", "Vitamin C + Zinc – once daily", "Isolation + hydration"],
    "diabetes": ["Metformin 500mg – twice daily", "Glimepiride 1mg – once daily", "Diet + exercise"],
    "hypertension": ["Amlodipine 5mg – once daily", "Losartan 50mg – once daily", "Low salt diet"],
    "asthma": ["Inhaler (Salbutamol) – as needed", "Montelukast – once daily", "Avoid triggers"],
    "tuberculosis": ["Rifampicin, Isoniazid, Pyrazinamide, Ethambutol – daily (DOTS program)", "Doctor monitoring mandatory"],
    "migraine": ["Ibuprofen 400mg – as needed", "Paracetamol – when headache starts", "Rest in dark quiet room"],
    "anemia": ["Iron tablets – once daily after meals", "Folic acid – once daily", "Green leafy vegetables"],
    "jaundice": ["No specific drug – supportive care", "Plenty of fluids", "Avoid alcohol"],
    "chickenpox": ["Calamine lotion – for itching", "Paracetamol – for fever", "Rest + hydration"],
    "measles": ["Paracetamol – for fever", "Vitamin A supplement – once daily", "Hydration"],
    "rheumatoid arthritis": ["NSAIDs (Ibuprofen 400mg) – twice daily", "Methotrexate (doctor prescribed)", "Physiotherapy"],
    "sinusitis": ["Steam inhalation – 2–3 times daily", "Paracetamol – for pain", "Amoxicillin (if bacterial, doctor prescribed)"],
    "bronchitis": ["Cough syrup – 2 times daily", "Paracetamol – for fever", "Amoxicillin (if bacterial, doctor prescribed)"]
}


# -----------------------------
# Streamlit UI
# -----------------------------
st.set_page_config(page_title="🩺 Symptom-to-Disease Predictor", layout="wide")

st.markdown("<h1 style='text-align: center; color: #2C3E50;'>🩺 Symptom → Disease Predictor</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center;'>Enter symptoms in English. The AI model (FLAN-T5) will predict the most likely disease and suggest possible medicines (Demo Only).</p>", unsafe_allow_html=True)

# Input box
symptoms = st.text_area("📝 Enter symptoms:", placeholder="Example: fever, cough, body pain...", height=150)

# Load model
pipe = load_model()

# -----------------------------
# Prediction Button
# -----------------------------
if st.button("🔍 Predict Disease", use_container_width=True):
    if not pipe:
        st.warning("Model not available. Please try again later.")
    elif not symptoms.strip():
        st.warning("Please enter some symptoms first.")
    else:
        try:
            # AI Prediction
            prompt = f"Predict the disease based on these symptoms: {symptoms}"
            result = pipe(prompt, max_new_tokens=50)[0]["generated_text"].strip().lower()

            # Clean up prediction
            words = result.split()
            prediction = words[0] if words.count(words[0]) > 2 else result

            # Hybrid medicine lookup
            if prediction in medicine_dict:
                medicines = medicine_dict[prediction]
            else:
                med_prompt = f"Suggest 3 common medicines with dosage per day for {prediction}. Keep answer short."
                ai_result = pipe(med_prompt, max_new_tokens=80)[0]["generated_text"].strip()
                medicines = [line.strip("-• ") for line in ai_result.split("\n") if line.strip()]

            # -----------------------------
            # UI Display
            # -----------------------------
            col1, col2 = st.columns(2)

            with col1:
                st.markdown("### 🧠 Predicted Disease")
                st.success(prediction.title())

            with col2:
                st.markdown("### 💊 Suggested Medicines")
                for med in medicines:
                    st.write(f"- {med}")

            # Separator
            st.markdown("---")

        except Exception:
            st.error("Something went wrong during prediction.")
            st.error(f"```\n{traceback.format_exc()}\n```")
