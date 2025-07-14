import streamlit as st
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
import traceback

# Load Model Function
# ----------------------------- 
# This function loads the model and tokenizer from Hugging Face.

@st.cache_resource
def load_model():
    try:
        repo = "google/flan-t5-base"
        tokenizer = AutoTokenizer.from_pretrained(repo)
        model = AutoModelForSeq2SeqLM.from_pretrained(repo)
        pipe = pipeline("text2text-generation", model=model, tokenizer=tokenizer)
        return pipe
    except Exception:
        st.error("❌ Failed to load model from Hugging Face.")
        st.error(f"```\n{traceback.format_exc()}\n```")
        return None

# Thonglish words to easily recognize diseases
thonglish_dict = {
    "flu": "flu",
    "cold": "kulir",
    "pneumonia": "nimoniyaa",
    "malaria": "maleriya",
    "dengue": "dengu",
    "typhoid": "taifoidu",
    "covid-19": "korona",
    "diabetes": "neerizhivu",
    "asthma": "asthuma",
    "tuberculosis": "kaasanoy",
    "migraine": "thalai vali",
    "rheumatoid arthritis": "rheumatoid arthritis",
    "chickenpox": "chikkan paks",
    "measles": "saruma nooy",
    "jaundice": "manjal kaamalai",
    "anemia": "iratha sogai"
}
print("✅ Transformers pipeline is working!")


#  Streamlit UI Setup
# -----------------------------

st.set_page_config(page_title="🩺 Symptom-to-Disease Predictor", layout="centered")
st.title("🩺 Symptom → Disease Predictor (LLM powered)")
st.markdown("Enter the patient's symptoms in English. The AI will predict the most likely disease and translate it into **Thonglish**.")

# User Input
symptoms = st.text_area(
    "📝 Enter symptoms (English):",
    placeholder="Example: fever, sore throat, fatigue...",
    height=150
)

# Load the model
pipe = load_model()

# Inference & Display
# -----------------------------
if st.button("🧬 Predict Disease"):
    if not pipe:
        st.warning("⚠️ Model not available. Please try again later.")
    elif not symptoms.strip():
        st.warning("⚠️ Please enter some symptoms first.")
    else:
        try:
            # Step 1: Generate prompt
            prompt = f"Predict the disease based on these symptoms: {symptoms}"

            # Step 2: Generate response
            result = pipe(prompt, max_new_tokens=50)[0]["generated_text"].strip().lower()

            # Step 3: Filter repetitive text
            words = result.split()
            prediction = words[0] if words.count(words[0]) > 2 else result

            # Step 4: Thonglish translation
            thonglish = thonglish_dict.get(prediction, "maruthuva vilakkam kedaiyadhu")

            # Step 5: Display output
            st.success("🧠 Predicted Disease:")
            st.markdown(f"""
**🔹 English**: `{prediction.title()}`  
**🔸 Thonglish**: `{thonglish}`
""")
        except Exception:
            st.error(" Something went wrong during prediction.")
            st.error(f"```\n{traceback.format_exc()}\n```")
