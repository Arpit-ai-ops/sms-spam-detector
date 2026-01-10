import streamlit as st
import pickle
import string
import nltk
import time
import json
import requests

from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from streamlit_lottie import st_lottie

# ------------------ NLTK SETUP ------------------
nltk.download('punkt')
nltk.download('punkt_tab') 
nltk.download('stopwords')

ps = PorterStemmer()

# ------------------ LOTTIE LOADER (SAFE) ------------------
def load_lottieurl(url):
    try:
        r = requests.get(url, timeout=10)
        if r.status_code != 200:
            return None
        return r.json()
    except Exception:
        return None

# Load animations
lottie_spam = load_lottieurl(
    "https://assets5.lottiefiles.com/packages/lf20_6p8ov8.json"
)
lottie_sidebar = load_lottieurl(
    "https://assets10.lottiefiles.com/packages/lf20_qp1q7mct.json"
)

# ------------------ TEXT PREPROCESSING ------------------
def transform_text(text):
    text = text.lower()
    text = nltk.word_tokenize(text)

    y = []
    for i in text:
        if i.isalnum() and i not in stopwords.words('english'):
            y.append(ps.stem(i))

    return " ".join(y)

# ------------------ PAGE CONFIG ------------------
st.set_page_config(
    page_title="Spam Detector Pro",
    page_icon="🛡️",
    layout="centered"
)

# ------------------ CUSTOM CSS ------------------
st.markdown("""
<style>
.main { background-color: #f0f2f6; }
.stButton>button {
    width: 100%;
    border-radius: 20px;
    height: 3em;
    background-color: #007bff;
    color: white;
    font-weight: bold;
}
.stTextArea>div>div>textarea {
    border-radius: 10px;
}
</style>
""", unsafe_allow_html=True)

# ------------------ SIDEBAR ------------------
with st.sidebar:
    if lottie_sidebar:
        st_lottie(lottie_sidebar, height=150, key="shield")
    else:
        st.warning("⚠️ Sidebar animation failed to load")

    st.title("About Detector")
    st.info("""
        
        **SMS Spam Detector** is an advanced NLP-driven utility that leverages machine learning to identify and filter suspicious communications. 

        Our primary objective is to fortify digital messaging ecosystems by providing real-time, high-accuracy threat detection.
        
    """)

    st.markdown("---")
    st.write("**Model:** Multinomial Naive Bayes")
    st.write("**Vectorizer:** TF-IDF")
    st.markdown("---")
    st.success("Developer: Arpit Thakur")

# ------------------ MAIN UI ------------------
st.title("SMS Spam Detector")
st.write("Check your text below to stay safe from spam messages.")

input_sms = st.text_area(
    "Paste your message here:",
    placeholder="Example: Congratulations! You've won a $1000 gift card. Click here...",
    height=150
)

# ------------------ LOAD MODEL ------------------
try:
    tfidf = pickle.load(open('vectorizer.pkl', 'rb'))
    model = pickle.load(open('model.pkl', 'rb'))
    model_loaded = True
except FileNotFoundError:
    st.error("❌ 'vectorizer.pkl' or 'model.pkl' not found!")
    model_loaded = False

# ------------------ PREDICTION ------------------
if st.button("Analyze Message") and model_loaded:
    if input_sms.strip() == "":
        st.warning("⚠️ Please enter some text first!")
    else:
        # Preprocess
        transformed_sms = transform_text(input_sms)
        vector_input = tfidf.transform([transformed_sms])
        result = model.predict(vector_input)[0]

        # Animation placeholder
        animation_placeholder = st.empty()

        if lottie_spam:
            with animation_placeholder:
                st_lottie(lottie_spam, height=200, key="scan")
                time.sleep(1.5)
        else:
            animation_placeholder.info("🔍 Analyzing message...")

        animation_placeholder.empty()

        # Result
        st.divider()
        st.subheader("Result:")

        if result == 1:
            st.error("🚨 **SPAM DETECTED!**")
            st.markdown(
                "This looks like a suspicious message. "
                "Avoid clicking links or sharing personal info."
            )
        else:
            st.success("✅ **NOT SPAM (SAFE)**")
            st.markdown("This message seems safe to read.")
            st.toast("Verification Successful!", icon="🛡️")

# ------------------ FOOTER ------------------
st.markdown("---")
st.caption("Note: Accuracy depends on training data quality.")

