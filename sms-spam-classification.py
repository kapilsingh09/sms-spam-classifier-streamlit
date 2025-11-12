import streamlit as st
import pandas as pd
import numpy as np
import pickle
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
import string

ps = PorterStemmer()

def transform_text(text):
    text = text.lower()
    text = nltk.word_tokenize(text)

    y = []
    for i in text:
        if i.isalnum():  
            y.append(i)
    
    text = y[:]
    y.clear()

    for i in text:
        if i not in stopwords.words('english') and i not in string.punctuation:
            y.append(i)
    
    text = y[:]
    y.clear()

    for i in text:
        y.append(ps.stem(i))

    return " ".join(y)


try:
    tfidf = pickle.load(open('vectorizer1.pkl', 'rb'))
    model = pickle.load(open('modell.pkl', 'rb'))
except Exception as e:
    st.error(f"Error loading model/vectorizer: {e}")
    st.stop()

# ---------- Streamlit UI ----------
st.title("📩 SMS Spam Classifier")

input_sms = st.text_input("Enter your SMS message:")

if st.button("Predict"):
    if not input_sms.strip():
        st.warning("Please enter a message before predicting.")
    else:
        # 1️⃣ Preprocess
        transformed_sms = transform_text(input_sms)

        # 2️⃣ Vectorize
        try:
            vector_input = tfidf.transform([transformed_sms])
        except Exception as e:
            st.error(f"Vectorization error: {e}")
            st.stop()

        # 3️⃣ Predict
        prediction = model.predict(vector_input)[0]

        # 4️⃣ Display
        if prediction == 1:
            st.header("🚨 Spam Message")
        else:
            st.header("✅ Not Spam")
