import streamlit as st
import pickle

import re
import nltk

nltk.download('punkt')
nltk.download('punkt_tab')   # 🔴 ADD THIS LINE
nltk.download('stopwords')
nltk.download('wordnet')

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def clean_text(text):
    # 1. Lowercase
    text = text.lower()

    # 2. Remove URLs
    text = re.sub(r'http\S+|www\S+', ' ', text)

    # 3. Remove emails
    text = re.sub(r'\S+@\S+\.\S+', ' ', text)

    # 4. Remove numbers & special chars
    text = re.sub(r'[^a-z\s]', ' ', text)

    # 5. Tokenization
    tokens = nltk.word_tokenize(text)

    # 6. Remove stopwords & punctuation
    tokens = [word for word in tokens if word not in stop_words and word.isalpha()]

    # 7. Lemmatization
    tokens = [lemmatizer.lemmatize(word) for word in tokens]

    # 8. Join tokens back
    return " ".join(tokens)


tfidf = pickle.load(open('vectorizerff.pkl','rb'))
model = pickle.load(open('modelff.pkl','rb'))

st.title("email/sms spam classifier")

inp_email = st.text_input("enter the message")
if st.button("predict"):
    # preprocess
    transform_email = clean_text(inp_email)

    # vectorize
    vector_i = tfidf.transform([transform_email])

    # predict
    result = model.predict(vector_i)

    # display
    if result == 1:
        st.header("spam")
    else:
        st.header("not spam")
