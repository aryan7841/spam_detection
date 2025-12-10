import streamlit as st
import pickle
import re
import nltk

# 🚀 Page config
st.set_page_config(
    page_title="Spam Shield - Email/SMS Classifier",
    page_icon="🛡️",
    layout="centered"
)

# 📥 NLTK downloads
nltk.download('punkt')
nltk.download('punkt_tab')   # important for newer NLTK versions
nltk.download('stopwords')
nltk.download('wordnet')

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

# 🎨 Custom CSS for nicer UI
st.markdown(
    """
    <style>
        .main {
            background: radial-gradient(circle at top, #1f2933 0, #020617 55%);
            color: #e5e7eb;
        }
        .block-container {
            padding-top: 2rem;
            padding-bottom: 2rem;
            max-width: 800px;
        }
        .title-card {
            background: linear-gradient(135deg, #0f172a, #1e293b);
            padding: 1.5rem 1.75rem;
            border-radius: 1.5rem;
            border: 1px solid rgba(148,163,184,0.4);
            box-shadow: 0 18px 40px rgba(15,23,42,0.8);
        }
        .title-text-main {
            font-size: 2rem;
            font-weight: 700;
            letter-spacing: 0.04em;
        }
        .subtitle-text {
            font-size: 0.95rem;
            color: #9ca3af;
            margin-top: 0.25rem;
        }
        .pill {
            display: inline-flex;
            align-items: center;
            gap: 0.4rem;
            padding: 0.25rem 0.75rem;
            border-radius: 999px;
            font-size: 0.8rem;
            background: rgba(15,23,42,0.9);
            border: 1px solid rgba(148,163,184,0.5);
            color: #e5e7eb;
        }
        .email-card {
            margin-top: 1.5rem;
            background: rgba(15,23,42,0.9);
            padding: 1.5rem;
            border-radius: 1.5rem;
            border: 1px solid rgba(148,163,184,0.4);
            box-shadow: 0 14px 30px rgba(15,23,42,0.7);
        }
        .stTextInput > div > div > input {
            background: rgba(15,23,42,0.9);
            color: #e5e7eb;
            border-radius: 999px;
            border: 1px solid rgba(148,163,184,0.7);
        }
        .result-badge {
            display: inline-flex;
            align-items: center;
            gap: 0.5rem;
            padding: 0.4rem 0.9rem;
            border-radius: 999px;
            font-size: 0.85rem;
        }
        .result-spam {
            background: rgba(248, 113, 113, 0.14);
            border: 1px solid rgba(239, 68, 68, 0.8);
            color: #fecaca;
        }
        .result-ham {
            background: rgba(34,197,94,0.14);
            border: 1px solid rgba(22,163,74,0.8);
            color: #bbf7d0;
        }
        .footer {
            margin-top: 2.5rem;
            font-size: 0.75rem;
            color: #6b7280;
            text-align: center;
        }
    </style>
    """,
    unsafe_allow_html=True
)

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


# 🧠 Load vectorizer & model (same as before)
tfidf = pickle.load(open('vectorizerff.pkl', 'rb'))
model = pickle.load(open('modelff.pkl', 'rb'))

# ========================= UI LAYOUT =============================

# Header card
st.markdown(
    """
    <div class="title-card">
        <div class="pill">
            🛡️ <span>Real-time Email & SMS Protection</span>
        </div>
        <div style="margin-top: 0.6rem;">
            <span class="title-text-main">Spam Shield Classifier</span>
        </div>
        <div class="subtitle-text">
            Paste any email or SMS and let the model detect whether it's <b>spam</b> or <b>safe</b>.
        </div>
    </div>
    """,
    unsafe_allow_html=True
)

st.markdown("")  # small gap

# Main card
with st.container():
    st.markdown('<div class="email-card">', unsafe_allow_html=True)

    col1, col2 = st.columns([3, 2])

    with col1:
        inp_email = st.text_area(
            "✉️ Enter the message",
            placeholder="Example: Congratulations! You’ve won a $500 gift card. Click here to claim your prize...",
            height=150
        )

        predict_button = st.button("🔍 Analyze message", use_container_width=True)

    with col2:
        st.markdown("### ℹ️ Tips")
        st.write(
            "- Check suspicious offers\n"
            "- Verify unknown links\n"
            "- Be careful with OTP / bank messages"
        )
        st.markdown("---")
        st.markdown("**Model pipeline:**\n- Cleaning & lemmatization\n- TF-IDF vectorization\n- Trained ML classifier")

    st.markdown("</div>", unsafe_allow_html=True)

# Prediction section
if predict_button:
    if not inp_email.strip():
        st.warning("Please enter a message to analyze.")
    else:
        # preprocess
        transform_email = clean_text(inp_email)

        # vectorize
        vector_i = tfidf.transform([transform_email])

        # predict
        result = model.predict(vector_i)

        st.markdown("### 🔮 Prediction")
        if result == 1:
            # SPAM
            st.markdown(
                """
                <div class="result-badge result-spam">
                    <span>🚫 Classified as <b>SPAM</b></span>
                </div>
                """,
                unsafe_allow_html=True
            )
            st.error("Be careful! This message looks suspicious.")
        else:
            # NOT SPAM
            st.markdown(
                """
                <div class="result-badge result-ham">
                    <span>✅ Classified as <b>NOT SPAM</b></span>
                </div>
                """,
                unsafe_allow_html=True
            )
            st.success("This message appears safe.")

        with st.expander("🔍 View cleaned text (what the model sees)"):
            st.code(transform_email, language="text")

# Footer
st.markdown(
    """
    <div class="footer">
        Built with ❤️ using Streamlit, NLTK, and scikit-learn.
    </div>
    """,
    unsafe_allow_html=True
)
