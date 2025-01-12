import nltk
nltk.download('punkt')
nltk.download('stopwords')

import streamlit as st
import pickle
import string
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

ps = PorterStemmer()

# Function to preprocess and transform text
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

# Load vectorizer and model
tk = pickle.load(open("vectorizer.pkl", 'rb'))
model = pickle.load(open("model.pkl", 'rb'))

# Background and Heading Styles
st.markdown(
    """
    <style>
        body {
            background: linear-gradient(to right, #ff7e5f, #feb47b); /* Vibrant gradient background */
            font-family: 'Arial', sans-serif;
        }
        .main-title {
            background: linear-gradient(to right, #36d1dc, #5b86e5); /* Vibrant gradient for heading */
            padding: 20px;
            border-radius: 12px;
            color: white;
            text-align: center;
            box-shadow: 0px 4px 10px rgba(0, 0, 0, 0.25);
        }
        .sub-title {
            font-size: 16px;
            font-style: italic;
            color: white;
        }
        .result-spam {
            background: linear-gradient(to right, #ff512f, #dd2476); /* Gradient for Spam */
            padding: 15px;
            border-radius: 8px;
            margin-top: 20px;
            color: white;
            text-align: center;
            font-size: 24px;
            font-weight: bold;
        }
        .result-not-spam {
            background: linear-gradient(to right, #56ab2f, #a8e063); /* Gradient for Not Spam */
            padding: 15px;
            border-radius: 8px;
            margin-top: 20px;
            color: white;
            text-align: center;
            font-size: 24px;
            font-weight: bold;
        }
    </style>
    <div class="main-title">
        <h1>SMS Spam Detection System</h1>
        <p class="sub-title">Made by Vinayak Mehta</p>
    </div>
    """,
    unsafe_allow_html=True,
)

# Input Box
st.markdown(
    """
    <div style="padding: 10px; background: rgba(255, 255, 255, 0.8); border-radius: 8px; margin-bottom: 20px;">
        <h3 style="text-align: center; color: #333;">Enter the SMS you want to check:</h3>
    </div>
    """,
    unsafe_allow_html=True,
)
input_sms = st.text_area("Type your message here", height=100)

if st.button("Predict"):
    if not input_sms.strip():
        st.warning("Please enter a valid SMS to predict!")
    else:
        # Preprocess the SMS
        transformed_sms = transform_text(input_sms)

        # Vectorize the text
        vector_input = tk.transform([transformed_sms])

        # Predict whether it's Spam or Not
        result = model.predict(vector_input)[0]

        # Display Results with Gradient Backgrounds
        if result == 1:
            st.markdown(
                """
                <div class="result-spam">
                    🚨 Spam! 🚨
                </div>
                """,
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                """
                <div class="result-not-spam">
                    ✅ Not Spam ✅
                </div>
                """,
                unsafe_allow_html=True,
            )

# Footer
st.markdown(
    """
    <hr>
    <div style="text-align: center; background: rgba(255, 255, 255, 0.8); padding: 10px; border-radius: 8px;">
        <p style="font-size: 14px; color: #333;">© 2025 | Built with ❤️ by <b>Vinayak Mehta</b></p>
    </div>
    """,
    unsafe_allow_html=True,
)
