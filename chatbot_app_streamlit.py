import streamlit as st
import json
import string
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

nltk.download('punkt')
nltk.download('stopwords')

# Load FAQs
with open('faq_data.json', 'r') as f:
    faq_data = json.load(f)

questions = [faq['question'] for faq in faq_data]
answers = [faq['answer'] for faq in faq_data]

def preprocess(text):
    text = text.lower()
    text = text.translate(str.maketrans('', '', string.punctuation))
    tokens = word_tokenize(text)
    return ' '.join([w for w in tokens if w not in stopwords.words('english')])

preprocessed_questions = [preprocess(q) for q in questions]
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(preprocessed_questions)

def get_response(user_input):
    processed = preprocess(user_input)
    vec = vectorizer.transform([processed])
    similarity = cosine_similarity(vec, X)
    if similarity.max() < 0.3:
        return "Sorry, I don't understand that."
    else:
        return answers[similarity.argmax()]

# Streamlit UI
st.title("🤖 FAQ Chatbot")

user_input = st.text_input("Ask a question:")

if user_input:
    response = get_response(user_input)
    st.write("**Bot:**", response)
