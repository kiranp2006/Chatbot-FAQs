import streamlit as st
import json
import string
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Load FAQs from JSON
with open("faq_data.json", "r") as f:
    faq_data = json.load(f)

questions = [faq['question'] for faq in faq_data]
answers = [faq['answer'] for faq in faq_data]

# Completely NLTK-free preprocessing
def preprocess(text):
    text = text.lower()
    text = text.translate(str.maketrans('', '', string.punctuation))  # Remove punctuation
    tokens = text.split()  # Split by whitespace
    stop_words = {'is', 'a', 'the', 'what', 'how', 'to', 'in', 'on', 'and', 'for', 'of', 'this', 'that'}
    filtered = [word for word in tokens if word not in stop_words]
    return ' '.join(filtered)

# Vectorize the questions
preprocessed_questions = [preprocess(q) for q in questions]
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(preprocessed_questions)

# Find best matching answer
def get_response(user_input):
    processed_input = preprocess(user_input)
    input_vec = vectorizer.transform([processed_input])
    similarity = cosine_similarity(input_vec, X)

    if similarity.max() < 0.3:
        return "❌ Sorry, I don't understand that."
    else:
        best_match_index = similarity.argmax()
        return answers[best_match_index]

# Streamlit UI
st.title("🤖 FAQ Chatbot")
st.write("Ask me a question related to Python or AI!")

user_input = st.text_input("You:")

if user_input:
    st.write("**Bot:**", get_response(user_input))


