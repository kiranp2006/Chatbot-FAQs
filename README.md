# 🤖 FAQ Chatbot using Streamlit

This is a simple AI-powered chatbot that answers frequently asked questions (FAQs). It uses **NLP techniques** and **cosine similarity** to match user input with the best-matching FAQ and reply with a suitable answer.

Built using:
- Python
- Streamlit (Web UI)
- NLTK (for preprocessing)
- scikit-learn (TF-IDF + similarity)

---

## 🚀 Live Demo

🔗 [Click here to try the chatbot live](https://your-app-url.streamlit.app)  
_(Replace with your Streamlit link after deploying)_

---

## 🧠 How It Works

1. Loads questions and answers from `faq_data.json`
2. Preprocesses text using NLTK (lowercase, remove stopwords, etc.)
3. Vectorizes questions with **TF-IDF**
4. Compares user input using **cosine similarity**
5. Returns the best-matching answer

---

## 💻 How to Run Locally

### ✅ Prerequisites

Install Python packages:

```bash
pip install -r requirements.txt

