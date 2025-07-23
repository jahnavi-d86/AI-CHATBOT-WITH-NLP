!pip install --quiet openai nltk scikit-learn

import openai
import nltk
import random
import string
import warnings
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

warnings.filterwarnings("ignore")

nltk.download('punkt')
nltk.download('wordnet')
nltk.download('stopwords')

client = openai.OpenAI(
    api_key="sk-your-new-api-key-here"  
)
chat_corpus = """
Hello! How can I help you?
Hi there! I have a question.
What is your name?
I am a chatbot developed for CODTECH Internship Task 3.
What can you do?
I can answer basic questions based on this demo.
How are you?
I am just code, but I'm functioning as expected!
Thanks
You're welcome!
Bye
Goodbye and have a great day!
"""
sent_tokens = nltk.sent_tokenize(chat_corpus.lower())
lemmer = WordNetLemmatizer()
remove_punct_dict = dict((ord(punct), None) for punct in string.punctuation)

def LemTokens(tokens):
    return [lemmer.lemmatize(token) for token in tokens if token not in stopwords.words('english')]

def LemNormalize(text):
    return LemTokens(nltk.word_tokenize(text.lower().translate(remove_punct_dict)))
GREETING_INPUTS = ("hello", "hi", "greetings", "sup", "what's up", "hey")
GREETING_RESPONSES = ["hi there", "hello", "I'm glad you're here", "how can I help you?"]

def greeting(sentence):
    for word in sentence.split():
        if word.lower() in GREETING_INPUTS:
            return random.choice(GREETING_RESPONSES)
def ask_ai(query):
    try:
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": query}
            ]
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"[AI Error] {e}"
def generate_response(user_input):
    sent_tokens.append(user_input.lower())
    vectorizer = TfidfVectorizer(tokenizer=LemNormalize, stop_words='english')
    tfidf = vectorizer.fit_transform(sent_tokens)
    
    vals = cosine_similarity(tfidf[-1], tfidf[:-1])
    idx = vals.argsort()[0][-1]
    flat = vals.flatten()
    flat.sort()
    score = flat[-1]
    
    sent_tokens.pop()

    if score < 0.3:
        return ask_ai(user_input)
    else:
        return sent_tokens[idx]
print("ChatBot: Hello! I'm your hybrid chatbot. Ask me anything (type 'bye' to exit).\n")

while True:
    user_input = input("You: ").lower()
    if user_input == 'bye':
        print("ChatBot: Goodbye! Have a nice day.")
        break
    elif user_input in ['thanks', 'thank you']:
        print("ChatBot: You're welcome!")
    elif greeting(user_input):
        print("ChatBot:", greeting(user_input))
    else:
        print("ChatBot:", generate_response(user_input))
