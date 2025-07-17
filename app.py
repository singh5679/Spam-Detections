from flask import Flask, render_template, request
import pickle
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

model = pickle.load(open('spam_model.pkl', 'rb'))
vectorizer = pickle.load(open('vectorizer.pkl', 'rb'))

app = Flask(__name__)

nltk.download('stopwords')
stop_words = set(stopwords.words('english'))
stemmer = PorterStemmer()

def preprocess_text(text):
    text = re.sub(r'\W', ' ', text)
    text = text.lower()
    words = text.split()
    words = [stemmer.stem(word) for word in words if word not in stop_words]
    return ' '.join(words)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    message = request.form['message']
    processed_message = preprocess_text(message)
    vect_message = vectorizer.transform([processed_message])
    prediction = model.predict(vect_message)

    result = "🚫 SPAM!" if prediction[0] else "✅ NOT SPAM!"
    return render_template('index.html', prediction=result, user_message=message)

if __name__ == "__main__":
    app.run(debug=True)
