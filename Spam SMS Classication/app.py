import streamlit as st
import nltk
import pickle
import string
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

ps = PorterStemmer()

tfidf = pickle.load(open('vectorizer.pkl', 'rb'))
model = pickle.load(open('model.pkl', 'rb'))

st.title('Spam Classification')
input_sms = st.text_area('Enter the message')



def transform_text(text):
    text = text.lower()
    text = nltk.word_tokenize(text)

    y = []
    for i in text:
        if i.isalnum() and i not in stopwords.words('english') and i not in string.punctuation:
            y.append(ps.stem(i))

    return " ".join(y)

if st.button('Predict'):
    transformed_sms = transform_text(input_sms)

    vector_input = tfidf.transform([transformed_sms]).toarray()

    prediction = model.predict(vector_input)[0]

    if prediction == 1:
        st.header('The message is spam')
    else:
        st.header('The message is Not Spam')


