import streamlit as st
import pickle
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
stem=PorterStemmer()

def transform_test(text):
    text = text.lower()
    text = nltk.word_tokenize(text)  # used to divide into words

    y = []
    for i in text:
        if (i.isalnum()):
            y.append(i)  # remove special characters
    text = y[:]
    y.clear()

    for i in text:
        if (i not in stopwords.words('English') and i not in string.punctuation):  # remove stopwords and punctuations
            y.append(i)
    text = y[:]
    y.clear()

    for i in text:
        y.append(stem.stem(i))
    return " ".join(y)
tf=pickle.load(open("C:/Users/PAVILION/Desktop/Mini_project_final/Spam_ham_classifier/vectorizer.pkl",'rb'))
model=pickle.load(open("C:/Users/PAVILION/Desktop/Mini_project_final/Spam_ham_classifier/model.pkl",'rb'))

st.title('SMS spam Classifier')

sms=st.text_input("Enter the message:")

transformed_sms=transform_test(sms)

vector_input=tf.transform([transformed_sms])

result=model.predict(vector_input)[0]

if result==1:
    st.header("Spam")
else:
    st.header("Not spam")