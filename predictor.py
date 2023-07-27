import tensorflow as tf
import numpy as np
import streamlit as st
import os

# ML stuff
import transformers
from transformers import BertTokenizer, TFBertModel
from tensorflow import keras

# preprocessing library
# from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
# from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory
# from nlpretext import Preprocessor
# from nlpretext.basic.preprocess import normalize_whitespace, lower_text, remove_eol_characters, replace_currency_symbols, \
#                                         remove_punct, remove_multiple_spaces_and_strip_text, filter_non_latin_characters

#GOOGLE_DRIVE_FILE_ID = "YOUR GOOGLE DRIVE FILE ID HERE"

# set maximum length and tokenizer
MAX_LEN = 100
tokenizer = BertTokenizer.from_pretrained('bert-large-uncased')

# stemmer
# stemmer_factory = StemmerFactory()
# stemmer = stemmer_factory.create_stemmer()

# stopword
# stopword_factory = StopWordRemoverFactory()
# stopword = stopword_factory.create_stop_word_remover()

# use nlpretext processor
# preprocessor = Preprocessor()
# preprocessor.pipe(lower_text)
# preprocessor.pipe(remove_eol_characters)
# preprocessor.pipe(normalize_whitespace)
# preprocessor.pipe(remove_multiple_spaces_and_strip_text)
# preprocessor.pipe(remove_punct)
# preprocessor.pipe(replace_currency_symbols)
# preprocessor.pipe(filter_non_latin_characters)

# load model on first launch
@st.cache(allow_output_mutation=True)
def load_model():
	# filepath = "model/model.h5"

	# folder exists?
	#if not os.path.exists('model'):
		# create folder
		#s.mkdir('model')
	
	# file exists?
	#if not os.path.exists(filepath):
		# download file
		#from gd_download import download_file_from_google_drive
		#download_file_from_google_drive(id=GOOGLE_DRIVE_FILE_ID, destination=filepath)*/
	
	# load model
	model = tf.keras.models.load_model("model.h5", custom_objects={"TFBertModel": transformers.TFBertModel})
	return model

# def cleanText(sentence):
#     # process with PySastrawi first
#     stemmed = stemmer.stem(sentence)
#     stopwordremoved = stopword.remove(stemmed)

#     # then with nlpretext
#     cleaned = preprocessor.run(stopwordremoved)

#     # return
#     return cleaned
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer 
ps= PorterStemmer()
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

def bert_encode(data, MAX_LEN) :
    input_ids = []
    attention_masks = []

    for text in data:
        encoded = tokenizer.encode_plus(
            text, 
            add_special_tokens=True,
            max_length=60,
            pad_to_max_length=True,

            return_attention_mask=True,
        )
        input_ids.append(encoded['input_ids'])
        attention_masks.append(encoded['attention_mask'])
        
    return np.array(input_ids),np.array(attention_masks)

def predict(model, input):
	train_input_ids, train_attention_masks = bert_encode(input,60)
	data = [train_input_ids, train_attention_masks]

	prediction = model.predict(data)
	prediction = prediction[0].item() * 100

	return prediction
print("successful")    
