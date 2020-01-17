import pickle
import numpy as np
import re
import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords

import os
import json

import sys,inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0,parentdir) 

from utils import identity

from sklearn.feature_extraction.text import CountVectorizer



with open ("../X_tr_val.pkl", "rb") as f:
    X_tr_val = pickle.load(f)
with open ("../y_tr_val.pkl", "rb") as f:
    y_tr_val = pickle.load(f)
with open ("../stop_words.pkl", "rb") as f:
    stop_words = pickle.load(f)
with open ("../model.pkl", "rb") as f:
    model = pickle.load(f)

    
    


def lemmatize_text(text):
    '''Function to lemmatize any given text'''
    w_tokenizer = nltk.tokenize.WhitespaceTokenizer()
    lemmatizer = nltk.stem.WordNetLemmatizer()
    return [lemmatizer.lemmatize(w) for w in w_tokenizer.tokenize(text)]



def clean_text(text):
    '''Function to preprocess text from user and return
    cleaned and tokenized words
    '''
    text = [lemmatize_text(text)]
    stop_words = stopwords.words('english')
    cleaned = [item for item in text if item not in stop_words]
    return cleaned



def make_prediction(text):
    """
    Function takes in tokenized text, applies count vectorizer
    and returns prediction- fake or real and prediction probability

    """
    c_bin = CountVectorizer(tokenizer=lambda doc: doc,lowercase=False, binary = 'boolean')
    c_vectorizer = c_bin.fit(X_tr_val)
    cleaned = clean_text(text)

    text_vector = c_vectorizer.transform(cleaned)
    text_vector = text_vector.todense()
    prediction = model.predict(text_vector) 
    probability = model.predict_proba(text_vector)
    
    if prediction[0] == 0:
        prob = np.round(probability[0][0]* 100,2)
        part_1 = "Hmmm... I'm "
        part_2 = str(prob)
        part_3 = "% confident that this is fake news!"
        pred = part_1 + part_2 + part_3
        
    else:
        prob = np.round(probability[0][1]*100,2)
        part_1 = "Hurray, I'm "
        part_2 = str(prob)
        part_3 = "% confident that you are reading real news!"
        pred = part_1 + part_2 + part_3
    return {'pred':pred}
if __name__ == '__main__':
    pass
  
