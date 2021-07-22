import streamlit as st
import pickle
import numpy as np
import pandas as pd
import nltk
import re

nltk.download('punkt') # punkt tokenizer

nltk.download('stopwords')
from nltk.corpus import stopwords

nltk.download('wordnet')
from nltk.stem import WordNetLemmatizer
wordnet = WordNetLemmatizer()

mnb, tfidf, le = pickle.load(open('/content/naivebayes.pkl', 'rb'))

def clean_data(X):
  # corpus = []

  # for review in X:
  review = X.lower()
  review = re.sub('<br \/>', ' ', review)
  review = re.sub('[^a-zA-Z]', ' ', review)
  words = nltk.word_tokenize(review)
  words = [wordnet.lemmatize(word) for word in words if word not in stopwords.words('english')]
  w = ' '.join(words)
  # corpus.append(w)
  # return corpus
  return w

def classify_sentiment(review):
  
  review = clean_data(review)
  review = tfidf.transform([review]).toarray()  
  y_pred = mnb.predict(review)
  y_pred = le.inverse_transform(y_pred)
  return y_pred[0]


def main():
    st.title('Sentiment Analysis')

    html_temp = """
        <div style="background-color:#025246 ;padding:10px;">
        <h2 style="color:white;text-align:center;">Sentiment Analysis</h2>
        </div>
    """
    st.markdown(html_temp, unsafe_allow_html=True)

    review = st.text_input('Add review', '')
    
    if st.button('Check sentiment'):
        output = classify_sentiment(review)
        st.success('{} review'.format(output))

if __name__ == '__main__':
    main()