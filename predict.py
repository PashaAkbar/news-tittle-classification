import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing.text import Tokenizer
from keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.tokenize import word_tokenize
from sklearn.model_selection import train_test_split
import tensorflow as tf
from nltk.corpus import stopwords 
from nltk.tokenize import word_tokenize
from itertools import chain
import pickle

df = pd.read_csv('data/indonesian-news-title.csv')

max_features = 2000
max_len = 200
teksPrediksi = ""


data = np.load('tokenized.npy',allow_pickle=True)
data = pd.Series(data)
joined = data.apply(lambda x: ' '.join(x))
tokenizer = Tokenizer(num_words=max_features, split=' ')
tokenizer.fit_on_texts(joined.values)

X = tokenizer.texts_to_sequences(joined.values)
X = pad_sequences(X, maxlen=max_len)
# Y = pd.get_dummies(df['category']).values

tfidf_vectorizer = TfidfVectorizer(use_idf=True, min_df=0.0001, max_df=0.02, lowercase=False)

# Dellete "news" and "hot" category to remove any overlapping category
df.drop(df[df.category == 'news'].index, inplace=True)
df.drop(df[df.category == 'hot'].index, inplace=True)

# See category stats
df['category'].value_counts()

df.drop(df[df['title'].duplicated()].index, inplace=True)

df['tokens'] = df['title'].apply(lambda x: word_tokenize(x))

stop_words = set(chain(stopwords.words('indonesian'), stopwords.words('english')))

df['tokens'].apply(lambda x: [w for w in x if not w in stop_words])

#Preprocess
df['tokens'] = np.load('tokenized.npy', allow_pickle=True)
df['joined_tokens'] = df['tokens'].apply(lambda x: ' '.join(x))

X_train, X_test, y_train, y_test = train_test_split(df['joined_tokens'], df['category'], test_size=0.3, random_state=42)

X_train_vectors_tfidf = tfidf_vectorizer.fit_transform(X_train) 

labels = ['finance','food','health','inet','oto','sport','travel']

filename = 'finalized_model.sav'
loaded_model = pickle.load(open(filename, 'rb'))
filename = 'SVM_model.sav'
loaded_model_SVM = pickle.load(open(filename, 'rb'))


def prediksiModel(text):
    text_seq = tokenizer.texts_to_sequences([text])
    text_seq = pad_sequences(text_seq, maxlen=200)
    pred = loaded_model.predict(text_seq)[0]
    predict = labels[np.argmax(pred)]
    tampilPred(predict)
    # st.subheader("Hasil Sentimen: "+ predict)

def preprocess(input):
    return tfidf_vectorizer.transform(input) 

def prediksiModelSVM(text):
    preprocessed_text = preprocess([text])
    # x = tfidf_vectorizer.transform([preprocessed_text]) 
    pred = loaded_model_SVM.predict(preprocessed_text)
    # predict = labels[np.argmax(pred)]
    tampilPred(pred[0])
    # st.subheader("Hasil Sentimen: "+ predict)
    

st.title('Teks Klasifikasi pada Judul Berita Detik.com dengan Menggunakan Metode LSTM')


teks = st.text_input("Masukkan Teks: ")
clicked = st.button("Prediksi Teks LSTM")
clickedSVM = st.button("Prediksi Teks SVM")
def tampilPred(txt):
    st.header("Kategori Berita: "+txt)


# clicked = st.button("Prediksi Teks SVM")
if clickedSVM:
    prediksiModelSVM(teks)

if clicked:
    prediksiModel(teks)

