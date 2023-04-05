import pandas as pd
from tqdm import tqdm
import re, string
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords 
from imblearn.over_sampling import SMOTE
from itertools import chain
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score



tqdm.pandas()

df = pd.read_csv('data/indonesian-news-title.csv')

# Total rows in each category
df['category'].value_counts()

# Get average words in each category
df['word_count'] = df['title'].apply(lambda x: len(x.split()))
df.groupby('category')['word_count'].mean()

# Get average characters in each category
df['char_count'] = df['title'].apply(lambda x: len(str(x)))
df.groupby('category')['char_count'].mean()

# Is there any duplicated rows?
df[df['title'].duplicated()]


# Tokenizing
# nltk.download('punkt')
# # Stopwords removal
# nltk.download('stopwords')
# # Lemmatizer
# nltk.download('wordnet')
# nltk.download('omw-1.4')

# Dellete "news" and "hot" category to remove any overlapping category
df.drop(df[df.category == 'news'].index, inplace=True)
df.drop(df[df.category == 'hot'].index, inplace=True)

# See category stats
df['category'].value_counts()

df.drop(df[df['title'].duplicated()].index, inplace=True)


# Cleaning
def cleaning(text):
  # Case folding
  text = text.lower() 
  # Trim text
  text = text.strip()
  # Remove punctuations, special characters, and double whitespace
  text = re.compile('<.*?>').sub('', text) 
  text = re.compile('[%s]' % re.escape(string.punctuation)).sub(' ', text)
  text = re.sub('\s+', ' ', text)
  # Number removal
  text = re.sub(r'\[[0-9]*\]', ' ', text) 
  text = re.sub(r'[^\w\s]', '', str(text).lower().strip())
  # Remove number and whitespaces
  text = re.sub(r'\d', ' ', text)
  text = re.sub(r'\s+', ' ', text)

  return text

df['title'] = df['title'].apply(lambda x: cleaning(x))

df['tokens'] = df['title'].apply(lambda x: word_tokenize(x))

stop_words = set(chain(stopwords.words('indonesian'), stopwords.words('english')))

df['tokens'].apply(lambda x: [w for w in x if not w in stop_words])

factory = StemmerFactory()
stemmer = factory.create_stemmer()

df['tokens'] = np.load('tokenized.npy',allow_pickle=True)
df['joined_tokens'] = df['tokens'].apply(lambda x: ' '.join(x))
X_train, X_test, y_train, y_test = train_test_split(df['joined_tokens'], df['category'], test_size=0.3, random_state=42)


tfidf_vectorizer = TfidfVectorizer(use_idf=True, min_df=0.0001, max_df=0.02, lowercase=False)

X_train_vectors_tfidf = tfidf_vectorizer.fit_transform(X_train) 
X_test_vectors_tfidf = tfidf_vectorizer.transform(X_test)


X_train_resampled, y_train_resampled = SMOTE(random_state=42).fit_resample(X_train_vectors_tfidf, y_train)

clf = SVC(kernel='rbf')
clf.fit(X_train_resampled, y_train_resampled)

y_pred = clf.predict(X_test_vectors_tfidf)

print('Accuracy score : ', accuracy_score(y_test, y_pred))
print('Precision score : ', precision_score(y_test, y_pred, average='weighted'))
print('Recall score : ', recall_score(y_test, y_pred, average='weighted'))
print('F1 score : ', f1_score(y_test, y_pred, average='weighted'))

filename = 'SVM_model_1.sav'
pickle.dump(clf, open(filename, 'wb'))

# Plot the training and validation accuracy
plt.plot(clf.history['accuracy'])
plt.plot(clf.history['val_accuracy'])
plt.title('Model Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.savefig('accuracy_new.png')
plt.show()

