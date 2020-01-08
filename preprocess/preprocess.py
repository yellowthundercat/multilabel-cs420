import pandas as pd
import re
import nltk
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
from nltk.corpus import stopwords 
from nltk.stem import WordNetLemmatizer 
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize 
import string
from collections import defaultdict

stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()
stemmer = PorterStemmer()

def getPreprocessTrain(train_data_path):
  train = pd.read_csv(train_data_path)
  print('Processing text dataset for train')
  train["comment_text"] = train["comment_text"].apply(clean_text)
  return train
  

def getPreprocessTest(test_data_path):
  test = pd.read_csv(test_data_path)
  print('Processing text dataset for testing')
  test["comment_text"] = test["comment_text"].apply(clean_text)
  return test


def clean_text(text, isRemoveDigit = True, isLemma = True, isStem = False):
  text = text.lower()
  if isRemoveDigit:
    text = re.sub('\'m|n\'t|\'s|\'re|\'d|\'ll|\'t|\'ve|\d', '', text)
  else:
    text = re.sub('\'m|n\'t|\'s|\'re|\'d|\'ll|\'t|\'ve', '', text)
  text = text.translate(str.maketrans('', '', string.punctuation))

  # text become text list
  text = word_tokenize(text)
  text = [w for w in text if not w in stop_words]
  if isLemma:
    text = [lemmatizer.lemmatize(w) for w in text]
  if isStem:
    text = [stemmer.stem(w) for w in text]
  return text
  