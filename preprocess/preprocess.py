import pandas as pd
import re
from collections import defaultdict

def getPreprocessTrain(train_data_path):
  train = pd.read_csv(train_data_path)
  print('Processing text dataset for train')
  commentText = train["comment_text"]
  for commentSentence in commentText:
    commentSentence = clean_text(commentSentence)
  return train
  

def getPreprocessText(test_data_path):
  test = pd.read_csv(test_data_path)

def clean_text(text, stem_words=False):
  # regex to remove all Non-Alpha Numeric and space
  special_character_removal=re.compile(r'[^a-z\d ]',re.IGNORECASE)
  # regex to replace all numeric
  replace_numbers=re.compile(r'\d+',re.IGNORECASE)

  # Clean the text, with the option to remove stopwords and to stem words.
  text = text.lower()
  text = re.sub(r"what's", "what is ", text)
  text = re.sub(r"\'s", " ", text)
  text = re.sub(r"\'ve", " have ", text)
  text = re.sub(r"can't", "cannot ", text)
  text = re.sub(r"n't", " not ", text)
  text = re.sub(r"i'm", "i am ", text)
  text = re.sub(r"i’m", "i am", text)
  text = re.sub(r"\'re", " are ", text)
  text = re.sub(r"\'d", " would ", text)
  text = re.sub(r"\'ll", " will ", text)
  text = re.sub(r",", " ", text)
  text = re.sub(r"\.", " ", text)
  text = re.sub(r"'", " ", text)
  text = re.sub(r"\s{2,}", " ", text)
  text = replace_numbers.sub('', text)
  text = special_character_removal.sub('',text)
    
  return text