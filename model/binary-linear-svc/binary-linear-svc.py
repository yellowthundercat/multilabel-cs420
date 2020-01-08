import sys
sys.path.append('../../preprocess/')
import preprocess

import re
import matplotlib
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score
from sklearn.multiclass import OneVsRestClassifier
from nltk.corpus import stopwords
stop_words = set(stopwords.words('english'))
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

data_path = '../../data/train.csv'

train_data = preprocess.getPreprocessTrain(data_path)
seperator = ' '
train_data['comment_text'] = train_data['comment_text'].apply(lambda x: seperator.join(x))

# choosing number of data
train_data = train_data.loc[:10, :]

X_train = train.comment_text

SVC_pipeline = Pipeline([
    ('tfidf', TfidfVectorizer(stop_words=stop_words)),
    ('clf', OneVsRestClassifier(LinearSVC(), n_jobs=1)),
  ])

categories = list(train_data.columns.values)[2:]

for category in categories:
  print('... Processing {}'.format(category))
  # train the model using X_dtm & y
  SVC_pipeline.fit(X_train, train[category])
  svc_name = '/content/drive/My Drive/Models/SVC/SVC_{}'.format(category)

  # save svc model
  pickle.dump(SVC_pipeline, open(svc_name, "wb"))
  # compute the testing accuracy
  # prediction = SVC_pipeline.predict(X_test)