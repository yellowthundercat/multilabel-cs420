import sys
sys.path.append('../../preprocess/')
import preprocess
from sklearn.metrics import accuracy_score

data_path = '../../data/train.csv'

train_data = preprocess.getPreprocessTrain(data_path)

# train_data = train_data.loc[:10000, :]

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
from sklearn.externals import joblib
print('Libraries Imported')

# insert column label
train_data['label'] = 0

# transform label to powerset
for index, column in enumerate(train_data.columns[2:-1]):
  print(index, column)
  train_data['label'] = train_data['label'] | train_data[column].apply(lambda x: (x << index))

train_data = train_data.loc[:, ['comment_text', 'label']]
train_data = train_data[train_data['label'] > 0]
print(train_data.head())

# vectorize text
print('Vectorizing text ...')
vectorizer = CountVectorizer(min_df=1)
x = vectorizer.fit_transform(train_data['comment_text']).toarray()
y = train_data['label']

# Creating the Training and Test set from data
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.25, random_state = 21)

# Feature Scaling
print('Feature Scaling')
scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

# using Label Powerset
from sklearn.linear_model import LogisticRegression
from skmultilearn.problem_transform import LabelPowerset
# initialize label powerset multi-label classifier
classifier = LogisticRegression()
# train
classifier.fit(x_train, y_train)
# predict
predictions = classifier.predict(x_test)
# accuracy
print("Accuracy = ",accuracy_score(y_test,predictions))
print("\n")