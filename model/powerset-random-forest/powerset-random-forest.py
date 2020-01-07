import sys
sys.path.append('../../preprocess/')
import preprocess

data_path = '../../data/train.csv'

# train_data = preprocess.getPreprocessTrain(data_path)
train_data = preprocess.getPreprocessText(data_path)

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
# print(train_data.head())

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

# Fitting Random Forest Classification to the Training set
print('Training ...')
classifier = RandomForestClassifier(n_estimators = 10, criterion = 'entropy', random_state = 42)
classifier.fit(x_train, y_train)

# Predicting the Test set results
print('Get predictions ...')
y_pred = classifier.predict(x_test)

y_test_np = y_test.to_numpy()
print(type(y_test_np))
print(type(y_pred))

# Making the Confusion Matrix
print(pd.crosstab(y_test, y_pred, rownames=['Actual Label'], colnames=['Predicted Label']))