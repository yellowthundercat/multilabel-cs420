import sys
sys.path.append('../../preprocess/')
import preprocess
from sklearn.metrics import accuracy_score

data_path = '../../data/train.csv'

train_data = preprocess.getPreprocessTrain(data_path)

# choosing number of data
# train_data = train_data.loc[:100, :]

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.externals import joblib
print('Libraries Imported')

# insert column label
train_data['label'] = 0

# transform label to powerset
for index, column in enumerate(train_data.columns[2:-1]):
  print(index, column)
  train_data['label'] = train_data['label'] | train_data[column].apply(lambda x: (x << index))

train_data = train_data.loc[:, ['comment_text', 'label']]
# filter 0 label
train_data = train_data[train_data['label'] > 0]
print(train_data.head())

# vectorize text
print('Vectorizing text ...')
# vectorizer = CountVectorizer(min_df=1)
vectorizer = TfidfVectorizer(binary=True, use_idf=True)
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
classifier = RandomForestClassifier(n_estimators = 20, criterion = 'entropy', random_state = 42)
classifier.fit(x_train, y_train)

# Predicting the Test set results
print('Get predictions ...')
y_pred = classifier.predict(x_test)
y_test_np = y_test.to_numpy()

# Making the Confusion Matrix
print(pd.crosstab(y_test, y_pred, rownames=['Actual Label'], colnames=['Predicted Label']))

# accuracy
print("Accuracy = ", accuracy_score(y_test, y_pred))
print("\n")

# use to predict
print('Testing ...')
test_data_path = '../../data/clean_test.csv'
test_data = preprocess.getPreprocessTrain(data_path)

# vectorize text
print('Vectorizing ...')
test_x = vectorizer.transform(test_data['comment_text']).toarray()

# Feature Scaling
print('Feature Scaling ...')
scaler = StandardScaler()
test_x = scaler.transform(test_x)

output = classifier.predict(test_x)
output_df = pd.DataFrame({'label': output[:]})


label_list = ['toxic','severe_toxic','obscene','threat','insult','identity_hate']

for index, label in enumerate(label_list):
  output_df[label] = 0
  output_df[label] = (output_df['label'].apply(lambda x: (x >> index) & 1))

del output_df['label']
output_df = pd.concat([test_data['id'], output_df], axis=1, sort=False)

output_df.to_csv('./powerset-random-forest.csv',index=False)