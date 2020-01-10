import sys
import pandas as pd
sys.path.append('../../preprocess/')
import preprocess
import copy
import numpy as np
from UnigramLanguageModel import UnigramLanguageModel

data_path = '../../data/small_train.csv'
data_full_path = '../../data/preprocess_train.csv'
test_path = '../../data/preprocess_clean_test.csv'
test_label_path = '../../data/clean_test_labels.csv'


train_data = pd.read_csv(data_full_path)

# insert column label
train_data['label'] = 0

print('run unigram model')
unigram = UnigramLanguageModel(train_data)

#testing
testLabel = pd.read_csv(test_label_path)

# transform label to powerset
testLabel['label'] = 0
for index, column in enumerate(testLabel.columns[1:-1]):
  testLabel['label'] = testLabel['label'] | testLabel[column].apply(lambda x: (x << index))

# testData = pd.read_csv(test_path)
testData = pd.read_csv(test_path)
count = 0
countRight = 0
countDifZero = 0
ans_predict = []
ans_actual = []
label_list = ['toxic','severe_toxic','obscene','threat','insult','identity_hate']
for i in range(0, len(testLabel['id'])):
  if testLabel['toxic'][i] != -1:
    testSentence = testData['comment_text'][i]
    
    predict = unigram.score(testSentence)
    for index, label in enumerate(label_list):
      testLabel[label][i] = predict[index] 
    count += 1
    if (count % 1000 == 0):
      print(count)
    if count > 1000:
      break

del testLabel['label']
testLabel.to_csv('./binary-bayesian-result.csv',index=False)