import sys
import pandas as pd
sys.path.append('../../preprocess/')
import preprocess
import numpy as np
from UnigramLanguageModel import UnigramLanguageModel

data_path = '../../data/small_train.csv'
data_full_path = '../../data/train.csv'
test_path = '../../data/test.csv'
test_label_path = '../../data/test_labels.csv'

train_data = preprocess.getPreprocessTrain(data_path)
#train_data = preprocess.getPreprocessTrain(data_path)
# test_data = preprocess.getPreprocessTest(test_path)

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

testData = pd.read_csv(test_path)
count = 0
countRight = 0
countDifZero = 0
ans_predict = []
ans_actual = []
ArrayLabel = np.asarray(testLabel)
for i in range(0, len(testLabel['id'])):
  if testLabel['toxic'][i] != -1:
    testSentence = testData['comment_text'][i]
    testSentence = preprocess.clean_text(testSentence)
    predict = unigram.score(testSentence)
    print(predict)
    print(ArrayLabel[i])
    count += 1
    if count > 200:
      break
# print (count, countRight, countDifZero)
