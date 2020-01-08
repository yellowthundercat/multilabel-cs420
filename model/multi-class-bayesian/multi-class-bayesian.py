import sys
import pandas as pd
sys.path.append('../../preprocess/')
import preprocess
from UnigramLanguageModel import UnigramLanguageModel

data_path = '../../data/small_train.csv'
data_full_path = '../../data/train.csv'
test_path = '../../data/test.csv'
test_label_path = '../../data/test_labels.csv'

def test(LanguageModel):
  return False

train_data = preprocess.getPreprocessTrain(data_full_path)
# test_data = preprocess.getPreprocessTest(test_path)

# insert column label
train_data['label'] = 0

# transform label to powerset
for index, column in enumerate(train_data.columns[2:-1]):
  # print(index, column)
  train_data['label'] = train_data['label'] | train_data[column].apply(lambda x: (x << index))

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
for i in range(0, len(testLabel['id'])):
  if testLabel['toxic'][i] != -1:
    testSentence = testData['comment_text'][i]
    testSentence = preprocess.clean_text(testSentence)
    predict = unigram.score(testSentence)
    # print(predict, testLabel['label'][i])
    count += 1
    if (predict == testLabel['label'][i]):
      countRight += 1
      if predict != 0:
        countDifZero += 1
    if count > 2000:
      break
print (count, countRight, countDifZero)




