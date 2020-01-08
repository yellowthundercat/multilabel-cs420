import sys
import pandas as pd
sys.path.append('../../preprocess/')
import preprocess
from UnigramLanguageModel import UnigramLanguageModel
import numpy as np

data_path = '../../data/small_train.csv'
data_full_path = '../../data/preprocess_train.csv'
test_path = '../../data/preprocess_clean_test.csv'
test_label_path = '../../data/clean_test_labels.csv'


# train_data = preprocess.getPreprocessTrain(data_full_path)
# test_data = preprocess.getPreprocessTest(test_path)
train_data = pd.read_csv(data_full_path)

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
ans_predict = []
ans_actual = []
for i in range(0, len(testLabel['id'])):
  if testLabel['toxic'][i] != -1:
    testSentence = testData['comment_text'][i]
    predict = unigram.score(testSentence)
    ans_predict.append(predict)
    ans_actual.append(testLabel['label'][i])
    count += 1
    if (count % 1000 == 0):
      print(count)
    # if count > 200:
    #   break

output_df = pd.DataFrame({'label': ans_predict[:]})
label_list = ['toxic','severe_toxic','obscene','threat','insult','identity_hate']
for index, label in enumerate(label_list):
  output_df[label] = 0
  output_df[label] = (output_df['label'].apply(lambda x: (x >> index) & 1))

del output_df['label']
output_df = pd.concat([testData['id'], output_df], axis=1, sort=False)
output_df.to_csv('./multi-class-bayesian-result.csv',index=False)

# Making the Confusion Matrix
ans_actual = np.asarray(ans_actual)
ans_predict = np.asarray(ans_predict)
print(pd.crosstab(ans_actual, ans_predict, rownames=['Actual Label'], colnames=['Predicted Label']))