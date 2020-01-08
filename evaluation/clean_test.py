import sys
sys.path.append('../preprocess/')
import preprocess
import pandas as pd

test_labels = pd.read_csv('../data/test_labels.csv')
test = pd.read_csv('../data/test.csv')

# test = pd.concat([test, test_labels], axis=1, sort=False)
test = test.merge(test_labels, how='outer').fillna(0)

test = test[test['toxic'] >= 0]

test_labels = test.loc[:, ['id', 'toxic','severe_toxic','obscene','threat','insult','identity_hate']]
test = test.loc[:, ['id', 'comment_text']]

print(test_labels.head())
print(test.head())

test_labels.to_csv('../data/clean_test_labels.csv',index=False)
test.to_csv('../data/clean_test.csv',index=False)

