import pandas as pd

# load actual labels
actual_label_path = '../data/clean_test_labels.csv'
actual_label = pd.read_csv(actual_label_path)

# load predicted labels
pred_label_path = '../model/binary-linear-svc/binary-linear_svc.csv'
pred_label = pd.read_csv(pred_label_path)


# removing hidden test labels
actual_label = actual_label[actual_label['toxic'] >= 0]

def HammingLoss(actual_label, pred_label):
  # calculating xor of each label
  actual_label['mismatch'] = 0
  (n, l) = pred_label.shape
  actual_label = actual_label.loc[:n, :]
  for index, column in enumerate(actual_label.columns[1:-1]):
    actual_label['mismatch'] = actual_label['mismatch'] + (actual_label[column] ^ pred_label[column])

  # get number of test & example
  (n, l) = pred_label.shape
  total = actual_label['mismatch'].sum()
  loss = total / (n * (l - 1))
  print(loss)

HammingLoss(actual_label, pred_label)
