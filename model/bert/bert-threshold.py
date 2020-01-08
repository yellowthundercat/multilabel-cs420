import pandas as pd

prob_path = './bert-clean-probability.csv'
prob = pd.read_csv(prob_path)


threshold = 0.3
while threshold < 1:
  print('threshold', threshold)
  filename = './pred/bert_' + str(threshold) + '.csv'
  pred = prob.copy()
  for index, column in enumerate(pred.columns[1:]):
    print(pred[column][24])
    pred[column] = pred[column].apply(lambda x: int(float(x) >= threshold))
    print(pred[column][24])
  pred.to_csv(filename, index=False)
  threshold += 0.1