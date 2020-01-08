import pandas as pd

# load clean test label
clean_label_path = '../../data/clean_test_labels.csv'
clean_label = pd.read_csv(clean_label_path)

# load probability
prob_path = './bert-probability.csv'
prob_label = pd.read_csv(prob_path)

id_list = clean_label['id']

clean_prob = pd.merge(id_list, prob_label, on='id')

print(id_list.head())
print(prob_label.head())
print(clean_prob.head())

clean_prob.to_csv('./bert-clean-probability.csv',index=False)