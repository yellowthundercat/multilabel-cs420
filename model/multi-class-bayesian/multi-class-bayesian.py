import sys
sys.path.append('../../preprocess/')
import preprocess

data_path = '../../data/small_train.csv'
data_full_path = '../../data/train.csv'

train_data = preprocess.getPreprocessTrain(data_path)
print(train_data["comment_text"][0])