import math, collections
powerSet = 6
thresh_hold = 0.1

class UnigramLanguageModel:

  def __init__(self, corpus):
    self.unigramCounts = [collections.defaultdict(lambda: 0)] * powerSet
    self.total = [0] * powerSet
    self.train(corpus)

  def train(self, corpus):
    for index in range(len(corpus['comment_text'])):
      sentence = corpus['comment_text'][index]
      for token in sentence:
        for column in range(0, powerSet):
          if corpus[column + 1][index] == 1:
            self.unigramCounts[column][token] = self.unigramCounts[column][token] + 1
            self.total[column] += 1
    
    print(self.total)
  
  def score(self, sentence):
    score = [0.0] * powerSet
    for i in range(0, powerSet):
      for token in sentence:
        count = self.unigramCounts[i][token]
        score[i] += math.log(count + 1)
        score[i] -= math.log(self.total[i] + 70000)
    
    maxLabel = [0] * powerSet
    for i in range(0, powerSet):
      if (score[i] > thresh_hold):
        maxLabel[i] = 1
    
    return maxLabel
