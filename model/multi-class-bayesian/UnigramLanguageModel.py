import math, collections
powerSet = (1 << 6)

class UnigramLanguageModel:

  def __init__(self, corpus):
    self.unigramCounts = [collections.defaultdict(lambda: 0)] * powerSet
    self.total = [0] * powerSet
    self.train(corpus)

  def train(self, corpus):
    for index in range(len(corpus['comment_text'])):
      sentence = corpus['comment_text'][index]
      label = corpus['label'][index]
      for token in sentence:
        self.unigramCounts[label][token] = self.unigramCounts[label][token] + 1
        self.total[label] += 1
    
    print(self.total)
  
  def score(self, sentence):
    score = [0.0] * powerSet
    for i in range(0, powerSet):
      for token in sentence:
        count = self.unigramCounts[i][token]
        score[i] += math.log(count + 1)
        score[i] -= math.log(self.total[i] + 70000)
    
    maxLabel = 0
    maxScore = 0.0
    for i in range(0, powerSet):
      if (score[i] > maxScore):
        maxScore = score[i]
        maxLabel = i
    
    return maxLabel
