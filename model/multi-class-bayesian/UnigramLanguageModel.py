import math, collections
powerSet = (1 << 6)

class UnigramLanguageModel:

  def __init__(self, corpus):
    self.unigramCounts = [collections.defaultdict(lambda: 0)] * powerSet
    self.total = [0] * powerSet
    self.N = 13000 # estimate total number of words (including unknown words)
    self.ukp = [0.05] * powerSet # probability of unknown words
    self.train(corpus)

  def train(self, corpus):
    for index in range(len(corpus['comment_text'])):
      sentence = corpus['comment_text'][index]
      label = corpus['label'][index]
      for token in sentence:
        self.unigramCounts[label][token] = self.unigramCounts[label][token] + 1
        self.total[label] += 1
    
    totalWord = len(self.unigramCounts[0])
    for label in range(1, powerSet):
      self.ukp[label] = (float(totalWord - len(self.unigramCounts[label])) / float(totalWord)) * 0.9
    print(self.total)
  
  def score(self, sentence):
    score = [0.0] * powerSet
    for i in range(0, powerSet):
      for token in sentence:
        count = self.unigramCounts[i][token]
        if (count != 0):
          probability = (1.0 - self.ukp[i]) * (count / float(self.N)) + self.ukp[i] * (1.0 / self.N)
        else:
          probability = self.ukp[i] * (1.0 / self.N)
        if probability > 0:
          score[i] += math.log(probability)
        #else:
          #print(probability)
        # score[i] += math.log(count + 1)
        # score[i] -= math.log(self.total[i] + 70000)
    
    maxLabel = 0
    maxScore = 0.0
    minDelta = 1
    score[0] -= 0.05
    for i in range(0, powerSet):
      if (score[i] > maxScore):
        maxScore = score[i]
        maxLabel = i
      # elif minDelta > maxScore - score[i]:
      #   minDelta = maxScore - score[i]
    
    # if minDelta < 1:
    #   print(minDelta)
    return maxLabel
