import math, collections
powerSet = (1 << 6)

class UnigramLanguageModel:

  def __init__(self, corpus):
    self.unigramCounts = [collections.defaultdict(lambda: 0)] * powerSet
    self.total = [0] * powerSet
    self.N = 13000 # estimate total number of words (including unknown words)
    self.ukp = [0.05] * powerSet # probability of unknown words
    self.p_label = [0] * powerSet
    self.total_label = 0
    self.train(corpus)

  def train(self, corpus):
    for index in range(len(corpus['comment_text'])):
    # for index in range(min(1000, len(corpus['comment_text']))):
      if (index % 1000) == 0: 
        print(index/1000)
      self.total_label += 1
      sentence = corpus['comment_text'][index]
      label = corpus['label'][index]
      self.p_label[label] += 1
      for token in sentence:
        self.unigramCounts[label][token] = self.unigramCounts[label][token] + 1
        self.total[label] += 1
    
    for i in range(powerSet):
      self.p_label[i] = max(float(self.p_label[i]) / self.total_label, 0.0005)

    for label in range(powerSet):
      self.ukp[label] = (1.0 - self.p_label[label]) * 0.1
  
  def score(self, sentence):
    score = [0.0] * powerSet
    for i in range(0, powerSet):
      for token in sentence:
        count = self.unigramCounts[i][token]
        if (count != 0):
          probability = (1.0 - self.ukp[i]) * (float(count) / float(self.total[i]+self.N)) + self.ukp[i] * (1.0 / self.N)
        else:
          probability = self.ukp[i] * (1.0 / float(self.N + self.total[i]))
        score[i] += math.log(probability)
        score[i] += math.log(self.p_label[i])*1.0
      
    maxLabel = 0
    maxScore = score[0]
    minDelta = 1
    for i in range(0, powerSet):
      if (score[i] > maxScore):
        maxScore = score[i]
        maxLabel = i
      # elif minDelta > maxScore - score[i]:
      #   minDelta = maxScore - score[i]
    
    # if minDelta < 1:
    #   print(minDelta)
    return maxLabel
