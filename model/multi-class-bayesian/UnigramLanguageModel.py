import math, collections
powerSet = (1 << 6)

class UnigramLanguageModel:

  def __init__(self, corpus):
    self.unigramCounts = [collections.defaultdict(lambda: 0) for i in range(powerSet)]
    self.total = [0 for i in range(powerSet)]
    self.N = 13000 # estimate total number of words (including unknown words)
    self.ukp = 0.05
    self.p_label = [0 for i in range(powerSet)]
    self.total_label = 0
    self.train(corpus)

  def train(self, corpus):
    for index in range(len(corpus['comment_text'])):
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
      self.p_label[i] = max(float(self.p_label[i]) / self.total_label, 0.000005)

  
  def score(self, sentence):
    score = [0.0 for i in range(powerSet)]
    for i in range(0, powerSet):
      for token in sentence:
        count = self.unigramCounts[i][token]
        if (count != 0):
          probability = (1.0 - self.ukp) * (float(count) / float(self.total[i]+self.N)) + self.ukp * (1.0 / float(self.N))
        else:
          probability = self.ukp * (1.0 / float(self.N + self.total[i]))
        
        score[i] += math.log(probability)
      score[i] += math.log(self.p_label[i])
      
    maxLabel = 0
    maxScore = score[0]
    minDelta = 1
    for i in range(0, powerSet):
      if (score[i] > maxScore):
        maxScore = score[i]
        maxLabel = i
    
    return maxLabel
