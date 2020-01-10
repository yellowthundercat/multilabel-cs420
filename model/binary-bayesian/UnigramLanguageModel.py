import math, collections
powerSet = 6
listLabel = ['toxic','severe_toxic','obscene','threat','insult','identity_hate']

class UnigramLanguageModel:

  def __init__(self, corpus):
    self.unigramCounts = [collections.defaultdict(lambda: 0) for i in range(7)] 
    self.nonInUnigramCounts = [collections.defaultdict(lambda: 0) for i in range(7)] 
    self.totalYes = [0 for i in range(7)]
    self.totalNo = [0 for i in range(7)]
    self.p_label = [0 for i in range(7)]
    self.N = 13000 # estimate total number of words (including unknown words)
    self.ukp = 0.05
    self.train(corpus)

  def train(self, corpus):
    for index in range(min(1000, len(corpus['comment_text']))):
    
      if (index % 1000) == 0: 
        print(index/1000)
      sentence = corpus['comment_text'][index]
      self.p_label[6] += 1
      for column in range(0, len(listLabel)):
        label = listLabel[column]
        if corpus[label][index] == 1:
          self.p_label[column] += 1
        for token in sentence:
          
          
          if corpus[label][index] == 1:
            self.unigramCounts[column][token] += 1
            self.totalYes[column] += 1
          else:
            self.nonInUnigramCounts[column][token] += 1
            self.totalNo[column] += 1
    
    for i in range(6):
      self.p_label[i] = max(float(self.p_label[i]) / self.p_label[6], 0.000005)
    print(self.p_label)
  
  def score(self, sentence):
    YesScore = [0.0 for i in range(powerSet)]
    NoScore = [0.0 for i in range(powerSet)]
    for i in range(0, len(listLabel)):
      label = listLabel[i]
      for token in sentence:
        countYes = self.unigramCounts[i][token]
        countNo = self.nonInUnigramCounts[i][token]
        if (countYes != 0):
          yesPro = (1.0 - self.ukp) * (float(countYes) / float(self.totalYes[i]+self.N)) + self.ukp * (1.0 / float(self.N))
        else:
          yesPro = self.ukp * (1.0 / float(self.N + self.totalYes[i]))

        if (countNo != 0):
          noPro = (1.0 - self.ukp) * (float(countNo) / float(self.totalNo[i]+self.N)) + self.ukp * (1.0 / float(self.N))
        else:
          noPro = self.ukp * (1.0 / float(self.N + self.totalNo[i]))
        
        
        YesScore[i] += math.log(yesPro)
        NoScore[i] += math.log(noPro)

      YesScore[i] += math.log(self.p_label[i])
      NoScore[i] += math.log(1.0 - self.p_label[i])
      
    
    maxLabel = [0 for i in range(powerSet)]
    for i in range(0, powerSet):
      if (YesScore[i] > NoScore[i]):
        maxLabel[i] = 1
    
    return maxLabel
