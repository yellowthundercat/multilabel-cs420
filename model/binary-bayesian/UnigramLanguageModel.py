import math, collections
powerSet = 6
thresh_hold = 0.1
listLabel = ['toxic','severe_toxic','obscene','threat','insult','identity_hate']

class UnigramLanguageModel:

  def __init__(self, corpus):
    self.unigramCounts = [collections.defaultdict(lambda: 0)] * 7
    self.nonInUnigramCounts = [collections.defaultdict(lambda: 0)] * 7
    self.totalYes = [0] * 7
    self.totalNo = [0] * 7
    self.p_label = [0] * 7
    self.train(corpus)

  def train(self, corpus):
    # for index in range(min(1000, len(corpus['comment_text']))):
    for index in range(len(corpus['comment_text'])):
      if (index % 1000) == 0: 
        print(index/1000)
      sentence = corpus['comment_text'][index]
      self.p_label[6] += 1
      for column in range(0, len(listLabel)):
        label = listLabel[column]
        if corpus[label][index] == 1:
          self.p_label[column] += 1
        for token in sentence:
          if column == 0:
            self.unigramCounts[6][token] = self.unigramCounts[6][token] + 1
            self.totalYes[6] += 1
          
          if corpus[label][index] == 1:
            self.unigramCounts[column][token] = self.unigramCounts[column][token] + 1
            self.totalYes[column] += 1
          else:
            self.nonInUnigramCounts[column][token] += 1
            self.totalNo[column] += 1
    
    for i in range(6):
      self.p_label[i] = max(float(self.p_label[i]) / self.p_label[6], 0.005)
    print(self.p_label)
  
  def score(self, sentence):
    YesScore = [0.0] * powerSet
    NoScore = [0.0] * powerSet
    for i in range(0, len(listLabel)):
      label = listLabel[i]
      for token in sentence:
        countYes = self.unigramCounts[i][token]
        countNo = self.nonInUnigramCounts[i][token]
        # score[i] += float(count + 1) / float(self.total[i] + 70000) * self.p_label[i] 
        YesScore[i] += math.log(countYes + 1)
        YesScore[i] -= math.log(self.totalYes[i] + 70000)

        NoScore[i] += math.log(countNo + 1)
        NoScore[i] -= math.log(self.totalNo[i] + 70000)

        YesScore[i] += math.log(self.p_label[i])
        NoScore[i] += math.log(1.0 - self.p_label[i])
      
      # print('Test ', i)
      # print(YesScore)
      # print(NoScore)
      
    # print(YesScore)
    # print(NoScore)
    
    maxLabel = [0] * powerSet
    for i in range(0, powerSet):
      if (YesScore[i] > NoScore[i]):
        maxLabel[i] = 1
    
    return maxLabel
