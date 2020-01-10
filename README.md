# Approaches to multi-label classification for text

The github is for final project of CS418-Natural Language Processing course. The report can be found [here]().
The github includes:
  - Dataset from [Kaggle Toxic Comment Classification Challenge](https://www.kaggle.com/c/jigsaw-toxic-comment-classification-challenge/)
  - Implementation and Colab Notebook of 5 models:
    - Powerset using Naive Bayesian
    - Powerset using Random Forest
    - OneVsAll using Linear SVC
    - OneVsAll using Naive Bayesian
    - Transfer learning using BERT
  - Evaluation using Hamming-Loss

##### Powerset using Naive Bayesian
[Colab notebook](https://colab.research.google.com/drive/1EBXVLeoPRWydzW0cXWg2Oijpp7Ic4xIc)
##### Powerset using Random Forest
[Colab notebook](https://colab.research.google.com/drive/1htwCaWZuCKVlwEeOw_7l1wDxuEOaBn7L)
##### OneVsAll using Linear SVC
[Colab notebook](https://colab.research.google.com/drive/1QtMF8tO9y_DBCwV0uex_81bVxTGyNgt7)
##### OneVsAll using Naive Bayesian
[Colab notebook](https://colab.research.google.com/drive/1wZVsZdIjPGQaSM_nEeoTLuWkAovex3hq)
##### Transfer learning using BERT


### Implementations and solutions references:
###### Multilabel problem:
https://medium.com/towards-artificial-intelligence/understanding-multi-label-classification-model-and-accuracy-metrics-1b2a8e2648ca

###### Basic with preprocessing steps and use deep-learning models
https://medium.com/@armandj.olivares/a-basic-nlp-tutorial-for-news-multiclass-categorization-82afa6d46aa5

###### Text classification using transformer:
https://towardsdatascience.com/https-medium-com-chaturangarajapakshe-text-classification-with-transformer-models-d370944b50ca

###### XLNet & Bert tutorial:
https://mlexplained.com/2019/06/30/paper-dissected-xlnet-generalized-autoregressive-pretraining-for-language-understanding-explained/

###### Random forest:
https://towardsdatascience.com/understanding-random-forest-58381e0602d2
https://www.codementor.io/@agarrahul01/multiclass-classification-using-random-forest-on-scikit-learn-library-hkk4lwawu
https://medium.com/@tenzin_ngodup/simple-text-classification-using-random-forest-fe230be1e857

###### Methods with scikit learn:
https://towardsdatascience.com/multi-label-text-classification-with-scikit-learn-30714b7819c5


###### Linear SVC:
https://www.kaggle.com/xingewang/the-math-behind-linear-svc-classifier
https://www.datacamp.com/community/tutorials/svm-classification-scikit-learn-python
https://scikit-learn.org/stable/modules/generated/sklearn.svm.LinearSVC.html#sklearn.svm.LinearSVC