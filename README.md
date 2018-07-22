## Testing the best algorithm for "Text Classification"

This is an experiment of choosing the best algorithm to classify text with python.

### Dataset

The dataset used for this experiment is "Twenty Newsgroups" dataset. The dataset is stored on the folder 'dataset/' inside the root folder. For this only 6 of the 20 newsgroups are chosen:

	`space`, `graphics`, `windows`, `religion`, `motorcycles` and `forsale`

#### UTF-8 incompatibility

Some of the supplied text files had incompatibility with UTF-8. So, they are deleted as the part of preprocessing.

### Requirements

* python 2.7

* python modules:

  * scikit-learn
  * scipy
  * colorama
  * termcolor

### Running the code

	python2.7 main.py

### Experiments

For the experiments, we assume that we like `graphics`, `space` and `religion` newsgroups, and we dislike `windows`, `motorcycles` and `forsale` newsgroups.

For a test size of 20%, we have three different experiments:

#### TFIDF with Naive Bayes
__Results__:

```
             precision    recall  f1-score   support

   dislikes       0.94      0.97      0.96       574
      likes       0.97      0.93      0.95       530

avg / total       0.96      0.95      0.95      1104
```
#### TFIDF with Support Vector Machine
__Results__:

```
             precision    recall  f1-score   support

   dislikes       0.97      0.98      0.97       594
      likes       0.98      0.96      0.97       510

avg / total       0.97      0.97      0.97      1104
```
#### TFIDF with K Neighbours
__Results__:

```
             precision    recall  f1-score   support

   dislikes       0.95      0.93      0.94       558
      likes       0.93      0.95      0.94       546

avg / total       0.94      0.94      0.94      1104
```

### Conclusion

These experiments concludes that TFIDF with Support Vector Machine (SVM) yielded the best results than other algorithms.