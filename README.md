# Data Science in Practice: Study of Dynamic Weighted Majority Ensemble Method

We add an ensemble algorithm: Dynamic Weighted Majority, to River.

## Experiment 1
Dataset: Synthetic datase containing concept drift, provide by López Lobo on Harvard Dataverse [1]. The dataset contains data generated with 4 concepts using Sine function.
- Compare 3 methods:
  1. Naive Bayes Classifier
  2. Dynamic Weighted Majority method
  3. Adwin bagging
- Validate our results with Scikit-multiflow library 

## Experiment 2
Dataset: Electricity is described by M. Harries and analysed by Gama. Electricity contains 45,312 instances, with 8 input attributes, recorded every half an hour for two years from Australian New South Wales Electricity. The classification task is to predict a rise (Up) or a fall (Down) in the electricity price. The concept drift may happen because of changes in consumption habits, unexpected events, and seasonality .
- Compare 2 methods:
  1.Naive Bayes Classifier
  2.Dynamic Weighted Majority method

## Experiment 3
Dataset: Electricity dataset.
- Chunk_size : the number of x-y pairs to train a learner in this dwm algorithm
- Compare different Chunk_size

## Experiment 4
Dataset: Electricity dataset.
- Accuracy score is the percentage of exact matches, 
- Kappa score is introduced by Cohen, If the classifier is always correct then kappa = 1. If its predictions are correct as often as those of a chance classifier, then kappa = 0. 
- Compare 3 methods
  1. Naive Bayes Classifier
  2. Dynamic Weighted Majority method
  3. Adwin bagging

## Experiment 5
Covertype dataset with 7 classes: Contains the forest cover type for 30 x 30 meter cells obtained from US Forest Service (USFS) Region 2 Resource Information System (RIS) data. It contains 581, 012 instances and 54 attributes, and it has been used in several papers on data stream classification.
- Compare Naive bayes and Dynamic weighted majority accuracy 

## Conclusion
Our project is in the directory `project`.
DWM can improve accuracy, especially when there is concept drift.
We’ve build docker image for our codings, it can be found on [docker hub](https://hub.docker.com/r/wen1109/data_project).


## Reference
[1] López Lobo, Jesús, 2020, "Synthetic datasets for concept drift detection purposes", https://doi.org/10.7910/DVN/5OWRGB, Harvard Dataverse, V1, UNF:6:VVTBgRNMEV+B/GmoE3Myng== [fileUNF]

