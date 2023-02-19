import copy as cp
import numpy as np
import river.naive_bayes

import warnings

""" Dynamic Weighted Majority ensemble classifier.
    Authors
    ----------
    Wen Yang, Yixin Huang
    Parameters
    ----------
    n_estimators: int (default=5)
        Maximum number of estimators to hold.
    base_estimator: river.naive_bayes.GaussianNB()
        Each member of the ensemble is an instance of the base estimator.
    period: int (default=50)
        Period between expert removal, creation, and weight update.
    beta: float (default=0.5)
        Factor for which to decrease weights by.
    theta: float (default=0.01)
        Minimum fraction of weight per model.
    Notes
    -----
    The dynamic weighted majority (DWM) [1]_, uses four mechanisms to
    cope with concept drift: It trains online learners of the ensemble,
    it weights those learners based on their performance, it removes them,
    also based on their performance, and it adds new experts based on the
    global performance of the ensemble.
    References
    ----------
    .. [1] Kolter and Maloof. Dynamic weighted majority: An ensemble method
       for drifting concepts. The Journal of Machine Learning Research,
       8:2755-2790, December 2007. ISSN 1532-4435.
    Examples
    --------
    >>> # Imports
    >>> import dwm
    >>> import pandas as pd
    >>> from river.stream import iter_pandas
    >>> from river import datasets
    >>> import river
    >>> import pandas as pd
    >>> from river.stream import iter_pandas
    >>> from river import datasets
    >>> from river import metrics
    >>> from river import utils
    >>> from river.utils import dict2numpy
    >>> # Setup a data stream
    >>> dataset2 = "mixed_0101_gradual"
    >>> df2 = pd.read_csv(dataset2+".csv")
    >>> label_col = df2.columns[-1]
    >>> feature_cols = list(df2.columns)
    >>> feature_cols.pop()
    >>> X = df2[feature_cols]
    >>> Y = df2[label_col].astype('int')
    >>> stream=iter_pandas(X=X, y=Y)
    >>>     

    >>>
    >>> # Setup Dynamic Weighted Majority Ensemble Classifier
    >>> test = dwm.DynamicWeightedMajorityClassifier_river(n_estimators=5, period= 500)
    >>>
    >>> # Setup variables to control loop and track performance
    >>> n_samples = 0
    >>> correct_cnt = 0
    >>> max_samples = df2.shape[0]
    >>>
    >>> # Train the classifier with the samples provided by the data stream
    >>> for x, y in stream:
    >>>     test.learn_one(x, y)
    >>>     y_pred = test.predict_one(x)
    >>>     
    >>>    if y == y_pred:
    >>>         correct_cnt += 1
    >>>         
    >>>    n_samples += 1
    >>> 
    >>>     if n_samples ==  max_samples:
    >>>        break
    >>> # Display results       
    >>> print('{} samples analyzed.'.format(n_samples))
    >>> print('DynamicWeightedMajority with Naive Bayes error rate: {}'.format(1-correct_cnt / n_samples))

    """


class DynamicWeightedMajorityClassifier_river(): 

    class WeightedExpert:
        """
        Wrapper that includes an estimator and its weight.
        Parameters
        ----------
        estimator: StreamModel or sklearn.BaseEstimator
            The estimator to wrap.
        weight: float
            The estimator's weight.
        """
        def __init__(self, estimator, weight):
            self.estimator = estimator
            self.weight = weight

    def __init__(self, n_estimators=5, base_estimator=river.naive_bayes.GaussianNB(),
                 period=50, beta=0.5, theta=0.01):
        """
        Creates a new instance of DynamicWeightedMajorityClassifier.
        """
        super().__init__()

        self.n_estimators = n_estimators
        self.base_estimator = base_estimator

        self.beta = beta
        self.theta = theta
        self.period = period

        # Following attributes are set later
        self.epochs = 0
        self.num_classes = 2
        self.experts =[
            self.create_new_expert()
        ]


    def predict_one(self, X):
        """ predict
        The predict function will take an average of the predictions of its
        learners, weighted by their respective weights, and return the most
        likely class.
        Parameters
        ----------
        X: numpy.ndarray of shape (n_samples, n_features)
            A matrix of the samples we want to predict.
        Returns
        -------
        numpy.ndarray
            A numpy.ndarray with the label prediction for all the samples in X.
        """
        
      
        preds = np.array([np.array(exp.estimator.predict_one(X)) * exp.weight
                          for exp in self.experts])
       
        sum_weights = sum(exp.weight for exp in self.experts)
        aggregate = np.sum(preds / sum_weights, axis=0)
        return (aggregate + 0.5).astype(int)    # Round to nearest int


    def learn_one(self, X, y, classes=None, sample_weight=None):
        """ Fits the model on the supplied X and y matrices.
        Since it's an ensemble learner, if X and y matrix of more than one
        sample are passed, the algorithm will partial fit the model one sample
        at a time.
        Parameters
        ----------
        X : numpy.ndarray of shape (n_samples, n_features)
            The features to train the model.
        y: numpy.ndarray of shape (n_samples)
            An array-like with the class labels of all samples in X.
        classes: numpy.ndarray, optional (default=None)
            Array with all possible/known class labels. This is an optional parameter, except
            for the first partial_fit call where it is compulsory.
        sample_weight: numpy.ndarray of shape (n_samples), optional (default=None)
            Samples weight. If not provided, uniform weights are assumed. Usage varies depending on the base estimator.
        Returns
        -------
        DynamicWeightedMajorityClassifier
            self
        """
        self.epochs += 1
        self.num_classes = max(
            len(classes) if classes is not None else 0,
            (int(np.max(y)) + 1), self.num_classes)
        predictions = np.zeros((self.num_classes,))
        max_weight = 0
        weakest_expert_weight = 1
        weakest_expert_index = None

        for i, exp in enumerate(self.experts):
            y_hat = exp.estimator.predict_one(X)
            if np.any(y_hat != y) and (self.epochs % self.period == 0):
                exp.weight *= self.beta

            predictions[y_hat] += exp.weight
            max_weight = max(max_weight, exp.weight)

            if exp.weight < weakest_expert_weight:
                weakest_expert_index = i
                weakest_expert_weight = exp.weight

        y_hat = np.array([np.argmax(predictions)])
        if self.epochs % self.period == 0:
            self.Normalize_Weights(max_weight)
            self.remove_expert()
            if np.any(y_hat != y):
                if len(self.experts) == self.n_estimators:
                    self.experts.pop(weakest_expert_index)
                self.experts.append(self.create_new_expert())

        # Train individual experts
        for exp in self.experts:
            exp.estimator.learn_one(X, y)

    def Normalize_Weights(self, max_weight):
        """
        Calculate the experts' normalized weights such that the max is 1.
        """
        scale_factor = 1 / max_weight
        for exp in self.experts:
            exp.weight *= scale_factor

    def remove_expert(self):
        """
        Removes all experts whose weight is lower than self.theta.
        """
       
        self.experts = [ex for ex in self.experts if ex.weight >= self.theta]

    def create_new_expert(self):
        """
        Constructs a new WeightedExpert from the provided base_estimator.
        """
        return self.WeightedExpert(cp.deepcopy(self.base_estimator), 1)