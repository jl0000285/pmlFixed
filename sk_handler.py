import time
import random
from sklearn.metrics import accuracy_score
import pdb
import numpy as np


class SkHandler(object):
    """Class to help handle sk-learn methods
    SVM, Clusterer, Regression, Neural Networks, Bayesian
    """
    def __init__(self,X_train,y_train,X_test,y_test):
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test

    
    def catcher(meth):
       # pdb.set_trace()
        def mcall(self):
            try:
                meth(self)
            except Exception as ex:
                pdb.set_trace()
                pass
        mcall.__name__ = meth.__name__
        return mcall

    #@catcher    
    def svm(self):
        """
        X = Data
        y = Labels
        per = percentage of data to withhold for testing (remainder will be used to train)
        """
        from sklearn.svm import SVC
        clf = SVC()
        #TODO: Explore possibility of using timeit as stopwatch
        t0 = time.time()
        clf.fit(self.X_train, self.y_train)
        t1 = time.time()
        duration = t1-t0
        result = list(clf.predict(self.X_test))
        acc = accuracy_score(result,self.y_test)
        #TODO: measure the accuracy of trained clf with test data here
        return duration, acc

   #@catcher
    def clustering(self):
        from sklearn.cluster import KMeans
        X = np.array(self.X_train)
        num_clust = len(np.unique(self.y_train))
        duration = 0
        t0 = time.time()
        Kmeans = KMeans(n_clusters=num_clust, random_state=0).fit(X)
        t1 = time.time()
        labs = list(Kmeans.labels_)
        orig_labs = labs
        duration = t1-t0
        acc = 0
        nums = dict()
        nlabs = dict()
        for i in range(len(np.unique(self.y_train))):
            nums[i]=dict()
        for key in nums:
            for i in range(len(np.unique(self.y_train))):
                nums[key][i] = 0
        for inx, item in enumerate(self.y_train):
            lab = labs[inx]
            nums[item][lab] += 1
        for key in nums:
            max_count = 0
            max_inx = 0
            for key2 in nums[key]:
                count = nums[key][key2]
                if(count > max_count):
                    max_count = count
                    max_inx = key
            nlabs[key]=max_inx
        #Not actually needed here, keeping this loop incase I
        #want to compute inner class accuracy later
        for inx,item in enumerate(labs):
            labs[inx] = nlabs[item]
        test_labs = Kmeans.predict(self.X_test)
        for inx,item in enumerate(test_labs):
            test_labs[inx] = nlabs[item]
        acc = accuracy_score(test_labs,self.y_test)
        return duration, acc

    #@catcher
    def neural_network(self):
        from sklearn.neural_network import MLPClassifier
        X = np.array(self.X_train)
        y = np.array(self.y_train)
        clf = MLPClassifier(solver='lbfgs', alpha=1e-5,
                       hidden_layer_sizes=(5, 2), random_state=1)
        t0 = time.time()
        clf.fit(self.X_train, self.y_train)
        t1 = time.time()
        result = list(clf.predict(self.X_test))
        duration = t1-t0
        acc = accuracy_score(result,self.y_test)
        return duration, acc

    #@catcher
    def bayes(self):
        from sklearn.naive_bayes import GaussianNB
        X = np.array(self.X_train)
        y = np.array(self.y_train)
        gnb = GaussianNB()
        t0 = time.time()
        gnb.fit(X,y)
        t1 = time.time()
        result = list(gnb.predict(self.X_test))
        acc = accuracy_score(result,self.y_test)
        duration = t1-t0
        return duration, acc

    #@catcher
    def regression(self):
        from sklearn import linear_model
        regr = linear_model.LinearRegression()
        t0 = time.time()
        regr.fit(self.X_train,self.y_train)
        t1 = time.time()
        duration = t1-t0
        result = list(regr.predict(self.X_test))
        result_ar = np.array(result)
        result_ar = result_ar + 0.5
        acc = accuracy_score(np.round(result_ar),self.y_test)
        return duration, acc


