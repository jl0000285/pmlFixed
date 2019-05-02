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
        def mcall(self):
            try:
                duration, acc = meth(self)
                return duration,acc
            except Exception as ex:
                pdb.set_trace()
                meth(self)
                pass
        mcall.__name__ = meth.__name__
        return mcall

    @catcher    
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

    @catcher
    def clustering(self):
        from sklearn.cluster import KMeans
        import math
        
        def determine_if_floats(column):
            are_floats = False
            for item in column:
                decimal, number = math.modf(item)
                if decimal != 0:
                    are_floats = True
            return are_floats 

        def relabel_with_bins(train, test):
            all = train + test
            bin_size = abs(min(all)-max(all))/10
            for inx, item in enumerate(train):
                train[inx] = math.floor(item/bin_size)
            for inx, item in enumerate(test):
                test[inx] = math.floor(item/bin_size)

            return train, test
            
        X = np.array(self.X_train)
        
        if not determine_if_floats(self.y_train):
            num_clust = len(np.unique(self.y_train))
        else:
            num_clust = 10
            self.y_train, self.y_test = relabel_with_bins(self.y_train,self.y_test)
            
        duration = 0
        t0 = time.time()
        Kmeans = KMeans(n_clusters=num_clust, random_state=0).fit(X)
        t1 = time.time()
        labs = list(Kmeans.labels_)
        orig_labs = labs
        duration = t1-t0
        acc = 0        
        clust_labs = dict()
        trans_map = dict() #translation map between cluster labels and original labels 
        for item in np.unique(labs):
            clust_labs[item]=dict()
        for key in clust_labs:
            for item in np.unique(self.y_train):
                clust_labs[key][item] = 0
        for inx, item in enumerate(self.y_train):
            lab = labs[inx]
            clust_labs[lab][item] += 1
        for key in clust_labs:
            max_count = 0
            max_key = 0
            for key2 in clust_labs[key]:
                count = clust_labs[key][key2]
                if(count > max_count):
                    max_count = count
                    max_key = key2
            trans_map[key]=max_key
        #Not actually needed here, keeping this loop incase I
        #want to compute inner class accuracy later
        for inx,item in enumerate(labs):
            labs[inx] = trans_map[item]
        #inner_acc = accuracy_score(labs,self.y_train)
        test_labs = Kmeans.predict(self.X_test)
        for inx,item in enumerate(test_labs):
            test_labs[inx] = trans_map[item] 
        acc = accuracy_score(test_labs,self.y_test)        
        return duration, acc

    @catcher
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

    @catcher
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

    @catcher
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


