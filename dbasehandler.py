# -*- coding: utf-8 -*-
"""
Created on Tue Oct 18 22:44:00 2016

@author: o-4
"""

from __future__ import print_function, division
import dbasehandler as dbh
import mysql.connector
import os
import fparser as fp
import re
import time
import subprocess
from mysql.connector import errorcode
import repo
import pdb
import random
import math

class DbHandler(object):

    def __init__(self):
        """
        Handler object for interfacing with the metabase 
        """
        #Reminder: First item is the name of the generated repo class, second is the table name
        self.baseDataTables = [('BasesetA','base_sets_a'),
                               ('BasesetB','base_sets_b'),
                               ('BasesetC','base_sets_c'),
                               ('BasesetD','base_sets_d'),
                               ('BasesetE','base_sets_e'),
                               ('BasesetF','base_sets_f'),
                               ('BasesetG','base_sets_g'),
                               ('BasesetH','base_sets_h'),
                               ('BasesetI','base_sets_i'),
                               ('BasesetJ','base_sets_j')]
        
        self.testDataTables = [('TestsetA','test_sets_a'),
                               ('TestsetB','test_sets_b'),
                               ('TestsetC','test_sets_c')]
        self.RUN_NOT_FOUND = False 

    def get_session(self):
        self.session = repo.get_session()
        
    def setup_session(self):
        """Get session object then define repo metadata"""        
        self.get_session()
        repo.defineMeta()
        
    def dbInit(self):
        """
        Create and populate extended metabase
        """
        repo.craftSystem()
        self.get_session()        
        print("Populating Data Tables")
        self.populate_data_all()
        self.populate_metabases()
        print("Populating Algorithm Tables")
        self.populate_alg_class()
        self.populate_algorithms()
        print("Perfoming Runs")
        self.populate_runs_all()
        print("Crafting Learning Curves")
        self.populate_learning_curves()
        print("Making Exhaustive Guesses")
        self.guesses_exhaustive()
        print("Making Active Guesses ")
        self.guesses_active()
        print("Making Sampling Guesses")
        self.guesses_sampling()
        print("Compiling Results")
        self.populate_results()
        
        
    def populate_data_from_init_folder(self):
        for dirpath,dirname,filelist in os.walk('./data/init'):
            for filename in filelist:
                if(re.search(r".*[.]data$",filename)):
                    print ("dirpath: {}, dirname: {}, filename: {}"
                           .format(dirpath,dirname,filename))
                    dpath = '{}/{}'.format(dirpath,filename)
                    data = fp.Parser(fp.LC_SINGLE,dpath,False,.25,',')
                    full_set = data.convert_file()
                    pdb.set_trace()
                    repo.add_dset(filename,
                                  dpath,
                                  full_set,
                                  self.session)

    def populate_data_all(self):
        all_dict = {'className': 'DatasetAll',
                    'tableName': 'all_data'}
        filelist = self.get_allowed_files()
        for dpath,filename in filelist:
            print("Adding set {} to all_sets at {}".format(filename,dpath))
            data = fp.Parser(fp.LC_SINGLE,dpath,False,.25,',')
            target_input = data.convert_file()
            target_input = data.limit_size(target_input) #limiting size of datasets for sanity

            try:                
                repo.ext_add_dset(all_dict['className'],
                                  all_dict['tableName'],
                                  filename,
                                  dpath,
                                  target_input,                                  
                                  self.session)
            except Exception as ex:
                print("Exception occured whilst trying to add dataset: {}".format(ex))

            
    def populate_metabases(self):
        filelist = self.get_allowed_files()
        for baseTup in self.baseDataTables:
            base = []
            for i in range(int(math.floor(len(filelist)/5))):
                inx = random.randrange(0,len(filelist)-1)
                base.append(filelist[inx])

            print("Determining base sets for table {}".format(baseTup[0]))
                
            for tup in base:
                print("Adding set {} at {}".format(tup[1],tup[0]))
                data = fp.Parser(fp.LC_SINGLE,tup[0],False,.25,',')
                full_set = data.convert_file()
                repo.ext_add_dset(baseTup[0],
                                  baseTup[1],
                                  tup[1],
                                  tup[0],
                                  full_set,                               
                                  self.session)

    def populate_alg_class(self):
        """Initialize algorithms class table"""
        class_A = repo.AlgClass(class_name='supervised')
        self.session.add(class_A)
        self.session.commit()

        
    def populate_algorithms(self):
        algTypes= {
                  'svm':('sk.svm','supervised'),
                  'clustering':('sk.clustering','supervised'),
                  'neural_network':('sk.neural_network','supervised'),
                  'bayes':('sk.bayes','supervised'),
                  'regression':('sk.regression','supervised')
                  }
        for key in algTypes:
            class_id = self.session.query(repo.AlgClass).\
                filter_by(class_name=algTypes[key][1]).first()
            class_id = class_id.class_id
            repo.add_alg(key,algTypes[key][0],class_id,self.session)

            
    def populate_runs_all(self):
        """Populate runs_all database table with a run of every dataset with every algorithm"""
        import sk_handler as skh
        from random import shuffle
        try:
            d_sets = self.session.query(repo.DatasetAll).all()
        except AttributeError:
            print('Repo metabases likely not defined, defining now')
            repo.defineMeta()
            d_sets = self.session.query(repo.DatasetAll).all()
            
        algs = self.session.query(repo.Algorithm).all()
        for d_set in d_sets:
            print("Analyzing dataset: {}".format(d_set.data_name))            
            data_id = d_set.data_id
            data = fp.Parser(fp.COMMA_DL,d_set.data_path,
                             fp.TP_TRUE,per=.25)
            target_input = data.convert_file()
            shuffle(target_input) #keep commented while debugging
            target_input = data.limit_size(target_input) #limiting size of datasets for sanity 
            train_data,test_data = data.partition(target_input)
            X_train,y_train = data.split_last_column(train_data)
            X_test,y_test = data.split_last_column(test_data)
            sk = skh.SkHandler(X_train,y_train,X_test,y_test)            
            for alg  in algs:
                alg_id = alg.alg_id
                evstring = '{}()'.format(alg.alg_path)
                print(evstring)
                try:                    
                    durr,acc = eval(evstring)
                except Exception as ex:                    
                    print("Could not train dataset {} with method {}: {}".format(d_set.data_path,
                                                                                          alg.alg_path,
                                                                                          ex))
                    durr,acc = [float('inf'),0]
                    pdb.set_trace()
                    
                repo.add_run(data_id,alg_id,durr,acc,self.session)

    def populate_learning_curves(self):
        """Populate learning_curves database table with a curve for every dataset with every algorithm"""
        import sk_handler as skh
        from random import shuffle
        try:
            d_sets = self.session.query(repo.DatasetAll).all()
        except AttributeError:
            print('Repo metabases likely not defined, defining now')
            repo.defineMeta()
            d_sets = self.session.query(repo.DatasetAll).all()
            
        algs = self.session.query(repo.Algorithm).all()
        for d_set in d_sets:
            print("Crafting Learning Curve for  dataset: {}".format(d_set.data_name))            
            data_id = d_set.data_id
            data = fp.Parser(fp.COMMA_DL,d_set.data_path,
                             fp.TP_TRUE,per=.25)
            target_input = data.convert_file()
            shuffle(target_input)
            target_input = data.limit_size(target_input) #limiting size of datasets for sanity
            percents = [0.1,0.2,0.3]
            
            for alg  in algs:
                results = []
                train_time = 0
                alg_id = alg.alg_id
                evstring = '{}()'.format(alg.alg_path)
                for percent in percents:
                       shuffle(target_input)
                       train_data,test_data = data.partition(target_input, per=percent)
                       X_train,y_train = data.split_last_column(train_data)
                       X_test,y_test = data.split_last_column(test_data)	
                       sk = skh.SkHandler(X_train,y_train,X_test,y_test)           
                       print('{} evaluated at {} percent'.format(evstring,str(percent)))
                       try:                    
                           durr,acc = eval(evstring)
                           train_time += durr
                           results.append(acc)
                       except Exception as ex:
                           print("Could not train dataset {} with method {}: {}".format(d_set.data_path,
                                                                                          alg.alg_path,
                                                                                          ex))
                           durr, acc = [float('inf'),0]
                           results.append(acc)
                           
                results.append(train_time)
                repo.add_curve(data_id,alg_id,results,self.session)

    def populate_results(self):
        def calculate_accuracy(guesses):            
            num_correct = guesses.filter_by(correct=0).count()
            num_overall = guesses.count()
            if num_overall > 0:
                acc = num_correct/num_overall
            else:
                acc = 0
            return acc

        def calculate_training_time(guesses,alg):
            time = 0
            if alg == 'GuessesSamp':
               curves = self.session.query(repo.LearningCurve)
               for guess in guesses:
                   g_curves = curves.filter_by(data_id=guess.data_id)
                   for c in g_curves:
                       time += c.train_time
            else:
                sets = self.session.query(repo.DatasetAll)
                for guess in guesses:
                    g_sets = sets.filter_by(data_id=guess.data_id)
                    for s in g_sets:
                        time += s.metric_time
                runs = self.session.query(repo.Run)
                for guess in guesses:
                    g_runs = runs.filter_by(data_id=guess.data_id)
                    for r in g_runs:
                        time += r.train_time                
            return time 
            
        def calculate_rate_correct_score(acc,train_time):
            if not train_time > 0:
                rcs = 0
            else: 
                rcs = acc/train_time
                
            return rcs 
            
        meta_algs = ['GuessesEx', 'GuessesActive', 'GuessesSamp']
        base_sets = [tup[1] for tup in self.baseDataTables]

        for alg in meta_algs:
            class_string = 'repo.{}'.format(alg)
            class_object = eval(class_string)
            for set in base_sets:
                guesses = self.session.query(class_object).filter_by(metabase_table=set)                
                acc = calculate_accuracy(guesses)
                train_time = calculate_training_time(guesses,alg)
                rcs = calculate_rate_correct_score(acc,train_time)
                repo.add_to_results(alg,set,acc,train_time,rcs,self.session)
            
    def get_active_base(self, base_name):
        """
        Steps: 
        1. Obtain metabase candidate datasets
        2. Add half of them to active base for training
        3. Decide on another fourth of them using active learning 
          (by comparing amount of information in datasets)
        4. Return active base
        """        
        def sum_of_distances(metafeature,dinx,candidates):            
            vector = get_feature_vector(metafeature,candidates)            
            dist_summ = sum([abs(x-vector[dinx]) for x in vector])
            return dist_summ 

        def get_feature_vector(metafeature, candidates):
            vector = []
            for dset in candidates:
                value_string = 'dset.{}'.format(metafeature)
                value = eval(value_string)
                vector.append(value)
            return vector 
        
        def spread_without_set(metafeature,dinx,candidates):
            vector = get_feature_vector(metafeature, candidates)
            vector.pop(dinx)

            max_val = max(vector)
            min_val = min(vector)
          
            spread = abs(max_val - min_val)
            return spread

        def calculate_uncertainty_for_feature(metafeature,dinx,candidates):
            try:
                uncertainty = sum_of_distances(metafeature,dinx,candidates)\
                    /spread_without_set(metafeature,dinx,candidates)
            except Exception as ex:
                pdb.set_trace()
                pass            
            return uncertainty
        
        def rank_uncertainty_for_feature(metafeature, candidates):            
            ranked_tuples = []
            for inx,dset in enumerate(candidates):
                uncertainty = calculate_uncertainty_for_feature(metafeature,inx,candidates)
                rank_tuple = [uncertainty,inx,dset]
                ranked_tuples.append(rank_tuple)
            
            ranked_tuples.sort(reverse=True)            
            return ranked_tuples

        def get_most_uncertain_dataset(candidates):
            metafeatures = ['weighted_mean', 'coefficient_variation', 'fpskew', 'kurtosis', 'entropy']
            score_list = [[0, dset] for dset in candidates] # (Score, dataset)
            
            for feature in metafeatures:
                ranked_tuples = rank_uncertainty_for_feature(feature,candidates) #list of (uncertainty,original Index,candidate) where index is rank
                
                for inx,(uncertainty, original_index, candidate) in enumerate(ranked_tuples):
                    score_list[original_index][0] += inx + 1  #Here a lower total score means higher uncertainty
                    
            
            max_inx = 0 #index of set with highest uncertainity i.e set with lowest rank score

            for inx,tup in enumerate(score_list):
                if tup[0] < score_list[max_inx][0]: 
                    max_inx = inx
                
            return max_inx

        class_string = 'repo.{}'.format(base_name)
        class_object = eval(class_string)
        bases = self.session.query(class_object).all()        
        candidates = list(bases)
        active_base = []
            
        for i in range(int(math.floor(len(bases)/2))):
            inx = get_most_uncertain_dataset(candidates)
            cand = candidates.pop(inx)
            active_base.append(cand)            
        return active_base

    
    def compute_objective(self,run):
        """Compute loss function for a given run"""
        return run.accuracy

    
    def find_best_algorithm(self, data_id):
        """
        Fetch the best performing algorithm from the runs_all 
        table for some given dataset 
        """
        runs = self.session.query(repo.Run).filter_by(data_id=data_id).all()
        scores = [[run.alg_id,self.compute_objective(run)] for run in runs]
        values = [tup[-1] for tup in scores]
        try:
            max_inx = values.index(max(values))
            retVal = scores[max_inx][0]
        except ValueError as ex:            
            retVal = self.RUN_NOT_FOUND
            
        return retVal
    
    def guess_with_clusterer(self,base_sets,dataset):
        """
        Use clustering algorithm with given base_sets to guess
        datasets best performing algorithm 
        """
        from sklearn.cluster import KMeans
        import numpy as np
        points = [[set.weighted_mean,set.coefficient_variation,set.fpskew,set.kurtosis,set.entropy]
                   for set in base_sets]
        X = np.array(points)
        #num_clust = len(np.unique(points))
        num_clust = len(points)
        Kmeans = KMeans(n_clusters=num_clust, random_state=0).fit(X)
        test = [dataset.weighted_mean,
                dataset.coefficient_variation,
                dataset.fpskew,
                dataset.kurtosis,
                dataset.entropy]        
        guess_label = Kmeans.predict(np.array(test).reshape(1,-1)) #Returns dataset from base_sets dataset is closest too
        guess_inx = np.where(Kmeans.labels_ == guess_label)[0][0]
        guess = self.find_best_algorithm(base_sets[guess_inx].data_id) #Returns best algorithm for
                                                                   #guess_set, with the assumption that
                                                                   #that would then be the best algorithm for
                                                                   #dataset
        return (guess)

    def guess_with_sampler(self,base_sets,dataset):
        """
        Use learning curve distance comparisons to determine best algorithm 
        """
        def calculate_distance_between_sets(curvesA,curvesB):
            distance = 0.0
            percents = ['10','20','30']
            for i in range(len(curvesA)):
                for percent in percents:
                    a_string = 'curvesA[{}].accuracy_{}'.format(str(i),percent)
                    b_string = 'curvesB[{}].accuracy_{}'.format(str(i),percent)
                    acc_a = eval(a_string)
                    acc_b = eval(b_string)
                    distance += (acc_a - acc_b)**2
                    
            return distance 
                    
        def get_distance_between_sets(curvesA, curvesB):
            distance = 0.0
            algs = self.session.query(repo.Algorithm).all()
            for alg in algs:
                curveA = curvesA.filter_by(alg_id=alg.alg_id).all()
                curveB = curvesA.filter_by(alg_id=alg.alg_id).all()
                distance += calculate_distance_between_sets(curveA,curveB)                
                return distance  
        
        def get_all_set_distances(base_sets,dataset):
            """
            distance items look like [distance, base_set_id]
            """
            distances = []
            dset_curves = self.session.query(repo.LearningCurve).filter_by(data_id=dataset.data_id)
            for set in base_sets:
                set_curves = self.session.query(repo.LearningCurve).filter_by(data_id=set.data_id)
                distances.append([get_distance_between_sets(dset_curves,set_curves), set.data_id])
            return distances
                
        set_distances = get_all_set_distances(base_sets,dataset)
        set_distances.sort(reverse=True)
        guess = self.find_best_algorithm(set_distances[0][1])
        return guess
           
    def guesses_exhaustive(self):
        """Given a set of databases, make guesses as to what would be the best 
        machine based off entirety of current metabase 
        """
        guess_class, guess_table  = ('GuessesEx', 'guesses_ex')
        datasets = self.session.query(repo.DatasetAll).all()
        
        for className, tableName in self.baseDataTables:
            class_string = 'repo.{}'.format(className)
            class_object = eval(class_string)
            curr_base = self.session.query(class_object).all()
            base_names = [set.data_name for set in curr_base]   
            for dataset in datasets:
                if dataset.data_name not in base_names:
                    guess = self.guess_with_clusterer(curr_base,dataset)
                    """
                    #need to modify various declartive bases such that data_id is a key
                    #that exists only within datasets_all and so that the various base
                    #set classes store that key on them selves, changing the column called
                    data_id in the base set classes to set_id. 
                    """
                    solution = self.find_best_algorithm(dataset.data_id)
                    repo.add_to_guesses(tableName,
                                        guess_class,
                                        dataset.data_id,
                                        dataset.data_name,
                                        guess,
                                        solution,
                                        self.session)
                    
        

    def guesses_active(self):
        """Given a set of databases, make guesses as to what would be the best 
        machine based off the uncertainty values of the datasets contained within 
        """
        guess_class, guess_table = ('GuessesActive','guesses_active')
        datasets = self.session.query(repo.DatasetAll).all()
        
        for className, tableName in self.baseDataTables:            
            active_base = self.get_active_base(className)            
            base_names = [set.data_name for set in active_base]            
            for dataset in datasets:
                if dataset.data_name not in base_names:
                    guess = self.guess_with_clusterer(active_base,dataset)
                    """
                    #need to modify various declartive bases such that data_id is a key
                    #that exists only within datasets_all and so that the various base
                    #set classes store that key on them selves, changing the column called
                    data_id in the base set classes to set_id. 
                    """
                    solution = self.find_best_algorithm(dataset.data_id)
                    repo.add_to_guesses(tableName,
                                        guess_class,
                                        dataset.data_id,
                                        dataset.data_name,
                                        guess,
                                        solution,
                                        self.session)
                    
        
    def guesses_sampling(self):
        """Given a set of databases, make guesses as to what would be the best
        machine based off the sampling curves of the datasets contained within
        """
        guess_class, guess_table = ('GuessesSamp', 'guesses_samp')
        datasets = self.session.query(repo.DatasetAll).all()

        for className, tableName in self.baseDataTables:
            class_string = 'repo.{}'.format(className)
            class_object = eval(class_string)
            curr_base = self.session.query(class_object).all()
            base_names = [set.data_name for set in curr_base]
            for dataset in datasets:
                if dataset.data_name not in base_names:
                    guess = self.guess_with_sampler(curr_base,dataset)
                    solution = self.find_best_algorithm(dataset.data_id)
                    repo.add_to_guesses(tableName,
                                        guess_class,
                                        dataset.data_id,
                                        dataset.data_name,
                                        guess,
                                        solution,
                                        self.session)
        
    def print_databases(self):
        cnx = mysql.connector.connect(user='root', password='Welcome07', host='127.0.0.1')
        cursor = cnx.cursor()
        cursor.execute('show databases')
        for i in cursor:
            print (i)
        cnx.close()


    def get_allowed_files(self):
        """Return list of allowed file tuples where elem one is path and 
        elem two is name of the file
        """
        f = []
        allowed_types = ['data', 'svm', 'dat']
        for dirpath,dirname,filelist in os.walk('./data/datasets'):
            for filename in filelist:
                for t in allowed_types:
                    pat = '.*[.]{}$'.format(t)
                    if(re.search(pat,filename)):
                       #print ("dirpath: {}, dirname: {}, filename: {}"
                       #       .format(dirpath,dirname,filename))
                        dpath = '{}/{}'.format(dirpath,filename)
                        tup = (dpath,filename)
                        f.append(tup)
        return f 
                    
                    
