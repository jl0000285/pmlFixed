# -*- coding: utf-8 -*-
"""
Created on Tue Oct 18 22:44:00 2016

@author: o-4
"""

from __future__ import print_function
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
                               ('BasesetC','base_sets_c')]
        self.testDataTables = [('TestsetA','test_sets_a'),
                               ('TestsetB','test_sets_b'),
                               ('TestsetC','test_sets_c')]

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
        print("Populating data all table")
        self.populate_data_all()
        self.populate_metabases()
        print("Performing Algorithms init")
        self.populate_alg_class()
        self.populate_algorithms()
        print("Perfoming ext run init")
        self.populate_runs_all()
        #print("Performing Runs init")
        #self.run_active()
        
        
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
                    repo.add_run(data_id,alg_id,durr,acc,self.session)
                except Exception as ex:                    
                    print("Could not train dataset {} with method {}: {}".format(d_set.data_path,
                                                                                          alg.alg_path,
                                                                                          ex))

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
        pdb.set_trace()
        runs = self.session.query(repo.Run).filter_by(data_id=data_id).all()
        scores = [[run.alg_id,self.compute_objective(run)] for run in runs]
        values = [tup[-1] for tup in scores]
        max_inx = values.index(max(values))        
        return scores[max_inx]
    
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
        
           
    def guesses_exhaustive(self):
        """Given a set of databases, make guesses as to what would be the best 
        machine based off entirety of current metabase 
        """
        className  = 'GuessesEx'
        tableName = 'guesses_ex'
        for tup in self.baseDataTables:
            curr_base = self.session.query(tup[0])
            base_names = [set.name for set in curr_base]
            datasets = self.session.query('datasets_all').all()
            for dataset in datasets:
                if dataset.name not in base_names:
                    guess = self.guess_with_clusterer(curr_base,dataset)
                    """
                    #need to modify various declartive bases such that data_id is a key
                    #that exists only within datasets_all and so that the various base
                    #set classes store that key on them selves, changing the column called
                    data_id in the base set classes to set_id. 
                    """
                    solution = self.find_best_algorithm(dataset.data_id)
                    repo.add_to_guesses(className,
                                        tableName,
                                        dataset.data_id,
                                        dataset.data_name,
                                        guess,solution)
        

    def guesses_active(self):
        """Given a set of databases, make guesses as to what would be the best 
        machine based off the uncertainty values of the datasets contained within 
        """
        guess_tup = ('GuessesActive','guesses_active')
        datasets = self.session.query(repo.DatasetAll).all()
        
        for className, tableName in self.baseDataTables:            
            active_base = self.get_active_base(className)
            pdb.set_trace()
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
                    repo.add_to_guesses(guess_tup,dataset.data_id,guess[0],solution[0])
                    
        
    def guesses_sampling(self):
        """Take half a given metabase, then determine the label of best performer
        via sampling and matching the learning curves of the later datasets 
        """
        pass
        
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
                    
                    
