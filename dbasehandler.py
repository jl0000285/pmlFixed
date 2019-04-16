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

    def setup_session(self):
        """Get session object then define repo metadata"""        
        self.session = repo.get_session()
        repo.defineMeta()
        
    #@deprecated in its current form    
    def dbInit(self):
        """
        Create and populate 
        """
        repo.craftSystem()
        self.session = repo.get_session()        
        print("Populating data init table")
        self.populate_data_from_init_folder()
        print("Populating alg class table")
        self.populate_alg_class()
        print("Populating algorithms table")
        self.populate_algorithms()
     
    def extInit(self):
        """
        Create and populate extended metabase
        """
        repo.craftSystem()
        self.setup_session()        
        print("Populating data all table")
        self.populate_data_all()
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
                                  len(full_set[0]),
                                  self.session)

    def parse_all(self):
        """
        Test method to confirm that all the data is parasable
        """
        filelist = self.get_allowed_files()
        print("Testing files to confirm parsability")
        for tup in filelist:
            print("Adding set {} at {}".format(tup[1],tup[0]))
            data = fp.Parser(fp.LC_SINGLE,tup[0],False,.25,',')
            target_input = data.convert_file()
            target_input = data.limit_size(target_input)
            repo.add_dset(tup[1],
                          tup[0],
                          target_input,
                          len(target_input[0]),
                          self.session)

    def populate_data_all(self):
        all_tup = ('DatasetAll','all_data')
        filelist = self.get_allowed_files()
        for tup in filelist:
            print("Adding set {} to all_sets at {}".format(tup[1],tup[0]))
            data = fp.Parser(fp.LC_SINGLE,tup[0],False,.25,',')
            target_input = data.convert_file()
            target_input = data.limit_size(target_input) #limiting size of datasets for sanity

            try:
                repo.ext_add_dset(all_tup[0],
                                  all_tup[1],
                                  tup[1],
                                  tup[0],
                                  target_input,
                                  len(target_input[0]),
                                  self.session)
            except Exception as ex:
                print("Exception occured whilst trying to add dataset: {}".format(ex))

            
    def populate_metabases(self):
        filelist = self.get_allowed_files()
        for baseTup in self.baseDataTables:
            base = []
            for i in range(int(math.floor(len(filelist)/20))):
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
                                  len(full_set[0]),
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
        def make_score_list(candidates):
            score_list = []
            for dset in candidates:
                score_tup = [0, dset]
                score_list.append(score_tup)
            return score_list
            
        def distance_between_datasets(metafeature,datasetA,datasetB):
            eval_a = '{}.{}'.format('datasetA',metafeature)
            eval_b = '{}.{}'.format('datasetB',metafeature)
            distance = abs(eval(eval_a) - eval(eval_b))
            return distance

        def sum_of_distances(metafeature,dataset,candidates):
            dist_summ = 0
            for dset in candidates:
                dist_summ += distance_between_datasets(metafeature,dataset, dset)
            return dist_summ 

        def spread_without_set(metafeature,dinx,candidates):
            max_val = 0 # (Value, index)
            min_val = float('inf')
            candidates.pop(dinx)
            for inx, dset in enumerate(candidates):
                value_string = 'dset.{}'.format(metafeature)
                value = eval(value_string)
                if min_val > value:
                    min_val = value
                if max_val  < value:
                    max_val = value
            spread = abs(max_val - min_val)
            return spread

        def calculate_uncertainty_for_feature(metafeature,dinx,candidates):
            uncertainty = sum_of_distances(metafeature,candidates[dinx],candidates)\
                          /spread_without_set(metafeature,dinx,candidates)
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
            metafeatures = ['weighted_mean', 'standard_deviation', 'fpskew', 'kurtosis']
            score_list = make_score_list(candidates) # (Score, dataset)
            
            for feature in metafeatures:
                ranked_tuples = rank_uncertainty_for_feature(feature,candidates) #list of (uncertainty,original Index,candidate) where index is rank
                for inx,tup in enumerate(ranked_tuples):
                    score_list[tup[1]] += inx  #Here a lower total score means higher uncertainty

            max_inx = 0 #index of set with highest uncertainity i.e set with lowest rank score

            for inx,tup in enumerate(score_list):
                if tup[0] < score_list[max_inx][0]: 
                    max_inx = inx
                
            return max_inx

        class_string = 'repo.{}'.format(base_name)
        class_object = eval(class_string)
        bases = self.session.query(class_object).all()        
        candidates = bases
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
        runs = self.session.query('run_all').filter_by(data_id=data_id)
        scores = [[data_id,self.compute_objective(run)] for run in runs]
        max_inx = values.index(max([tup[-1] for tup in scores]))
        pdb.set_trace()
        return scores[max_inx]
    
    def guess_with_clusterer(self,base_sets,dataset):
        """
        Use clustering algorithm with give base_sets to guess
        datasets best performing algorithm 
        """
        from sklearn.cluster import Kmeans
        import numpy as np
        points = [[set.weighted_mean,set.standard_deviation,set.fpskew,set.kurtosis,set.information]
                   for set in base_sets]
        X = np.array(points)
        num_clust = len(np.unique(points))
        Kmeans = Kmeans(n_clusters=num_clust, random_state=0).fit(X)
        test = [dataset.weighted_mean,
                dataset.standard_deviation,
                dataset.fpskew,
                dataset.kurtosis,
                dataset.information]
        guess_set = Kmeans.predict(test)  #Returns dataset from base_sets dataset is closest too
        guess = self.find_best_algorithm(base_sets[guess].data_id) #Returns best algorithm for
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
                    guess = guess_with_clusterer(curr_base,dataset)
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
        
        for tup in self.baseDataTables:
            active_base = self.get_active_base(tup[0])
            base_names = [set.name for set in active_base]            
            for dataset in datasets:
                if dataset.name not in base_names:
                    guess = guess_with_clusterer(active_base,dataset)
                    """
                    #need to modify various declartive bases such that data_id is a key
                    #that exists only within datasets_all and so that the various base
                    #set classes store that key on them selves, changing the column called
                    data_id in the base set classes to set_id. 
                    """
                    solution = self.find_best_algorithm(dataset.data_id)
                    repo.add_to_guesses(guess_tup,dataset.data_id,guess,solution)
                    
        
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
                    
                    
