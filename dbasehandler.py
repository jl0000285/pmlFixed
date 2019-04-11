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
        self.baseDataTables = [('DatasetA','base_sets_a'),
                               ('DatasetB','base_sets_b'),
                               ('DatasetC','base_sets_c')]
        self.testDataTables = [('TestsetA','test_sets_a'),
                               ('TestsetB','test_sets_b'),
                               ('TestsetC','test_sets_c')]

    def get_session(self):
        """Get repo session"""
        self.session = repo.get_session()
        
        
    def dbInit(self):
        """
        Create and populate 
        """
        repo.craftSystem()
        self.session = repo.get_session()        
        print("Performing data init")
        self.data_init()
        print("Performing Alg Class Init")
        self.alg_class_init()
        print("Perfomring Algorithms init")
        self.algorithms_init()
     
    def extInit(self):
        """
        Create and populate extended metabase
        """
        repo.craftSystem()
        self.get_session()        
        print("Performing ext data init")
        self.ext_data_all_init()
        print("Performing Algorithms init")
        self.alg_class_init()
        self.algorithms_init()
        print("Perfoming ext run init")
        self.runs_all()
        #print("Performing Runs init")
        #self.run_active()
        
        
    def data_init(self):
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

    def ext_data_all_init(self):
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

            
    def ext_data_init(self):
        filelist = self.get_allowed_files()
        for baseTup in self.baseDataTables:
            base = []
            for i in range(int(math.floor(len(filelist)/10))):
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

    def alg_class_init(self):
        """Initialize algorithms class table"""
        class_A = repo.AlgClass(class_name='supervised')
        self.session.add(class_A)
        self.session.commit()

        
    def algorithms_init(self):
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

            
    def runs_all(self):
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
        bases = self.session.query(base_name)
        pdb.set_trace()
        candidates  = bases[:-(math.floor(len(bases/2)))]
        active_base = []
        for i in range(math.floor(len(bases)/2)):
            active_base.append(bases[i])
        #Add the candidates with the most information to the active base
        pdb.set_trace()
        for i in range(math.floor(len(candidates)/2)):
            max_inx = 0
            for i,candidate in enumerate(candidates):
                if candidate.information > candidates[max_inx].information:
                    max_inx = i
                max = candidates.pop(max_inx)
            active_base.append(max)
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
        machine based off 2/3rds of the metabase, with the first 4/5ths of the 
        active base choosen randomly and the last fifth choosen actively 
        """
        guess_tup = ('GuessesActive','guesses_active')
        for tup in self.baseDataTables:
            active_base = self.get_active_base(tup[0])
            base_names = [set.name for set in active_base]
            datasets = self.session.query('datasets_all').all()
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
                    
                    
