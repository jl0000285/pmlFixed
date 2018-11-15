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
        self.session = repo.get_session()
        
        
    def dbInit(self):
        """
        Create and populate 
        """
        repo.craftSystem()
        self.session = repo.get_session()
        print("Performing data init")
        dbh.data_init(self.session)
        print("Performing Alg Class Init")
        dbh.alg_class_init(self.session)
        print("Perfomring Algorithms init")
        dbh.algorithms_init(self.session)
        print("Performing Runs init")
        dbh.run_active(self.session)

        
    def extInit(self):
        """
        Create and populate extended metabase
        """
        repo.craftSystem()
        self.session = repo.get_session()
        print("Performing ext data init")
        self.ext_data_init()
        print("Performing Algorithms init")
        dbh.alg_class_init(session)
        print("Perfoming ext run init")
        dbh.run_exhaustive(session)
        
        
        
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
            full_set = data.convert_file()
            repo.add_dset(tup[1],
                          tup[0],
                          full_set,
                          len(full_set[0]),
                          self.session)

    def ext_data_all_init(self):
        all_tup = ('DatasetAll','all_data')
        filelist = self.get_allowed_files()
        for tup in filelist:
            print("Adding set {} to all_sets at {}".format(tup[1],tup[0]))
            data = fp.Parser(fp.LC_SINGLE,tup[0],False,.25,',')
            full_set = data.convert_file()
            repo.ext_add_dset(all_tup[0],
                              all_tup[1],
                              tup[1],
                              tup[0]
                              full_set,
                              len(full_set[0]),
                              self.session)
            
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

    def run_exhaustive(self):
        import sk_handler as skh
        d_sets = self.session.query(repo.Dataset).all()
        algs = self.session.query(repo.Algorithm).all()
        for d_set in d_sets:
            print("Analyzing dataset: {}".format(d_set.data_name))
            data_id = d_set.data_id
            data = fp.Parser(fp.COMMA_DL,d_set.data_path,
                             fp.TP_TRUE,per=.25)
            target_input = data.convert_file()
            train_data,test_data = data.write_csv(target_input)
            X_train,y_train = data.split_last_column(train_data)
            X_test,y_test = data.split_last_column(test_data)
            sk = skh.SkHandler(X_train,y_train,X_test,y_test)
            for alg  in algs:
                alg_id = alg.alg_id
                evstring = '{}()'.format(alg.alg_path)
                print(evstring)            
                durr,acc = eval(evstring)
                repo.add_run(data_id,alg_id,durr,acc,session)

    def active_run(self):
        pass

    def print_databases(self):
        cnx = mysql.connector.connect(user='root', password='Welcome07', host='127.0.0.1')
        cursor = cnx.cursor()
        cursor.execute('show databases')
        for i in cursor:
            print (i)
        cnx.close()


    def get_allowed_files(self):
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
                    
                    
