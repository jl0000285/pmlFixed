# -*- coding: utf-8 -*-
"""
Created on Tue Oct 18 22:44:00 2016

@author: o-4
"""

from __future__ import print_function
import dbasehandler as dbh
import mysql.connector
import metaCollector as mc
import os
import fparser as fp
import re
import time
import subprocess
from mysql.connector import errorcode
import repo


def main():
     #cnx = mysql.connector.connect(user='root', password='Welcome07')
     pass

def dbInit(passw='Welcome07'):
    for dirpath,dirname,filename in os.walk('./data/init'):
        if(re.search(r".*[.]data",filename)):
            dnames=re.split("[.]",filename)
            dpath = dirpath+filename
            dsetA, dsetB = fp.parse_file(fp.LC_SVM_B,dpath, fp.TP_TRUE, .25, fp.COMMA_DL)
            dbh.add_dset(filename, dsetA, len(dsetA[0]),passw)
            for dirpath2,dirname2,filename2 in os.walk('./seq'):
                if(re.search(r".*train",filename2)):
                    try:
                        #split classifier leximes
                        clfnames=re.split("[-]",filename2)
                        pwd = os.getcwd()
                        os.chdir(dirpath2+'/models')
                        #Training time of the initial classisfication
                        t0 = time.clock()
                        subprocess.call(["./" + filename2 + " " + dpath ])
                        ptime = time.clock()-t0
                        predictor = "../{0}-predict".format(clfnames[0])
                        testfile = dirpath + dnames[0] + ".test"
                        modelfile = filename + ".model"
                        outputfile = filename + ".output"
                        results=subprocess.check_call("{0} {1} {2} {3}".format(predictor, testfile, modelfile, outputfile), shell=True)
                        pattern=r"(.*[aA]ccuracy\s*=\s*)(\d*[.]\d*)"
                        x=re.split(pattern,results)
                        os.chdir(pwd)
                    except Exception:
                        pass

def craftSystem(user='', password='', metadata=repo.metadata):
    #cnx = mysql.connector.connect(user=user, password=password)
    #cursor = cnx.cursor()
    DB_NAME = 'METABASE'
    dbh.create_database(metadata, DB_NAME)
    TABLES={}

    TABLES['data_sets']=(
    "CREATE TABLE 'data_sets'("
    " 'name' varchar(16) NOT NULL,"
    " 'data_id' int(11) NOT NULL,"
    " 'weighted_mean' float(11),"
    " 'standard_deviation' float(11),"
    " 'fpskew' float(11),"
    " 'kurtosis' float(11),"
    " 'information' float(11),"
    " PRIMARY KEY ('data_id')"
     ") ENGINE=InnoDB")

    TABLES['alg_class']=(
    "CREATE TABLE 'alg_class'("
    " 'name' varchar(16) NOT NULL,"
    " 'class_id' int(11) NOT NULL,"
    " PRIMARY KEY ('class_id')"
    ") ENGINE=InnoDB")


    TABLES['algorithms']=(
    "CREATE TABLE 'algorithms'("
    "alg_name varchar(16) NOT NULL"
    "alg_id int(11) NOT NULL"
    "class_id int(11) NOT NULL"
    "PRIMARY KEY ('alg_id')"
    "FOREIGN KEY (class_id)"
    "    REFERENCES alg_class(class_id)"
    ") ENGINE=InnoDB")

    TABLES['runs']=(
    "CREATE TABLE 'runs'("
    "data_id int(11) NOT NULL"
    "alg_id int(11) NOT NULL"
    "run_id int(11) NOT NULL"
    "train_time float(11)"
    "accuracy float(11)"
    "PRIMARY KEY ('run_id')"
    "FOREIGN KEY (data_id)"
    "    REFERENCES data_sets(data_id)"
    "FOREIGN KEY (alg_id)"
    "    REFERENCES algorithms(alg_id)"
    ") ENGINE=InnoDB")

    create_tables(cursor, TABLES)
    cursor.close()
    cnx.close()

def connect(passw):
     cnx = mysql.connector.connect(user='root', password=passw)
     cursor = cnx.cursor()
     cursor.execute('use metabase')
     return cnx, cursor

# Adds record to the "runs" table"alg_id int(11) NOT NULL"
def add_run():
    pass

# Adds record to the "data set" table
def add_dset(dname, dset, nc, passw):
    dwmean,ds_dev,dpskew,dkurt = mc.extractFeatures(dset,nc)
    minfo = 0
    dw,ds,dp,dk = dwmean[0], ds_dev[0], dpskew[0],dkurt[0]
    cnx, cursor = dbh.connect(passw)
    add_drec = ("INSERT INTO data_sets "
               "(name, weighted_mean, standard_deviation, fpskew, kurtosis, information)"
               "VALUES (%(name)s, %(weighted_mean)s, %(standard_deviation)s, %(fpskew)s, %(kurtosis)s, %(information)s)")
    data_dset = {
        'name' : dname,
        'weighted_mean' : float(dw),
        'standard_deviation' : float(ds),
        'fpskew' : float(dp),
        'kurtosis' : float(dk),
        'information' : float(minfo),
    }
    cursor.execute(add_drec, data_dset)
    cnx.commit()
    cursor.close()
    cnx.close()

# Populates the algorithms table
def pop_alg():
    algs=[]
    cnx, cursor = dbasehandler.connect()
    for root, dirs in os.walk('/seq'):
        algs = algs+dirs


    add_libsvm=("INSERT INTO algorithms"
               "(alg_name, alg_id, class_id)"
               "VALUES (%(alg_name)s, %(alg_id)s, %(class_id)s)")



# Adds record to the "algorithms" class
def add_algClass():
    pass

def create_database(metadata,DB_NAME):
    import create_db as cdb
    cdb.setup_module()
    cdb.create_tables()
    '''
    try:
        cursor.execute(
            "CREATE DATABASE {} DEFAULT CHARACTER SET 'utf8'".format(DB_NAME))
    except mysql.connector.Error as err:
        print("Failed creating database: {}".format(err))
    '''

def swap_database(cnx,cursor,DB_NAME):
    try:
        cnx.database = DB_NAME
    except mysql.connector.Error as err:
            if err.errno == errorcode.ER_BAD_DB_ERROR:
               create_database(cursor)
               cnx.database = DB_NAME
            else:
               print(err)

def create_tables(cursor, TABLES):
    for name, ddl in TABLES.iteritems():
        try:
            print("Creating table {}: ".format(name), end='')
            cursor.execute(ddl)
        except mysql.connector.Error as err:
            if err.errno == errorcode.ER_TABLE_EXISTS_ERROR:
                print("already exists.")
            else:
                print(err.msg)
        else:
            print("OK")

#Populate database with initial data sets
def initPopDB():
    pass


def print_databases():
    cnx = mysql.connector.connect(user='root', password='Welcome07', host='127.0.0.1')
    cursor = cnx.cursor()
    cursor.execute('show databases')
    for i in cursor:
        print (i)
    cnx.close()
