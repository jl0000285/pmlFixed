from sqlalchemy import Column, String, Integer, Float, MetaData, ForeignKey, create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, relationship
import pdb
import repo
import time
import re
import metaCollector as mc
import os
import subprocess
import fparser as fp

metadata = MetaData()
Base = declarative_base(metadata=metadata)

datasets = {
        'train_data_a': [],
        'train_data_b': [],
}

class Dataset(Base):
    __tablename__ = 'dataset'

    data_name = Column(String)
    data_path = Column(String)
    data_id = Column(Integer, primary_key = True)
    weighted_mean = Column(Float)
    standard_deviation = Column(Float)
    fpskew = Column(Float)
    kurtosis = Column(Float)
    information = Column(Float)

    def __repr__(self):
        return '<data_set(name= {}, path= {}, weighted_mean= {}, standard_deviation= {}, fpskew= {}, kurtosis= {}, .\
                information= {})>'.format(self.data_name,self.data_path,
                                          self.weighted_mean,self.standard_deviation,
                                          self.fpskew,self.kurtosis,self.information)

def data_set_factory(classname,tablename):
    def __repr__(self):
        return '<data_set(name= {}, path= {}, weighted_mean= {}, standard_deviation= {}, fpskew= {}, kurtosis= {}, .\
               information= {})>'.format(self.data_name,self.data_path,
                                         self.weighted_mean,self.standard_deviation,
                                         self.fpskew,self.kurtosis,self.information)
    dataset = type(classname,(Base,),{
        "__tablename__": tablename,
        "data_name": Column(String),
        "data_path": Column(String),
        "data_id": Column(Integer, primary_key = True),
        "weighted_mean": Column(Float),
        "standard_deviation": Column(Float),
        "fpskew": Column(Float),
        "kurtosis": Column(Float),
        "information": Column(Float),
        "__repr__": __repr__
    })

    globals()[classname] = dataset
    return dataset
        
class AlgClass(Base):
    __tablename__= 'alg_class'

    class_name = Column(String)
    class_id = Column(Integer, primary_key = True)

    def __repr__(self):
        return '<alg_class(id={}, name={})>'.format(self.class_id,self.class_name)

class Algorithm(Base):
    __tablename__= 'algorithm'

    alg_name = Column(String)
    alg_path = Column(String)
    alg_id = Column(Integer, primary_key = True)
    class_id = Column(Integer, ForeignKey('alg_class.class_id'))

    def __repr__(self):
        return '<algorithm(alg_id={}, name={}, path={}, class_id={})>'.format(self.alg_id,self.alg_name,self.alg_path,self.class_id)

    

class Run(Base):
    __tablename__= 'run'

    data_id = Column(Integer, ForeignKey('dataset.data_id'))
    alg_id = Column(Integer, ForeignKey('algorithm.alg_id'))
    run_id = Column(Integer, primary_key = True)
    train_time = Column(Float)
    accuracy = Column(Float)
    alg = relationship("Algorithm",backref="run")
    data = relationship("Dataset",backref="run")

    def __repr__(self):
        return '<run(run_id={},alg_name={},data_name={},train_time={},accuracy={})>'.format(self.run_id,
                                                                                                self.alg.alg_name,
                                                                                                self.data.data_name,
                                                                                                self.train_time,
                                                                                                self.accuracy)

def run_factory(classname,tablename,dataset):
    def __repr__(self):
        return '<run(run_id={},alg_name={},data_name={},train_time={},accuracy={})>'.format(self.run_id,
                                                                                                self.alg.alg_name,
                                                                                                self.data.data_name,
                                                                                                self.train_time,
                                                                                                self.accuracy)
    
    run = type(classname,(Base,),{
        "__tablename__": tablename,
        "data_id": Column(Integer, ForeignKey('{}.data_id'.format(dataset.__tablename__))),
        "ald_id": Column(Integer, ForeignKey('algorithm.alg_id')),
        "run_id": Column(Integer, primary_key = True),
        "train_time": Column(Float),
        "accuracy": Column(Float),
        "alg": relationship("Algorithm", backref=tablename),
        "data": relationship(dataset.__name__, backref=tablename),
        "__repr__": __repr__
    })

    globals()[classname] = run
    return run

def get_session():
    #Retrieve session object
    engine=create_engine('sqlite:///metabase.db')
    Session=sessionmaker(bind=engine)
    session=Session()
    return session

def craftSystem():
    import create_db as cdb
    data_all = data_set_factory('DatasetAll','all_data')
    base_a = data_set_factory('DatasetA','base_set_a')
    base_b = data_set_factory('DatasetB','base_set_b')
    base_c = data_set_factory('DatasetC','base_set_c')

    train_a = data_set_factory('TestsetA','test_set_a')
    train_b = data_set_factory('TestsetB','test_set_b')
    train_c = data_set_factory('TestsetC','test_set_c')

    run_all = run_factory('RunAll','run_all',data_all)
    runExA = run_factory('RunExA','run_ex_a',base_a)
    runExB = run_factory('RunExB','run_ex_b',base_b)
    runExC = run_factory('RunExC','run_ex_c',base_c)
    
    cdb.create_tables(metadata)

def add_dset(dname,dpath, dset, nc, session):
    dwmean,ds_dev,dpskew,dkurt = mc.extractFeatures(dset,nc)
    minfo = 0
    dw,ds,dp,dk = dwmean[0], ds_dev[0], dpskew[0],dkurt[0]
    d_set = Dataset(data_name=dname,
                     data_path=dpath,
                     weighted_mean=dw,
                     standard_deviation=ds,
                     fpskew=dp,
                     kurtosis=dk,
                     information=minfo)
    session.add(d_set)
    session.commit()

def ext_add_dset(classname,tablename,dname,dpath,dset,nc,session):
    base = globals()[classname]
        
    dwmean,ds_dev,dpskew,dkurt = mc.extractFeatures(dset,nc)
    minfo = 0
    
    try:
        dw,ds,dp,dk = dwmean[0], ds_dev[0], dpskew[0],dkurt[0]
    except Exception as ex:
        pdb.set_trace()
        pass
    
    d_set = base(data_name=dname,
                 data_path=dpath,
                 weighted_mean=dw,
                 standard_deviation=ds,
                 fpskew=dp,
                 kurtosis=dk,
                 information=minfo)
    session.add(d_set)
    session.commit()
    

def add_factory_dset(dname,dpath,dset,nc,session):
    """
    Add item to factory produced dset table 
    """
    pass

def add_run(data_id,alg_id,train_time,accuracy,session):
     n_run = Run(data_id=data_id,
                 alg_id=alg_id,
                 train_time=train_time,
                 accuracy=accuracy)
     session.add(n_run)
     session.commit()

def add_factory_run(className, data_id,alg_id,train_time,accuracy,session):
    """
    Add to item to factory produced run table 
    """
    pass

# Populates the algorithms table
def add_alg(alg_name,alg_path,class_id,session):
    n_algorithm = Algorithm(alg_name=alg_name,
                            alg_path=alg_path,
                            class_id=class_id)
    session.add(n_algorithm)
    session.commit()


def add_algClass(class_name,session):
    n_class = AlgClass(class_name=class_name)
    session.add(n_class)
    session.commit()
