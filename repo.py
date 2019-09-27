from sqlalchemy import Boolean, Column, String, Integer, Float, MetaData, ForeignKey, create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, relationship
from operator import itemgetter
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


# class Dataset(Base):
#     __tablename__ = 'dataset'

#     data_name = Column(String)
#     data_path = Column(String)
#     data_id = Column(Integer, primary_key = True)
#     weighted_mean = Column(Float)
#     standard_deviation = Column(Float)
#     fpskew = Column(Float)
#     kurtosis = Column(Float)
#     entropy = Column(Float)
#     metric_time = Column(Float) #Time spent extracting statistical metrics

#     def __repr__(self):
#         return '<data_set(name= {}, path= {}, weighted_mean= {}, standard_deviation= {}, fpskew= {}, kurtosis= {}, .\
#                 information= {})>'.format(self.data_name,self.data_path,
#                                           self.weighted_mean,self.standard_deviation,
#                                           self.fpskew,self.kurtosis,self.information)

def data_set_factory(classname, tablename):
    def __repr__(self):
        return '<data_set(name= {}, path= {}, weighted_mean= {}, coefficient_of_variation= {}, fpskew= {}, kurtosis= {}, shanon_entropy= {}.\
         )>'.format(self.data_name, self.data_path,
                    self.weighted_mean, self.coefficient_variation,
                    self.fpskew, self.kurtosis, self.entropy)

    dataset = type(classname, (Base,), {
        "__tablename__": tablename,
        "data_name": Column(String),
        "data_path": Column(String),
        "data_id": Column(Integer, primary_key=True),
        "weighted_mean": Column(Float),
        "coefficient_variation": Column(Float),
        "fpskew": Column(Float),
        "kurtosis": Column(Float),
        "entropy": Column(Float),
        "metric_time": Column(Float),
        "__repr__": __repr__
    })

    globals()[classname] = dataset
    return dataset


def base_set_collection_factory(classname, tablename):
    def __repr__(self):
        return '<data_set(name= {}, path= {}, base_name= {} weighted_mean= {}, coefficient_of_variation= {}, fpskew= {}, kurtosis= {}, shanon_entropy= {}.\
         )>'.format(self.data_name, self.data_path, self.base_name,
                    self.weighted_mean, self.coefficient_variation,
                    self.fpskew, self.kurtosis, self.entropy)

    dataset = type(classname, (Base,), {
        "__tablename__": tablename,
        "data_name": Column(String),
        "data_path": Column(String),
        "base_name": Column(String),
        "data_id": Column(Integer, primary_key=True),
        "weighted_mean": Column(Float),
        "coefficient_variation": Column(Float),
        "fpskew": Column(Float),
        "kurtosis": Column(Float),
        "entropy": Column(Float),
        "metric_time": Column(Float),
        "__repr__": __repr__
    })

    globals()[classname] = dataset
    return dataset


class AlgClass(Base):
    __tablename__ = 'alg_class'

    class_name = Column(String)
    class_id = Column(Integer, primary_key=True)

    def __repr__(self):
        return '<alg_class(id={}, name={})>'.format(self.class_id, self.class_name)


class Result(Base):
    __tablename__ = 'results'

    result_id = Column(Integer, primary_key=True)
    meta_alg_name = Column(String)
    meta_base_name = Column(String)
    accuracy = Column(Float)
    training_time = Column(Float)
    rate_correct_score = Column(Float)

    def __repr__(self):
        return ('<result(id={}, meta_alg_name={}, meta_base_name={},'
                'accuracy={}, training_time={}, rate_correct_score={})>').format(self.result_id,
                                                                                 self.meta_alg_name,
                                                                                 self.meta_base_name,
                                                                                 self.accuracy,
                                                                                 self.training_time,
                                                                                 self.rate_correct_score)


class Algorithm(Base):
    __tablename__ = 'algorithm'

    alg_name = Column(String)
    alg_path = Column(String)
    alg_id = Column(Integer, primary_key=True)
    class_id = Column(Integer, ForeignKey('alg_class.class_id'))

    def __repr__(self):
        return '<algorithm(alg_id={}, name={}, path={}, class_id={})>'.format(self.alg_id, self.alg_name, self.alg_path,
                                                                              self.class_id)


class LearningCurve(Base):
    __tablename__ = 'learning_curves'

    data_id = Column(Integer, ForeignKey('all_data.data_id'))
    alg_id = Column(Integer, ForeignKey('algorithm.alg_id'))
    curve_id = Column(Integer, primary_key=True)
    train_time = Column(Float)
    accuracy_10 = Column(Float)
    accuracy_20 = Column(Float)
    accuracy_30 = Column(Float)
    alg = relationship("Algorithm", backref="learning_curves")
    data = relationship("DatasetAll", backref="learning_curves")

    def __repr__(self):
        return ('<LearningCurve(alg_name={},'
                '               alg_id={},'
                '            data_name={},'
                '              data_id={},'
                '           train_time={},'
                '           accuracy_10={},'
                '           accuracy_20={},'
                '           accuracy_30={})>').format(self.alg.alg_name,
                                                      self.alg_id,
                                                      self.data.data_name,
                                                      self.data.data_id,
                                                      self.train_time,
                                                      self.accuracy_10,
                                                      self.accuracy_20,
                                                      self.accuracy_30)


class Run(Base):
    __tablename__ = 'runs_all'

    data_id = Column(Integer, ForeignKey('all_data.data_id'))
    alg_id = Column(Integer, ForeignKey('algorithm.alg_id'))
    run_id = Column(Integer, primary_key=True)
    train_time = Column(Float)
    accuracy = Column(Float)
    alg = relationship("Algorithm", backref="run")
    data = relationship("DatasetAll", backref="run")

    def __repr__(self):
        return '<run(run_id={},alg_name={}, alg_id={}, data_name={},train_time={},accuracy={})>'.format(self.run_id,
                                                                                                        self.alg.alg_name,
                                                                                                        self.alg_id,
                                                                                                        self.data.data_name,
                                                                                                        self.train_time,
                                                                                                        self.accuracy)


def guess_factory(classname, tablename, dataset):
    def __repr__(self):
        return '<guess_set(data_name={},metabase_set={},metabase_table={},guess_algorithm={},actual_algorithm={},' \
               'correct={}>'.format(self.data_name,
                                    self.metabase_set,
                                    self.metabase_table,
                                    self.guess_algorithm,
                                    self.actual_algorithm,
                                    self.correct)

    guess_set = type(classname, (Base,), {
        "__tablename__": tablename,
        "data_id": Column(Integer, ForeignKey('{}.data_id'.format(dataset.__tablename__))),
        "data_name": Column(String),
        "metabase_set": Column(String),
        "metabase_table": Column(String),
        "guess_id": Column(Integer, primary_key=True),
        "guess_algorithm": Column(String),
        "guess_algorithm_id": Column(Integer, ForeignKey('algorithm.alg_id')),
        "actual_algorithm": Column(String),
        "actual_algorithm_id": Column(Integer, ForeignKey('algorithm.alg_id')),
        "correct": Column(Boolean),
        "__repr__": __repr__
    })

    globals()[classname] = guess_set
    return guess_set


def run_factory(classname, tablename, dataset):
    def __repr__(self):
        return '<run(run_id={},alg_name={},data_name={},train_time={},accuracy={})>'.format(self.run_id,
                                                                                            self.alg.alg_name,
                                                                                            self.data.data_name,
                                                                                            self.train_time,
                                                                                            self.accuracy)

    run = type(classname, (Base,), {
        "__tablename__": tablename,
        "data_id": Column(Integer, ForeignKey('{}.data_id'.format(dataset.__tablename__))),
        "ald_id": Column(Integer, ForeignKey('algorithm.alg_id')),
        "run_id": Column(Integer, primary_key=True),
        "train_time": Column(Float),
        "accuracy": Column(Float),
        "alg": relationship("Algorithm", backref=tablename),
        "data": relationship(dataset.__name__, backref=tablename),
        "__repr__": __repr__
    })

    globals()[classname] = run
    return run


def get_session():
    # Retrieve session object
    engine = create_engine('sqlite:///metabase.db')
    Session = sessionmaker(bind=engine)
    session = Session()
    return session


def define_meta():
    data_all = data_set_factory('DatasetAll', 'all_data')
    base_set_collection_limit = 30
    base_set_collections = [('BaseSetCollection{}'.format(i), 'base_set_collection_{}'.format(i)) for i in
                            range(base_set_collection_limit)]

    for inx, (basename, tablename) in enumerate(base_set_collections):
        this_base = 'base_collection_{}'.format(inx)
        globals()[this_base] = base_set_collection_factory(basename, tablename)

    # base_a = data_set_factory('BasesetA','base_set_a')
    # base_b = data_set_factory('BasesetB','base_set_b')
    # base_c = data_set_factory('BasesetC','base_set_c')
    # base_d = data_set_factory('BasesetD','base_set_d')
    # base_e = data_set_factory('BasesetE','base_set_e')
    # base_f = data_set_factory('BasesetF','base_set_f')
    # base_g = data_set_factory('BasesetG','base_set_g')
    # base_h = data_set_factory('BasesetH','base_set_h')
    # base_i = data_set_factory('BasesetI','base_set_i')
    # base_j = data_set_factory('BasesetJ','base_set_j')

    guesses_act = guess_factory('GuessesActive', 'guesses_active', data_all)
    guesses_ex = guess_factory('GuessesEx', 'guesses_ex', data_all)
    guesses_samp = guess_factory('GuessesSamp', 'guesses_samp', data_all)


def craft_system():
    import create_db as cdb
    repo.define_meta()
    cdb.create_tables(metadata)


def add_dset(session, args):
    # Method requires defineMeta to have been previously run
    expected_args = ['class_name', 'data_name', 'data_path', 'data_set']
    class_name, data_name, data_path, data_set = itemgetter(*expected_args)(args)

    base = globals()[class_name]

    dwmean, ds_ccv, dpskew, dkurt, dent, duration = mc.extractFeatures(data_set)

    new_set = base(data_name=data_name,
                   data_path=data_path,
                   weighted_mean=dwmean,
                   coefficient_variation=ds_ccv,
                   fpskew=dpskew,
                   kurtosis=dkurt,
                   entropy=dent,
                   metric_time=duration)

    session.add(new_set)
    session.commit()


def add_set_to_collection(session, args):
    expected_args = ['meta_base_collection', 'meta_base_label',
                     'data_file_name', 'data_path']
    meta_base_collection, meta_base_label, data_name, data_path = itemgetter(*expected_args)(args)
    all_data = globals()['DatasetAll']
    base = globals()[meta_base_collection]
    data_set = session.query(all_data).filter_by(data_path=data_path).first()

    new_set = base(data_name=data_set.data_name,
                   data_path=data_set.data_path,
                   base_name=meta_base_label,
                   weighted_mean=data_set.weighted_mean,
                   coefficient_variation=data_set.coefficient_variation,
                   fpskew=data_set.fpskew,
                   kurtosis=data_set.kurtosis,
                   entropy=data_set.entropy,
                   metric_time=data_set.metric_time)

    session.add(new_set)
    session.commit()


def add_run(session, args):
    classname, tablename, data_id, alg_id, train_time, accuracy = args.values()
    base = globals()[classname]

    n_run = base(data_id=data_id,
                 alg_id=alg_id,
                 train_time=train_time,
                 accuracy=accuracy)

    session.add(n_run)
    session.commit()


def add_factory_dset(dname, dpath, dset, nc, session):
    """
    Add item to factory produced dset table 
    """
    pass


def add_curve(data_id, alg_id, results, session):
    accuracy_10, accuracy_20, accuracy_30, training_time = results
    n_curve = LearningCurve(data_id=data_id,
                            alg_id=alg_id,
                            accuracy_10=accuracy_10,
                            accuracy_20=accuracy_20,
                            accuracy_30=accuracy_30,
                            train_time=training_time)
    session.add(n_curve)
    session.commit()


def add_algorithm(alg_name, alg_path, class_id, session):
    n_algorithm = Algorithm(alg_name=alg_name,
                            alg_path=alg_path,
                            class_id=class_id)
    session.add(n_algorithm)
    session.commit()


def add_alg_class(class_name, session):
    n_class = AlgClass(class_name=class_name)
    session.add(n_class)
    session.commit()


def add_to_results(meta_alg, metabase, accuracy, train_time, rcs, session):
    n_result = Result(meta_alg_name=meta_alg,
                      meta_base_name=metabase,
                      accuracy=accuracy,
                      training_time=train_time,
                      rate_correct_score=rcs)
    session.add(n_result)
    session.commit()


def add_to_guesses(tableName, className, data_id, data_name, guess_algorithm_id, actual_algorithm_id, session):
    base = globals()[className]

    if guess_algorithm_id == actual_algorithm_id:
        correct = 0
    else:
        correct = 1

    algs = session.query(repo.Algorithm)
    try:
        guess_algorithm = algs.filter_by(alg_id=guess_algorithm_id).all()[0].alg_name
    except IndexError:
        guess_algorithm = 'NO ALG FOUND'

    try:
        actual_algorithm = algs.filter_by(alg_id=actual_algorithm_id).all()[0].alg_name
    except IndexError:
        actual_algorithm = 'NO ALG FOUND'

    guess = base(data_id=data_id,
                 data_name=data_name,
                 metabase_table=tableName,
                 guess_algorithm=guess_algorithm,
                 guess_algorithm_id=guess_algorithm_id,
                 actual_algorithm=actual_algorithm,
                 actual_algorithm_id=actual_algorithm_id,
                 correct=correct)

    session.add(guess)
    session.commit()
