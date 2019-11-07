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
import pandas as pd
from tabulate import tabulate


class DbHandler(object):

    def __init__(self):
        """
        Handler object for interfacing with the metabase 
        """
        # Reminder: First item is the name of the generated repo class, second is the table name
        self.base_set_collection_limit = 30
        self.base_set_limit = 10
        self.base_set_collections = [('BaseSetCollection{}'.format(i), 'base_set_collection_{}'.format(i)) for i in
                                     range(self.base_set_collection_limit)]
        self.set_labels = ['base_{}'.format(i) for i in range(self.base_set_limit)]

        # self.baseDataTables = [('BasesetA','base_sets_a'),
        #                        ('BasesetB','base_sets_b'),
        #                        ('BasesetC','base_sets_c'),
        #                        ('BasesetD','base_sets_d'),
        #                        ('BasesetE','base_sets_e'),
        #                        ('BasesetF','base_sets_f'),
        #                        ('BasesetG','base_sets_g'),
        #                        ('BasesetH','base_sets_h'),
        #                        ('BasesetI','base_sets_i'),
        #                        ('BasesetJ','base_sets_j')]

        self.RUN_NOT_FOUND = False

    def get_session(self):
        self.session = repo.get_session()

    def setup_session(self):
        """Get session object then define repo metadata"""
        self.get_session()
        repo.define_meta()

    def db_init(self):
        """
        Create and populate extended metabase
        """
        repo.craft_system()
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
        for dirpath, dirname, filelist in os.walk('./data/init'):
            for filename in filelist:
                if re.search(r".*[.]data$", filename):
                    print("dirpath: {}, dirname: {}, filename: {}"
                          .format(dirpath, dirname, filename))
                    dpath = '{}/{}'.format(dirpath, filename)
                    data = fp.Parser(fp.LC_SINGLE, dpath, False, .25, ',')
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
        for dpath, filename in filelist:
            print("Adding set {} to all_sets at {}".format(filename, dpath))
            data = fp.Parser(fp.LC_SINGLE, dpath, False, .25, ',')
            target_input = data.convert_file()
            target_input = data.limit_size(target_input)  # limiting size of datasets for sanity

            try:
                repo.add_dset(all_dict['className'],
                              all_dict['tableName'],
                              filename,
                              dpath,
                              target_input,
                              self.session)
            except Exception as ex:
                print("Exception occured whilst trying to add dataset: {}".format(ex))

    def populate_metabase_collections(self):
        filelist = self.get_allowed_files()
        datasets = self.session.query(repo.DatasetAll).all()

        for meta_base_collection, collection_table in self.base_set_collections:
            for meta_base_label in self.set_labels:
                print("Determining base sets for meta_base_collection:{} set:{}"
                      .format(meta_base_collection, meta_base_label))
                base = []
                for i in range(int(math.floor(len(filelist) / 5))):
                    inx = random.randrange(0, len(filelist) - 1)
                    base.append(filelist[inx])

                for data_path, data_file_name in base:
                    dataset = self.session.query(repo.DatasetAll).filter_by(data_path=data_path).first()
                    print("Adding set {} at {}:{}".format(data_file_name, meta_base_label, meta_base_collection))
                    repo.add_set_to_collection(self.session, dict(meta_base_collection=meta_base_collection,
                                                                  data_id=dataset.data_id,
                                                                  meta_base_label=meta_base_label,
                                                                  data_file_name=data_file_name,
                                                                  data_path=data_path))

    def populate_alg_class(self):
        """Initialize algorithms class table"""
        class_A = repo.AlgClass(class_name='supervised')
        self.session.add(class_A)
        self.session.commit()

    def populate_algorithms(self):
        algTypes = {
            'svm': ('sk.svm', 'supervised'),
            'clustering': ('sk.clustering', 'supervised'),
            'neural_network': ('sk.neural_network', 'supervised'),
            'bayes': ('sk.bayes', 'supervised'),
            'regression': ('sk.regression', 'supervised')
        }
        for key in algTypes:
            class_id = self.session.query(repo.AlgClass). \
                filter_by(class_name=algTypes[key][1]).first()
            class_id = class_id.class_id
            repo.add_alg(key, algTypes[key][0], class_id, self.session)

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
            data = fp.Parser(fp.COMMA_DL, d_set.data_path,
                             fp.TP_TRUE, per=.25)
            target_input = data.convert_file()
            shuffle(target_input)  # keep commented while debugging
            target_input = data.limit_size(target_input)  # limiting size of datasets for sanity
            train_data, test_data = data.partition(target_input)
            X_train, y_train = data.split_last_column(train_data)
            X_test, y_test = data.split_last_column(test_data)
            sk = skh.SkHandler(X_train, y_train, X_test, y_test)
            for alg in algs:
                alg_id = alg.alg_id
                evstring = '{}()'.format(alg.alg_path)
                print(evstring)
                try:
                    durr, acc = eval(evstring)
                except Exception as ex:
                    print("Could not train dataset {} with method {}: {}".format(d_set.data_path,
                                                                                 alg.alg_path,
                                                                                 ex))
                    durr, acc = [float('inf'), 0]
                    pdb.set_trace()

                repo.add_run(data_id, alg_id, durr, acc, self.session)

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
            data = fp.Parser(fp.COMMA_DL, d_set.data_path,
                             fp.TP_TRUE, per=.25)
            target_input = data.convert_file()
            shuffle(target_input)
            target_input = data.limit_size(target_input)  # limiting size of datasets for sanity
            percents = [0.1, 0.2, 0.3]

            for alg in algs:
                results = []
                train_time = 0
                alg_id = alg.alg_id
                evstring = '{}()'.format(alg.alg_path)
                for percent in percents:
                    shuffle(target_input)
                    train_data, test_data = data.partition(target_input, per=percent)
                    X_train, y_train = data.split_last_column(train_data)
                    X_test, y_test = data.split_last_column(test_data)
                    sk = skh.SkHandler(X_train, y_train, X_test, y_test)
                    print('{} evaluated at {} percent'.format(evstring, str(percent)))
                    try:
                        durr, acc = eval(evstring)
                        train_time += durr
                        results.append(acc)
                    except Exception as ex:
                        print("Could not train dataset {} with method {}: {}".format(d_set.data_path,
                                                                                     alg.alg_path,
                                                                                     ex))
                        durr, acc = [float('inf'), 0]
                        results.append(acc)

                results.append(train_time)
                repo.add_curve(data_id, alg_id, results, self.session)

    def populate_results(self):
        def calculate_accuracy(guesses):
            num_correct = guesses.filter_by(correct=0).count()
            num_overall = guesses.count()
            if num_overall > 0:
                acc = num_correct / num_overall
            else:
                acc = 0
            return acc

        def calculate_training_time(guesses, alg):
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

        def calculate_rate_correct_score(acc, train_time):
            if not train_time > 0:
                rcs = 0
            else:
                rcs = acc / train_time

            return rcs

        meta_algs = ['GuessesEx', 'GuessesActive', 'GuessesSamp']

        for alg in meta_algs:
            meta_alg_class = getattr(repo, alg)
            for collection_class_name, collection_table_name in self.base_set_collections:
                for set_label in self.set_labels:
                    print('Compiling results: alg:{} collection:{} base: {}'.format(alg,
                                                                                    collection_table_name,
                                                                                    set_label))
                    guesses = self.session.query(meta_alg_class).filter_by(collection_table=collection_table_name).\
                                                                 filter_by(metabase_name=set_label)
                    acc = calculate_accuracy(guesses)
                    train_time = calculate_training_time(guesses, alg)
                    rcs = calculate_rate_correct_score(acc, train_time)
                    repo.add_to_results(self.session, dict(meta_alg=alg,
                                                           collection_table=collection_table_name,
                                                           metabase_name=set_label,
                                                           accuracy=acc,
                                                           training_time=train_time,
                                                           rate_correct_score=rcs))

    def get_active_base(self, collection_name, base_name):
        """
        Steps: 
        1. Obtain metabase candidate datasets
        2. Add half of them to active base for training
        3. Decide on another fourth of them using active learning 
          (by comparing amount of information in datasets)
        4. Return active base
        """

        def sum_of_distances(metafeature, dinx, candidates):
            vector = get_feature_vector(metafeature, candidates)
            dist_summ = sum([abs(x - vector[dinx]) for x in vector])
            return dist_summ

        def get_feature_vector(metafeature, candidates):
            vector = []
            for dset in candidates:
                value_string = 'dset.{}'.format(metafeature)
                value = eval(value_string)
                vector.append(value)
            return vector

        def spread_without_set(metafeature, dinx, candidates):
            vector = get_feature_vector(metafeature, candidates)
            vector.pop(dinx)

            max_val = max(vector)
            min_val = min(vector)

            spread = abs(max_val - min_val)
            return spread

        def calculate_uncertainty_for_feature(metafeature, dinx, candidates):
            try:
                uncertainty = sum_of_distances(metafeature, dinx, candidates) \
                              / spread_without_set(metafeature, dinx, candidates)
            except Exception as ex:
                pdb.set_trace()
                pass
            return uncertainty

        def rank_uncertainty_for_feature(metafeature, candidates):
            ranked_tuples = []
            for inx, dset in enumerate(candidates):
                uncertainty = calculate_uncertainty_for_feature(metafeature, inx, candidates)
                rank_tuple = [uncertainty, inx, dset]
                ranked_tuples.append(rank_tuple)

            ranked_tuples.sort(reverse=True)
            return ranked_tuples

        def get_most_uncertain_dataset(candidates):
            metafeatures = ['weighted_mean', 'coefficient_variation', 'fpskew', 'kurtosis', 'entropy']
            score_list = [[0, dset] for dset in candidates]  # (Score, dataset)

            for feature in metafeatures:
                ranked_tuples = rank_uncertainty_for_feature(feature,
                                                             candidates)  # list of (uncertainty,original Index,candidate) where index is rank

                for inx, (uncertainty, original_index, candidate) in enumerate(ranked_tuples):
                    score_list[original_index][0] += inx + 1  # Here a lower total score means higher uncertainty

            max_inx = 0  # index of set with highest uncertainity i.e set with lowest rank score

            for inx, tup in enumerate(score_list):
                if tup[0] < score_list[max_inx][0]:
                    max_inx = inx

            return max_inx

        class_object = getattr(repo, collection_name)
        bases = self.session.query(class_object).filter_by(base_name=base_name).all()
        candidates = list(bases)
        active_base = []

        for i in range(int(math.floor(len(bases) / 2))):
            inx = get_most_uncertain_dataset(candidates)
            cand = candidates.pop(inx)
            active_base.append(cand)
        return active_base

    def compute_objective(self, run):
        """Compute loss function for a given run"""
        return run.accuracy

    def find_best_algorithm(self, data_id):
        """
        Fetch the best performing algorithm from the runs_all 
        table for some given dataset 
        """
        runs = self.session.query(repo.Run).filter_by(data_id=data_id).all()
        scores = [[run.alg_id, self.compute_objective(run)] for run in runs]
        values = [tup[-1] for tup in scores]
        try:
            max_inx = values.index(max(values))
            retVal = scores[max_inx][0]
        except ValueError as ex:
            retVal = self.RUN_NOT_FOUND

        return retVal

    def guess_with_clusterer(self, base_sets, dataset):
        """
        Use clustering algorithm with given base_sets to guess
        datasets best performing algorithm 
        """
        from sklearn.cluster import KMeans
        import numpy as np
        points = [[set.weighted_mean, set.coefficient_variation, set.fpskew, set.kurtosis, set.entropy]
                  for set in base_sets]
        X = np.array(points)
        # num_clust = len(np.unique(points))
        num_clust = len(points)
        Kmeans = KMeans(n_clusters=num_clust, random_state=0).fit(X)
        test = [dataset.weighted_mean,
                dataset.coefficient_variation,
                dataset.fpskew,
                dataset.kurtosis,
                dataset.entropy]
        guess_label = Kmeans.predict(
            np.array(test).reshape(1, -1))  # Returns dataset from base_sets dataset is closest too

        try:
            guess_inx = np.where(Kmeans.labels_ == guess_label)[0][0]
        except IndexError as ex:
            guess_label = np.random.choice(Kmeans.labels_.tolist())
            guess_inx = np.where(Kmeans.labels_ == guess_label)[0][0]

        guess = self.find_best_algorithm(base_sets[guess_inx].data_id)

        if guess is False:
            pdb.set_trace()

        return (guess)

    def guess_with_sampler(self, base_sets, dataset):
        """
        Use learning curve distance comparisons to determine best algorithm 
        """

        def calculate_distance_between_sets(curvesA, curvesB):
            distance = 0.0
            percents = ['10', '20', '30']
            for i in range(len(curvesA)):
                for percent in percents:
                    a_string = 'curvesA[{}].accuracy_{}'.format(str(i), percent)
                    b_string = 'curvesB[{}].accuracy_{}'.format(str(i), percent)
                    acc_a = eval(a_string)
                    acc_b = eval(b_string)
                    distance += (acc_a - acc_b) ** 2

            return distance

        def get_distance_between_sets(curvesA, curvesB):
            distance = 0.0
            algs = self.session.query(repo.Algorithm).all()
            for alg in algs:
                curveA = curvesA.filter_by(alg_id=alg.alg_id).all()
                curveB = curvesA.filter_by(alg_id=alg.alg_id).all()
                distance += calculate_distance_between_sets(curveA, curveB)
                return distance

        def get_all_set_distances(base_sets, dataset):
            """
            distance items look like [distance, base_set_id]
            """
            distances = []
            dset_curves = self.session.query(repo.LearningCurve).filter_by(data_id=dataset.data_id)
            for set in base_sets:
                set_curves = self.session.query(repo.LearningCurve).filter_by(data_id=set.data_id)
                distances.append([get_distance_between_sets(dset_curves, set_curves), set.data_id])
            return distances

        set_distances = get_all_set_distances(base_sets, dataset)
        set_distances.sort(reverse=True)
        guess = self.find_best_algorithm(set_distances[0][1])
        return guess

    def guesses_exhaustive(self):
        """Given a set of databases, make guesses as to what would be the best 
        machine based off entirety of current metabase 
        """
        guess_class, guess_table = ('GuessesEx', 'guesses_ex')
        datasets = self.session.query(repo.DatasetAll).all()

        for class_name, table_name in self.base_set_collections:
            class_object = getattr(repo, class_name)
            for set_label in self.set_labels:
                print('Guessing exhaustively: {}: {}'.format(table_name, set_label))
                curr_base = self.session.query(class_object).filter_by(base_name=set_label).all()
                base_names = [set.data_name for set in curr_base]
                for dataset in datasets:
                    if dataset.data_name not in base_names:
                        guess = self.guess_with_clusterer(curr_base, dataset)
                        """
                        #need to modify various declartive bases such that data_id is a key
                        #that exists only within datasets_all and so that the various base
                        #set classes store that key on them selves, changing the column called
                        data_id in the base set classes to set_id. 
                        """
                        solution = self.find_best_algorithm(dataset.data_id)
                        repo.add_to_guesses(self.session, dict(guess_class=guess_class,
                                                               collection_table=table_name,
                                                               metabase_name=set_label,
                                                               data_id=dataset.data_id,
                                                               data_name=dataset.data_name,
                                                               guess_algorithm_id=guess,
                                                               actual_algorithm_id=solution))

    def guesses_active(self):
        """Given a set of databases, make guesses as to what would be the best 
        machine based off the uncertainty values of the datasets contained within 
        """
        guess_class, guess_table = ('GuessesActive', 'guesses_active')
        datasets = self.session.query(repo.DatasetAll).all()

        for class_name, table_name in self.base_set_collections:
            for set_label in self.set_labels:
                print('Guessing Active: {}: {}'.format(table_name, set_label))
                active_base = self.get_active_base(class_name, set_label)
                base_names = [set.data_name for set in active_base]
                for dataset in datasets:
                    if dataset.data_name not in base_names:
                        guess = self.guess_with_clusterer(active_base, dataset)
                        """
                        #need to modify various declartive bases such that data_id is a key
                        #that exists only within datasets_all and so that the various base
                        #set classes store that key on them selves, changing the column called
                        data_id in the base set classes to set_id. 
                        """
                        solution = self.find_best_algorithm(dataset.data_id)
                        repo.add_to_guesses(self.session, dict(guess_class=guess_class,
                                                               collection_table=table_name,
                                                               metabase_name=set_label,
                                                               data_id=dataset.data_id,
                                                               data_name=dataset.data_name,
                                                               guess_algorithm_id=guess,
                                                               actual_algorithm_id=solution))

    def guesses_sampling(self):
        """Given a set of databases, make guesses as to what would be the best
        machine based off the sampling curves of the datasets contained within
        """
        guess_class, guess_table = ('GuessesSamp', 'guesses_samp')
        datasets = self.session.query(repo.DatasetAll).all()

        for class_name, table_name in self.base_set_collections:
            for set_label in self.set_labels:
                print('Guessing Sampling: {}: {}'.format(table_name, set_label))
                class_object = getattr(repo, class_name)
                curr_base = self.session.query(class_object).filter_by(base_name=set_label).all()
                base_names = [set.data_name for set in curr_base]
                for dataset in datasets:
                    if dataset.data_name not in base_names:
                        guess = self.guess_with_sampler(curr_base, dataset)
                        solution = self.find_best_algorithm(dataset.data_id)
                        repo.add_to_guesses(self.session, dict(guess_class=guess_class,
                                                               collection_table=table_name,
                                                               metabase_name=set_label,
                                                               data_id=dataset.data_id,
                                                               data_name=dataset.data_name,
                                                               guess_algorithm_id=guess,
                                                               actual_algorithm_id=solution))

    def print_databases(self):
        cnx = mysql.connector.connect(user='root', password='Welcome07', host='127.0.0.1')
        cursor = cnx.cursor()
        cursor.execute('show databases')
        for i in cursor:
            print(i)
        cnx.close()


    def get_allowed_files(self):
        """Return list of allowed file tuples where elem one is path and 
        elem two is name of the file
        """
        f = []
        allowed_types = ['data', 'svm', 'dat']
        for dirpath, dirname, filelist in os.walk('./data/datasets'):
            for filename in filelist:
                for t in allowed_types:
                    pat = '.*[.]{}$'.format(t)
                    if (re.search(pat, filename)):
                        # print ("dirpath: {}, dirname: {}, filename: {}"
                        #       .format(dirpath,dirname,filename))
                        dpath = '{}/{}'.format(dirpath, filename)
                        tup = (dpath, filename)
                        f.append(tup)
        return f


class ResultsAnalyzer:
    """
    Analyze results table and produce a likelihood of rejection of the null hypothesis
    where the null hypthesis is that n1=n2=n3 for all algorithms where n1 is the number of first place
    results, n2 the number of second place results, and n3 is the number of third place results
    """
    def __init__(self, dbh_):
        # WARNING: TONS of temporal coupling in this class. Have to use main method for it to work at all
        # Moreover main method is expected to be run once and only once
        self.dbh = dbh_  # Associated database handler
        self.meta_algs = self.dbh.session.query(repo.Result.meta_alg_name).distinct()
        self.results_dict = {}
        self.proportions_dict = {}
        self.t_scores_dict = {}

        self.means_dict = {}
        self.stds_dict = {}

        self.results_keys = []
        self.alg_keys = []
        self.pos_keys = []

        self.initialize()

    def initialize(self):
        E = self.dbh.base_set_limit / len(self.meta_algs.all())  # Expected value given all metalearners are equal
        self.craft_initial_results_dict()
        self.add_data_to_results_dict()
        self.get_means_over_collections()
        self.get_stds_from_collections()
        self.get_proportion_probabilities_from_collections(E)
        self.get_t_scores_from_collections(E)
        self.get_and_sort_keys()

    def craft_initial_results_dict(self):
        for i in range(self.dbh.base_set_collection_limit):
            self.results_dict['sample_{}'.format(i)] = dict()
            for alg in self.meta_algs:
                res_dict = {}
                for j in range(len(self.meta_algs.all())):
                    res_dict[str(j)] = 0
                self.results_dict['sample_{}'.format(i)][str(alg[0])] = res_dict.copy()

    def add_data_to_results_dict(self):
        for inx, (collection_name, collection_table) in enumerate(self.dbh.base_set_collections):
            for label in self.dbh.set_labels:
                accuracies = []
                for alg in self.meta_algs:
                    res = self.dbh.session.query(repo.Result).filter_by(meta_alg_name=str(alg[0]),
                                                                    collection_table=collection_table,
                                                                    meta_base_name=label).first()
                    tup = (res.accuracy, str(alg[0]))
                    accuracies.append(tup)
                accuracies.sort(reverse=True)
                for inx2, acc in enumerate(accuracies):
                    self.results_dict['sample_{}'.format(inx)][acc[1]][str(inx2)] += 1

    def get_means_over_collections(self):
        N = len(self.results_dict)
        for sample in self.results_dict:
            samp_dict = self.results_dict[sample]
            for alg in samp_dict:
                finish_dict = samp_dict[alg]
                for pos in finish_dict:
                    count = finish_dict[pos]
                    if alg in self.means_dict and pos in self.means_dict[alg]:
                        orig_count = self.means_dict[alg][pos]
                        self.means_dict[alg][pos] = orig_count + count
                    else:
                        if alg not in self.means_dict:
                            self.means_dict[alg] = dict()
                        if pos not in self.means_dict[alg]:
                            self.means_dict[alg][pos] = count
        for alg in self.means_dict:
            for pos in self.means_dict[alg]:
                self.means_dict[alg][pos] = self.means_dict[alg][pos] / N

    def get_stds_from_collections(self):
        # rho = sqrt(1/N * ( summation(i=1-to-N)(x_i - mean)^2 ))
        N = len(self.results_dict)

        # initialize stds_dict
        for alg in self.means_dict:
            self.stds_dict[alg] = dict()
            for pos in self.means_dict[alg]:
                self.stds_dict[alg][pos] = 0

        # sum squared diffrences
        for sample in self.results_dict:
            for alg in self.results_dict[sample]:
                for pos in self.results_dict[sample][alg]:
                    squared_difference = (self.results_dict[sample][alg][pos] - self.means_dict[alg][pos]) ** 2
                    self.stds_dict[alg][pos] += squared_difference

        # Normalize and root squared_differences
        for alg in self.stds_dict:
            for pos in self.stds_dict[alg]:
                self.stds_dict[alg][pos] = (self.stds_dict[alg][pos] / N) ** (1 / 2)

    def get_proportion_probabilities_from_collections(self, E):
        def calculate_sample_proportion_probability(i, E, N):
            """
            Calculate probability of sample proportion from:
            i = number of yesses,
            E = expected number of yesses
            N = the over all number of trials
            M = max number of yeses
            Logic based from Emperical methods for artificial intelligence by paul cohen,
            page 112
            """
            r = E / N
            NchooseK = (math.factorial(N) / (math.factorial(i) * math.factorial(N - i)))
            probCalc = r ** (i) * (1 - r) ** (N - i)
            P = NchooseK * probCalc
            return P
        N = 0
        for pos in self.results_dict['sample_0']['GuessesEx']:
            N += self.results_dict['sample_0']['GuessesEx'][pos]

        for sample in self.results_dict:
            self.proportions_dict[sample] = dict()
            for alg in self.results_dict[sample]:
                self.proportions_dict[sample][alg] = dict()
                for pos in self.results_dict[sample][alg]:
                    self.proportions_dict[sample][alg][pos] = calculate_sample_proportion_probability(
                        self.results_dict[sample][alg][pos],
                        E,
                        N
                    )

    def get_t_scores_from_collections(self, E):
        def calculate_t_score(value, std, E, N):
            # t = sample_mean-pop_mean / ( samp_std / root(number_samps) )
            mean_diff = value - E
            denom = std / (N) ** (1 / 2)
            t_score = mean_diff / denom
            return t_score

        N = len(self.results_dict)
        for pos in self.results_dict['sample_0']['GuessesEx']:
            N += self.results_dict['sample_0']['GuessesEx'][pos]

        for sample in self.results_dict:
            self.t_scores_dict[sample] = dict()
            for alg in self.results_dict[sample]:
                self.t_scores_dict[sample][alg] = dict()
                for pos in self.results_dict[sample][alg]:
                    self.t_scores_dict[sample][alg][pos] = calculate_t_score(
                        self.results_dict[sample][alg][pos],
                        self.stds_dict[alg][pos],
                        E,
                        N
                    )

    def get_and_sort_keys(self):
        self.results_keys = ResultsAnalyzer.human_sort(self.results_dict.keys())
        self.alg_keys = ResultsAnalyzer.human_sort(self.results_dict[self.results_keys[0]].keys())
        self.pos_keys = ResultsAnalyzer.human_sort(self.results_dict[self.results_keys[0]][self.alg_keys[0]].keys())

    def get_lists_of_lists_from_nested_dicts(self, nested):
        # Get lists of lists from nesteds that look like
        # results dict
        set_array = []
        for sample in self.results_keys:
            sample_ar = []
            for alg in self.alg_keys:
                alg_ar = []
                for pos in self.pos_keys:
                    alg_ar.append(nested[sample][alg][pos])
                sample_ar.append(alg_ar)
            set_array.append(sample_ar)
        # arrays = [[[p for d, p in v.items()] for k, v in i.items()] for j, i in nested.items()]
        return set_array

    def get_lists_of_lists_for_small_nested(self, nested):
        # Get lists of lists from nesteds that look like
        # means_dict
        set_array = []
        for alg in self.alg_keys:
            alg_ar = []
            for pos in self.pos_keys:
                alg_ar.append(nested[alg][pos])
            set_array.append(alg_ar)
        return set_array

    def get_average_of_samples(self, samples):
        averaged_grid = {}
        N = len(samples)

        # initialize averaged_grid:
        for alg in self.alg_keys:
            averaged_grid[alg] = dict()
            for pos in samples['sample_0'][alg]:
                averaged_grid[alg][pos] = 0

        # Sum values across samples
        for sample in self.results_keys:
            for alg in self.alg_keys:
                for pos in self.pos_keys:
                    averaged_grid[alg][pos] += samples[sample][alg][pos]

        # divide grid values by N to obtain means
        for alg in self.alg_keys:
            for pos in self.pos_keys:
                averaged_grid[alg][pos] = averaged_grid[alg][pos] / N

        return averaged_grid

    def get_dataframe_from_datasets(self):
        datasets = self.dbh.session.query(repo.DatasetAll).all()
        keys = ['data_name', 'weighted_mean', 'coefficient_variation',
                'fpskew', 'kurtosis', 'entropy', 'metric_time']
        dicts = []

        for dset in datasets:
            this_dict = {}
            for key in keys:
                value = getattr(dset, key)
                this_dict[key] = value
            dicts.append(this_dict)

        frame = pd.DataFrame(dicts)
        frame = frame[keys]

        sframe = pd.DataFrame()
        sframe['data_name'] = frame['data_name']
        sframe['description'] = ''

        pdb.set_trace()
        return dicts


    @staticmethod
    def get_nested_dict_frame(nested):
        frame = pd.DataFrame.from_dict({(i, j): nested[i][j]
                                        for i in nested.keys()
                                        for j in nested[i].keys()},
                                       orient='index')
        return frame

    @staticmethod
    def get_data_in_tabulate_table(frame, format_='grid'):
        # Relavent formats: latex, latex_raw, latex_booktabs
        frame.index.names = ['sample']
        h = [frame.index.names[0] + '/' + frame.columns.names[0]] + list(map('\n'.join, frame.columns.tolist()))
        table = tabulate(frame, headers=h, tablefmt=format_)
        return table

    @staticmethod
    def get_lists_of_long_samples_from_lists_of_lists(lists_lists):
        long_samples = []

        for outer_list in lists_lists:
            arr = []
            for inner_list in outer_list:
                for item in inner_list:
                    arr.append(item)
            long_samples.append(arr)
        return long_samples

    def get_multi_index_from_keys(self):
        iterables = [self.alg_keys, self.pos_keys]
        index = pd.MultiIndex.from_product(iterables, names=['algorithms', 'positions'])
        return index

    @staticmethod
    def human_sort(array):
        def try_int_else_text(text):
            return int(text) if text.isdigit() else text

        def natural_keys(text):
            '''
            alist.sort(key=natural_keys) sorts in human order
            http://nedbatchelder.com/blog/200712/human_sorting.html
            (See Toothy's implementation in the comments)
            '''
            return [try_int_else_text(c) for c in re.split(r'(\d+)', text)]

        array.sort(key=natural_keys)
        return array

    def main(self):
        N = len(self.meta_algs.all())
        averaged_probs = self.get_average_of_samples(self.proportions_dict)
        averaged_t_scores = self.get_average_of_samples(self.t_scores_dict)

        index = self.get_multi_index_from_keys()

        results_list = self.get_lists_of_lists_from_nested_dicts(self.results_dict)
        proportions_list = self.get_lists_of_lists_from_nested_dicts(self.proportions_dict)
        t_scores_list = self.get_lists_of_lists_from_nested_dicts(self.t_scores_dict)

        stds_list = self.get_lists_of_lists_for_small_nested(self.stds_dict)
        means_list = self.get_lists_of_lists_for_small_nested(self.means_dict)
        averaged_probs_list = self.get_lists_of_lists_for_small_nested(averaged_probs)
        averaged_t_scores_list = self.get_lists_of_lists_for_small_nested(averaged_t_scores)

        long_results = self.get_lists_of_long_samples_from_lists_of_lists(results_list)
        long_proportions = self.get_lists_of_long_samples_from_lists_of_lists(proportions_list)
        long_t_scores = self.get_lists_of_long_samples_from_lists_of_lists(t_scores_list)

        results_frame = pd.DataFrame(long_results, columns=index)
        props_frame = pd.DataFrame(long_proportions, columns=index).round(2)
        t_scores_frame = pd.DataFrame(long_t_scores, columns=index).round(2)

        stds_frame = pd.DataFrame(stds_list, columns=self.alg_keys).round(2)
        means_frame = pd.DataFrame(means_list, columns=self.alg_keys).round(2)
        averaged_props_frame = pd.DataFrame(averaged_probs_list, columns=self.alg_keys).round(2)
        averaged_t_scores_frame = pd.DataFrame(averaged_t_scores_list, columns=self.alg_keys).round(2)

        results_latex = results_frame.to_latex()
        props_latex = props_frame.to_latex()
        t_scores_latex = t_scores_frame.to_latex()
        stds_latex = stds_frame.to_latex()
        means_latex = means_frame.to_latex()
        averaged_props_latex = averaged_props_frame.to_latex()
        averaged_t_scores_latex = averaged_t_scores_frame.to_latex()

        dataset_frame = self.get_dataframe_from_datasets()

        print('Placement results')
        print(results_latex)
        print('----------------------------------')

        print('Placement results proportion probabilities')
        print(props_latex)
        print('-----------------------------------')

        print('t scores of results')
        print(t_scores_latex)
        print('------------------------------------')

        print('Standard deviation within each alg/position combination')
        print(stds_latex)
        print('-------------------------------------')

        print('means of results')
        print(means_latex)
        print('---------------------------------------')

        print('Average of proportion probabilities')
        print(averaged_props_latex)
        print('------------------------------------')

        print('Average of t scores')
        print(averaged_t_scores_latex)
        print('--------------------------------------')

        pdb.set_trace()




