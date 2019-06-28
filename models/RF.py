# -*- coding:utf-8 -*-
import numpy as np
import math as math
import os
import time
from sklearn.ensemble import RandomForestClassifier
from models.cpuutils import *
from sklearn.externals import joblib
import time

def run_RF(timesteps, numclasses, train_x, train_y, train_lens, valid_x, valid_y, valid_lens, test_x, test_y, test_lens, modeldir, batchsize, maxepochs, patience, expsnum, **kwargs):

    tests_y = []
    preds_y = []
    epoches = []

    for exp in range(expsnum):
        print('exp ', str(exp), ': RF...') 
        modelfile = modeldir + str(exp) +".pkl"
        if not os.path.exists(modelfile):
            clf = RandomForestClassifier(n_estimators = 30, max_depth=20, n_jobs=3, random_state=int(time.time()), verbose=True)
            clf.fit(train_x, train_y)
            joblib.dump(clf, modelfile)
        else:
            clf = joblib.load(modelfile)
        pred_y = clf.predict(test_x)
        tests_y.extend(test_y)
        preds_y.extend(pred_y)
        epoches.append(30)
        print('exp', exp, np.mean(test_y == pred_y))

    test_yo= np.reshape(tests_y, [-1])
    pred_yo= np.reshape(preds_y, [-1])

    target_names = ['Class 1', 'Class 2', 'Class 3', 'Class 4', 'Class 5']
    logfile = modeldir + "rf.log"
    savemetrics(logfile, epoches, test_yo, pred_yo, target_names)

def count_paramnum(model):
    paramnum = 0
    for i in range(model.n_estimators):
        tree = model[i].tree_
        paramnum = paramnum + tree.children_left.size + tree.children_right.size + tree.feature.size + tree.threshold.size + tree.value.size
    return paramnum
  
def test_RF(timesteps, numclasses, test_x, test_y, test_lens, modeldir, batchsize, expsnum, **kwargs):
    tests_y = []
    preds_y = []
    paramnums = []
    durations = []

    for exp in range(expsnum):
        modelfile = modeldir + str(exp) +".pkl"
        clf = joblib.load(modelfile)
        paramnum = count_paramnum(clf)
        start_time = time.time()
        pred_y = clf.predict(test_x)
        duration = (time.time() - start_time) * 1000.0/ float(np.shape(test_x)[0])
        tests_y.extend(test_y)
        preds_y.extend(pred_y)
        print('exp', exp, np.mean(test_y == pred_y))
        paramnums.append(paramnum)
        durations.append(duration)

    test_yo= np.reshape(tests_y, [-1])
    pred_yo= np.reshape(preds_y, [-1])

    target_names = ['Class 1', 'Class 2', 'Class 3', 'Class 4', 'Class 5']
    logfile = modeldir + "rf.benchmark"
    savebenchmark(logfile, paramnums, durations, test_yo, pred_yo, target_names)
