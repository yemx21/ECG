# -*- coding:utf-8 -*-
import pandas as pd
import numpy as np
import math as math
import os
import time
from sklearn.svm import SVC
from models.cpuutils import *
from sklearn.externals import joblib
import tensorflow as tf
import time


def run_SVM(timesteps, numclasses, train_x, train_y, train_lens, valid_x, valid_y, valid_lens, test_x, test_y, test_lens, modeldir, batchsize, maxepochs, patience, expsnum, **kwargs):

    tests_y = []
    preds_y = []
    epoches = []

    class_weights = {}
    for c in range(numclasses):
        class_weights.update({c:len(train_y) / float(np.count_nonzero(train_y == c))})

    arg_svm_kernel = kwargs['svm_kernel']
    svmkernel = 'linear' if arg_svm_kernel is None else arg_svm_kernel
    print(svmkernel)
    for exp in range(expsnum):
        print('exp ', str(exp), ': SVM...') 
        modelfile = modeldir + str(exp) + '_' + svmkernel + '.pkl'
        if not os.path.exists(modelfile):
            clf = SVC(random_state=int(time.time()), kernel=svmkernel, verbose=True, cache_size=400, class_weight=class_weights, max_iter=500000)
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
    logfile = modeldir + "svm.log"
    savemetrics(logfile, epoches, test_yo, pred_yo, target_names)

def test_SVM(timesteps, numclasses, test_x, test_y, test_lens, modeldir, batchsize, expsnum, **kwargs):

    tests_y = []
    preds_y = []
    paramnums = []
    durations = []

    arg_svm_kernel = kwargs['svm_kernel']
    svmkernel = 'linear' if arg_svm_kernel is None else arg_svm_kernel

    for exp in range(expsnum):
        print('exp ', str(exp), ': SVM...')
        modelfile = modeldir + str(exp) + '_' + svmkernel + '.pkl'
        clf = joblib.load(modelfile)
        paramnum = clf.support_vectors_.size
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
    logfile = modeldir + "svm_"+ svmkernel + ".benchmark"
    savebenchmark(logfile, paramnums, durations, test_yo, pred_yo, target_names)
