import numpy as np
import tensorflow as tf
from tensorflow.contrib.keras import backend as K
import os
import struct
import matplotlib.pyplot as plt
from tensorflow.python.keras.models import Sequential, load_model
from tensorflow.python.keras.layers import LSTM, Dense, Activation, Bidirectional, Dropout, GRU
from sklearn.metrics import confusion_matrix, classification_report
from tensorflow.python.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau

from collections import Counter
import models.utils as utils
import time

from sklearn.utils import class_weight


def build_model(time_steps, num_classes, inputdim):
    model = Sequential()
    model.add(Bidirectional(GRU(10, return_sequences=True), input_shape=(time_steps, inputdim)))
    model.add(Bidirectional(GRU(20, return_sequences=False)))
    model.add(Dropout(0.5))
    model.add(Dense(num_classes))
    model.add(Activation('softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.summary()
    return model

def train_model(train_x, train_y, valid_x, valid_y, test_x, test_y, time_steps, num_classes, class_weights, modelfile, batchsize, maxepochs, patience):
    inputdim = np.shape(train_x)[2]

    model = build_model(time_steps, num_classes, inputdim)
    checkpointer = utils.AutoCheckStopping(filepath=modelfile, verbose=1, save_best_only = True, patience = patience)
    try:
        model.fit(train_x, train_y, batch_size=batchsize, shuffle=True, epochs=maxepochs, validation_data =[test_x, test_y], callbacks= [checkpointer], class_weight = class_weights)
        bestmodel = build_model(time_steps, num_classes, inputdim)
        bestmodel.load_weights(modelfile)
        predict = bestmodel.predict(test_x, verbose=1, batch_size= batchsize)
    except KeyboardInterrupt:
        pass
    return predict, test_y, checkpointer.stopped_epoch

def run_BIGRU(timesteps, numclasses, train_x, train_y, train_lens, valid_x, valid_y, valid_lens, test_x, test_y, test_lens, modeldir, batchsize, maxepochs, patience, expsnum):
    K.set_session(utils.get_session())

    print('start class_weights...')

    class_weights =  class_weight.compute_class_weight('balanced', np.unique(train_y), train_y)

    print(class_weights)

    train_y = utils.one_hot(train_y, numclasses)
    valid_y = utils.one_hot(valid_y, numclasses)
    test_y = utils.one_hot(test_y, numclasses)

    preds = []
    tests = []
    epochs = []

    print('start...')

    for exp in range(1, 1 + expsnum):
        modelfile = modeldir + str(exp) +".hdf5"
        predict_y, test_y, stopped_epoch = train_model(train_x, train_y, valid_x, valid_y, test_x, test_y, timesteps, numclasses, class_weights, modelfile, batchsize, maxepochs, patience)
        preds.extend(predict_y)
        tests.extend(test_y)
        epochs.append(stopped_epoch)

    test_yo = np.argmax(tests, 1)
    pred_yo = np.argmax(preds, 1)

    target_names = ['Class 1', 'Class 2', 'Class 3', 'Class 4', 'Class 5']
    logfile = modeldir + "bigru.log"
    utils.savemetrics(logfile, epochs, test_yo, pred_yo, target_names)

def test_model(test_x, test_y, time_steps, num_classes, modelfile, batchsize=512):
    inputdim = np.shape(test_x)[2]
    try:
        bestmodel = build_model(time_steps, num_classes, inputdim)
        bestmodel.load_weights(modelfile)
        start_time = time.time()
        predict = bestmodel.predict(test_x, 1, verbose=1, batch_size= batchsize)
        duration = (time.time() - start_time) * 1000.0/ float(np.shape(test_x)[0])
    except KeyboardInterrupt:
        pass
    return predict, test_y, bestmodel.count_params(), duration

def test_BIGRU(timesteps, numclasses, test_x, test_y, test_lens, modeldir,  expsnum):
    K.set_session(utils.get_session())

    test_y = utils.one_hot(test_y, numclasses)

    preds = []
    tests = []
    paramnums = []
    durations = []

    for exp in range(1, 1 + expsnum):
        modelfile = modeldir + str(exp) + ".hdf5"
        predict_y, test_y, paramnum, duration = test_model(test_x, test_y, timesteps, numclasses, modelfile)
        preds.extend(predict_y)
        tests.extend(test_y)
        paramnums.append(paramnum)
        durations.append(duration)

    test_yo = np.argmax(tests, 1)
    pred_yo = np.argmax(preds, 1)

    target_names = ['Class 1', 'Class 2', 'Class 3', 'Class 4', 'Class 5']
    logfile = modeldir + "bigru.benchmark"
    utils.savebenchmark(logfile, paramnums, durations, test_yo, pred_yo, target_names)
