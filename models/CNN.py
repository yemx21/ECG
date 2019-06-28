import numpy as np
import tensorflow as tf
from tensorflow.contrib.keras import backend as K
import os
import matplotlib.pyplot as plt
from tensorflow.python.keras.models import Sequential, load_model
from tensorflow.python.keras.layers import Dense, Activation, Dropout, Flatten, Conv1D, BatchNormalization, MaxPool1D
from sklearn.metrics import confusion_matrix, classification_report
from tensorflow.python.keras.callbacks import EarlyStopping, ModelCheckpoint

from collections import Counter
import models.gpuutils as gpuutils
from models.cpuutils import *
import time

from sklearn.utils import class_weight

def deep_cnnblocks(inputdim, inputshape):
    if inputdim < 8:
        return (Conv1D(8,2,activation=tf.nn.relu, input_shape=inputshape, name='input'), BatchNormalization())
    elif inputdim < 16:
        return (Conv1D(16,2,activation=tf.nn.relu, input_shape=inputshape, name='input'), BatchNormalization())
    elif inputdim < 32:
        return (Conv1D(16,3,activation=tf.nn.relu, input_shape=inputshape, name='input'), Conv1D(16,3,activation=tf.nn.relu)
                , BatchNormalization(), MaxPool1D(2))
    elif inputdim < 64:
        return (Conv1D(16,3,activation=tf.nn.relu, input_shape=inputshape, name='input'), Conv1D(24,3,activation=tf.nn.relu)
                , BatchNormalization(), MaxPool1D(2))
    elif inputdim < 128:
        return (Conv1D(16,3,activation=tf.nn.relu, input_shape=inputshape, name='input'), Conv1D(16,3,activation=tf.nn.relu)
                , BatchNormalization(), MaxPool1D(3), Dropout(0.5),
                Conv1D(16,3,activation=tf.nn.relu), Conv1D(16,3,activation=tf.nn.relu)
                , BatchNormalization(), MaxPool1D(2))
    else:
        return (Conv1D(16,3,activation=tf.nn.relu, input_shape=inputshape, name='input'), Conv1D(16,3,activation=tf.nn.relu)
                , BatchNormalization(), MaxPool1D(3), Dropout(0.5),
                Conv1D(32,3,activation=tf.nn.relu), Conv1D(16,3,activation=tf.nn.relu)
                , BatchNormalization(), MaxPool1D(2))

def deep_units(inputdim):
    if inputdim < 8:
        return (64, 32)
    elif inputdim < 16:
        return (128, 32)
    elif inputdim < 32:
        return (256, 32)
    elif inputdim < 64:
        return (256, 64)
    elif inputdim < 128:
        return (512, 64)
    else:
        return (512, 128)

def build_model(inputdim, num_classes):
    model = Sequential()

    convs = deep_cnnblocks(inputdim, (inputdim, 1))
    units = deep_units(inputdim)

    for layer in convs:
        model.add(layer)

    model.add(Dropout(0.5))
    model.add(Flatten())

    for unit in units:
        model.add(Dense(unit, activation=tf.nn.relu))

    model.add(Dense(num_classes))
    model.add(Activation('softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.summary()
    return model

def train_model(inputdim, train_dataprovider, train_steps_per_epoch, valid_x, valid_y, test_x, test_y, time_steps, num_classes, modelfile, batchsize, maxepochs, patience):
    model = build_model(inputdim, num_classes)
    checkpointer = gpuutils.AutoCheckStopping(filepath=modelfile, verbose=1, save_best_only = True, patience = patience)
    try:
        model.fit_generator(train_dataprovider, steps_per_epoch=train_steps_per_epoch, epochs=maxepochs, validation_data =[valid_x, valid_y], callbacks= [checkpointer])
        bestmodel = build_model(inputdim, num_classes)
        bestmodel.load_weights(modelfile)
        predict = bestmodel.predict(test_x, verbose=1, batch_size= batchsize)
    except KeyboardInterrupt:
        pass
    return predict, test_y, checkpointer.stopped_epoch

def run_CNN(timesteps, numclasses, train_x, train_y, train_lens, valid_x, valid_y, valid_lens, test_x, test_y, test_lens, modeldir, batchsize, maxepochs, patience, expsnum, **kwargs):
    K.set_session(gpuutils.get_session())

    if len(np.shape(train_x))<3:
        train_x = np.expand_dims(train_x, 2)

    if len(np.shape(valid_x))<3:
        valid_x = np.expand_dims(valid_x, 2)

    if len(np.shape(test_x))<3:
        test_x = np.expand_dims(test_x, 2)

    valid_y = encode_target(valid_y, numclasses)
    test_y = encode_target(test_y, numclasses)

    preds = []
    tests = []
    epochs = []

    print('start...')

    unique_y =  list(range(numclasses))

    if kwargs.get('balancedgenerator')==True:
        train_dataprovider = gpuutils.SourceGenerator(gpuutils.BalancedGenerator(train_x, train_y, unique_y, int(batchsize/numclasses), True))
    else:
        train_dataprovider = gpuutils.SourceGenerator(gpuutils.RandomGenerator(train_x, train_y, unique_y, batchsize, True))
    
    inputdim = np.shape(train_x)[1]
    train_steps_per_epoch = int(np.shape(train_x)[0]/batchsize)

    for exp in range(1, 1 + expsnum):
        modelfile = modeldir + str(exp) +".hdf5"
        predict_y, test_y, stopped_epoch = train_model(inputdim, train_dataprovider, train_steps_per_epoch, valid_x, valid_y, test_x, test_y, timesteps, numclasses, modelfile, batchsize, maxepochs, patience)
        preds.extend(predict_y)
        tests.extend(test_y)
        epochs.append(stopped_epoch)

    test_yo = np.argmax(tests, 1)
    pred_yo = np.argmax(preds, 1)

    target_names = ['Class 1', 'Class 2', 'Class 3', 'Class 4', 'Class 5']
    logfile = modeldir + "cnn.log"
    savemetrics(logfile, epochs, test_yo, pred_yo, target_names)

def test_model(test_x, test_y, time_steps, num_classes, modelfile):
    inputdim = np.shape(test_x)[1]
    try:
        bestmodel = build_model(inputdim, num_classes)
        bestmodel.load_weights(modelfile)
        start_time = time.time()
        predict = bestmodel.predict(test_x, 1, verbose=1)
        duration = (time.time() - start_time) * 1000.0/ float(np.shape(test_x)[0])
    except KeyboardInterrupt:
        pass
    return predict, test_y, bestmodel.count_params(), duration

def test_CNN(timesteps, numclasses, test_x, test_y, test_lens, modeldir, batchsize, expsnum, **kwargs):
    K.set_session(gpuutils.get_session())

    test_y = encode_target(test_y, numclasses)

    if len(np.shape(test_x))<3:
        test_x = np.expand_dims(test_x, 2)


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
    logfile = modeldir + "cnn.benchmark"
    savebenchmark(logfile, paramnums, durations, test_yo, pred_yo, target_names)
