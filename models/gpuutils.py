import numpy as np
import os
import math
import tensorflow as tf
import warnings
import matplotlib.pyplot as plt
from tensorflow.python.keras.callbacks import Callback
from tensorflow.contrib.keras import backend as K
from collections import Counter
from models.cpuutils import *

def get_session(gpu_fraction=0.3):
    num_threads = os.environ.get('OMP_NUM_THREADS')
    if os.getenv('CUDA_VISIBLE_DEVICES') == -1:
        return tf.Session(config=tf.ConfigProto(device_count={'CPU' : 1, 'GPU' : 0}, allow_soft_placement=True, log_device_placement=True))

    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=gpu_fraction)

    if num_threads:
        return tf.Session(config=tf.ConfigProto(
            gpu_options=gpu_options, intra_op_parallelism_threads=num_threads))
    else:
        return tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))

class AutoCheckStopping(Callback):
    def __init__(self, filepath, monitor='val_loss', verbose=0,
                 save_best_only=False, save_weights_only=False, min_delta=0, patience=0,
                 mode='auto', period=1):
        super(AutoCheckStopping, self).__init__()
        self.monitor = monitor
        self.verbose = verbose
        self.filepath = filepath
        self.save_best_only = save_best_only
        self.save_weights_only = save_weights_only
        self.period = period
        self.epochs_since_last_save = 0
        self.patience = patience
        self.min_delta = min_delta
        self.wait = 0
        self.stopped_epoch = 0

        if mode not in ['auto', 'min', 'max']:
            warnings.warn('AutoCheckStopping mode %s is unknown, '
                          'fallback to auto mode.' % (mode),
                          RuntimeWarning)
            mode = 'auto'

        if mode == 'min':
            self.monitor_op = np.less
            self.best = np.Inf
        elif mode == 'max':
            self.monitor_op = np.greater
            self.best = -np.Inf
        else:
            if 'acc' in self.monitor or self.monitor.startswith('fmeasure'):
                self.monitor_op = np.greater
                self.best = -np.Inf
            else:
                self.monitor_op = np.less
                self.best = np.Inf
        if self.monitor_op == np.greater:
            self.min_delta *= 1
        else:
            self.min_delta *= -1

    def on_epoch_end(self, epoch, logs=None):
        current = logs.get(self.monitor)
        if current is None:
            warnings.warn(
                'Early stopping conditioned on metric `%s` '
                'which is not available. Available metrics are: %s' %
                (self.monitor, ','.join(list(logs.keys()))), RuntimeWarning
            )
            return
        if self.monitor_op(current - self.min_delta, self.best):
            self.best1 = current
            self.wait = 0
        else:
            if self.wait >= self.patience:
                self.stopped_epoch = epoch
                self.model.stop_training = True
            self.wait += 1

        logs = logs or {}
        self.epochs_since_last_save += 1
        if self.epochs_since_last_save >= self.period:
            self.epochs_since_last_save = 0
            filepath = self.filepath.format(epoch=epoch, **logs)
            if self.save_best_only:
                current = logs.get(self.monitor)
                if current is None:
                    warnings.warn('Can save best model only with %s available, '
                                  'skipping.' % (self.monitor), RuntimeWarning)
                else:
                    if self.monitor_op(current, self.best):
                        if self.verbose > 0:
                            print('Epoch %05d: %s improved from %0.5f to %0.5f,'
                                  ' saving model to %s'
                                  % (epoch, self.monitor, self.best,
                                     current, filepath))
                        self.best = current
                        if self.save_weights_only:
                            self.model.save_weights(filepath, overwrite=True)
                        else:
                            self.model.save(filepath, overwrite=True)
                    else:
                        if self.verbose > 0:
                            print('Epoch %05d: %s did not improve' %
                                  (epoch, self.monitor))
            else:
                if self.verbose > 0:
                    print('Epoch %05d: saving model to %s' % (epoch, filepath))
                if self.save_weights_only:
                    self.model.save_weights(filepath, overwrite=True)
                else:
                    self.model.save(filepath, overwrite=True)

    def on_train_begin(self, logs=None):
        # Allow instances to be re-used
        self.wait = 0
        self.stopped_epoch = 0
        self.best1 = np.Inf if self.monitor_op == np.less else -np.Inf

    def on_train_end(self, logs=None):
        if self.stopped_epoch > 0 and self.verbose > 0:
            print('Epoch %05d: early stopping' % (self.stopped_epoch))

class RandomGenerator(object):
    def __init__(self, x, y, uy, bs, encodey = False, randomstate = None, **kwargs):
        self.x = x
        self.y = y
        self.uy = uy
        self.bs = bs
        self.rng = np.random.RandomState(randomstate)

        self.inds = np.arange(0, len(self.y))

        #self.cw =  class_weight.compute_class_weight('balanced', uy, y)
        
        if encodey:
            self.y = encode_target(y, len(uy))

        return super().__init__(**kwargs)


    def get_batch(self):
        self.rng.shuffle(self.inds)

        uinds = self.inds[:self.bs]
        x = self.x[uinds]
        y = self.y[uinds]

        return x, y

class BalancedGenerator(object):
    def __init__(self, x, y, uy, preclassbatchsize, encodey = False, randomstate = None, **kwargs):
        self.x = x
        self.y = y
        self.uy = uy
        self.pcbs = preclassbatchsize
        self.bs = preclassbatchsize * len(uy)
        self.rng = np.random.RandomState(randomstate)
        self.ind = {}

        for u in uy:
            self.ind[u] = np.where(y==u)[0]
        
        if encodey:
            self.y = encode_target(y, len(uy))

        xs = np.shape(self.x)
        ys = np.shape(self.y)

        if len(xs)==2:
            self.xs = (self.bs, xs[1])
        elif len(xs)==3:
            self.xs = (self.bs, xs[1], xs[2])

        if len(ys)==1:
            self.ys = (self.bs,)
        elif len(ys)==2:
            self.ys = (self.bs, ys[1])


        return super().__init__(**kwargs)


    def get_batch(self):
        x = np.zeros(self.xs)
        y = np.zeros(self.ys)
        iu = 0
        for u in self.uy:
            uinds = self.rng.choice(self.ind[u], self.pcbs)
            x[iu:iu + self.pcbs] = self.x[uinds]
            y[iu:iu + self.pcbs] = self.y[uinds]
            iu += self.pcbs

        return x, y

def SourceGenerator(generator):
    while True:
        yield generator.get_batch()
            




        
