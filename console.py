import pickle
from models.SVM import *
from models.RF import *
from models.DNN import *
from models.CNN import *
from models.GRU import *
from models.LSTM import *
import os

def loaddataset(path):
    with open(path, 'rb') as handle:
        dict = pickle.load(handle)
        return dict['x'], dict['y'], dict['lens']


def run(method, valstr, datastr, timesteps, numclasses, batchsize = 128, maxepoch = 24, patience = 4, expsnum = 5, balanced = False, **kwargs):

    if balanced:
        datadir = "exps/" + valstr + "/" + datastr + "/"
        modeldir = "models/balanced/" + method + "/" + valstr + "/"
    else:
        datadir = "exps/" + valstr + "/" + datastr + "/"
        modeldir = "models/imbalanced/" + method + "/" + valstr + "/"

    if not os.path.exists(modeldir):
        os.makedirs(modeldir)

    if balanced:
        train_x, train_y, train_lens = loaddataset(datadir + "train.pickle.balance")
        valid_x, valid_y, valid_lens = loaddataset(datadir + "test.pickle.balance")
    else:
        train_x, train_y, train_lens = loaddataset(datadir + "train.pickle")
        valid_x, valid_y, valid_lens = loaddataset(datadir + "test.pickle")

    
    test_x, test_y, test_lens = loaddataset(datadir + "test.pickle")


    if method == 'svm':
        run_SVM(timesteps, numclasses, train_x, train_y, train_lens, valid_x, valid_y, valid_lens, test_x, test_y, test_lens, modeldir, batchsize, maxepoch, patience, expsnum, **kwargs)
    elif method == 'rf':
        run_RF(timesteps, numclasses, train_x, train_y, train_lens, valid_x, valid_y, valid_lens, test_x, test_y, test_lens, modeldir, batchsize, maxepoch, patience, expsnum, **kwargs)
    elif method == 'dnn':
        run_DNN(timesteps, numclasses, train_x, train_y, train_lens, valid_x, valid_y, valid_lens, test_x, test_y, test_lens, modeldir, batchsize, maxepoch, patience, expsnum, **kwargs)
    elif method == 'cnn':
        run_CNN(timesteps, numclasses, train_x, train_y, train_lens, valid_x, valid_y, valid_lens, test_x, test_y, test_lens, modeldir, batchsize, maxepoch, patience, expsnum, **kwargs)
    elif method == 'gru':
        run_GRU(timesteps, numclasses, train_x, train_y, train_lens, valid_x, valid_y, valid_lens, test_x, test_y, test_lens, modeldir, batchsize, maxepoch, patience, expsnum, **kwargs)
    elif method == 'lstm':
        run_LSTM(timesteps, numclasses, train_x, train_y, train_lens, valid_x, valid_y, valid_lens, test_x, test_y, test_lens, modeldir, batchsize, maxepoch, patience, expsnum, **kwargs)


def bench(method, valstr, datastr, timesteps, numclasses, batchsize = 512, expsnum = 5, balanced = False, **kwargs):

    if balanced:
        datadir = "exps/" + valstr + "/" + datastr + "/"
        modeldir = "models/balanced/" + method + "/" + valstr + "/"
    else:
        datadir = "exps/" + valstr + "/" + datastr + "/"
        modeldir = "models/imbalanced/" + method + "/" + valstr + "/"


    if os.path.exists(modeldir):
        if kwargs.get('test_usebalance') is None:
            test_x, test_y, test_lens = loaddataset(datadir + "test.pickle")
        else:
            test_x, test_y, test_lens = loaddataset(datadir + "test.pickle.balance")

        if method == 'svm':
            test_SVM(timesteps, numclasses, test_x, test_y, test_lens, modeldir, batchsize, expsnum, **kwargs)
        elif method == 'rf':
            test_RF(timesteps, numclasses, test_x, test_y, test_lens, modeldir, batchsize, expsnum, **kwargs)
        elif method == 'dnn':
            test_DNN(timesteps, numclasses, test_x, test_y, test_lens, modeldir, batchsize, expsnum, **kwargs)
        elif method == 'cnn':
            test_CNN(timesteps, numclasses, test_x, test_y, test_lens, modeldir, batchsize, expsnum, **kwargs)
        elif method == 'gru':
            test_GRU(timesteps, numclasses, test_x, test_y, test_lens, modeldir, batchsize, expsnum, **kwargs)
        elif method == 'lstm':
            test_LSTM(timesteps, numclasses, test_x, test_y, test_lens, modeldir, batchsize, expsnum, **kwargs)
