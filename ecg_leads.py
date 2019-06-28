import numpy as np
import pickle
import os

def loaddataset(path):
    with open(path, 'rb') as handle:
        dict = pickle.load(handle)
        return dict['x'], dict['y'], dict['lens']

def savedataset(x, y, lens, path):
    dir = os.path.dirname(os.path.abspath(path))
    if not os.path.exists(dir):
        os.makedirs(dir)

    with open(path, 'wb') as handle:
        pickle.dump({'x': x, 'y': y, 'lens': lens}, handle)


train_test_mode = 'intra'
datasetdir = 'dataset/' + train_test_mode + '/'

feats = {'raw'}
print('raw leads ...')
if not os.path.exists('exps/raw/leads/train.pickle'):
    if os.path.exists('exps/raw/flatten/train.pickle'):
        x, y, lens = loaddataset('exps/raw/flatten/train.pickle')
        x = np.reshape(x, (-1, 2, 180)).transpose([0,2,1])   
        savedataset(x, y, np.array([], np.int), 'exps/raw/leads/train.pickle')

        x, y, lens = loaddataset('exps/raw/flatten/valid.pickle')
        x = np.reshape(x, (-1, 2, 180)).transpose([0,2,1])
        savedataset(x, y, np.array([], np.int), 'exps/raw/leads/valid.pickle')

        x, y, lens = loaddataset('exps/raw/flatten/test.pickle')
        x = np.reshape(x, (-1, 2, 180)).transpose([0,2,1])
        savedataset(x, y, np.array([], np.int), 'exps/raw/leads/test.pickle')

feats = {'waveletdb1lvl3'}
print('waveletdb1lvl3 leads ...')
if not os.path.exists('exps/waveletdb1lvl3/leads/train.pickle'):
    if os.path.exists('exps/waveletdb1lvl3/flatten/train.pickle'):
        x, y, lens = loaddataset('exps/waveletdb1lvl3/flatten/train.pickle')
        x = np.reshape(x, (-1, 2, 23)).transpose([0,2,1])

        savedataset(x, y, np.array([], np.int), 'exps/waveletdb1lvl3/leads/train.pickle')

        x, y, lens = loaddataset('exps/waveletdb1lvl3/flatten/valid.pickle')
        x = np.reshape(x, (-1, 2, 23)).transpose([0,2,1])
        savedataset(x, y, np.array([], np.int), 'exps/waveletdb1lvl3/leads/valid.pickle')

        x, y, lens = loaddataset('exps/waveletdb1lvl3/flatten/test.pickle')
        x = np.reshape(x, (-1, 2, 23)).transpose([0,2,1])
        savedataset(x, y, np.array([], np.int), 'exps/waveletdb1lvl3/leads/test.pickle')

feats = {'raw'}
print('raw leads ...')
if not os.path.exists('exps/raw/leads/train.pickle.balance'):
    if os.path.exists('exps/raw/flatten/train.pickle.balance'):
        x, y, lens = loaddataset('exps/raw/flatten/train.pickle.balance')
        x = np.reshape(x, (-1, 2, 180)).transpose([0,2,1])   
        savedataset(x, y, np.array([], np.int), 'exps/raw/leads/train.pickle.balance')

        x, y, lens = loaddataset('exps/raw/flatten/valid.pickle.balance')
        x = np.reshape(x, (-1, 2, 180)).transpose([0,2,1])
        savedataset(x, y, np.array([], np.int), 'exps/raw/leads/valid.pickle.balance')

        x, y, lens = loaddataset('exps/raw/flatten/test.pickle.balance')
        x = np.reshape(x, (-1, 2, 180)).transpose([0,2,1])
        savedataset(x, y, np.array([], np.int), 'exps/raw/leads/test.pickle.balance')

feats = {'waveletdb1lvl3'}
print('waveletdb1lvl3 leads ...')
if not os.path.exists('exps/waveletdb1lvl3/leads/train.pickle.balance'):
    if os.path.exists('exps/waveletdb1lvl3/flatten/train.pickle.balance'):
        x, y, lens = loaddataset('exps/waveletdb1lvl3/flatten/train.pickle.balance')
        x = np.reshape(x, (-1, 2, 23)).transpose([0,2,1])

        savedataset(x, y, np.array([], np.int), 'exps/waveletdb1lvl3/leads/train.pickle.balance')

        x, y, lens = loaddataset('exps/waveletdb1lvl3/flatten/valid.pickle.balance')
        x = np.reshape(x, (-1, 2, 23)).transpose([0,2,1])
        savedataset(x, y, np.array([], np.int), 'exps/waveletdb1lvl3/leads/valid.pickle.balance')

        x, y, lens = loaddataset('exps/waveletdb1lvl3/flatten/test.pickle.balance')
        x = np.reshape(x, (-1, 2, 23)).transpose([0,2,1])
        savedataset(x, y, np.array([], np.int), 'exps/waveletdb1lvl3/leads/test.pickle.balance')
