import pickle
import os
import numpy as np
from imblearn.combine import SMOTEENN
from collections import Counter

def loaddataset(path):
    with open(path, 'rb') as handle:
        dict = pickle.load(handle)
        return dict['x'], dict['y']

def savedataset(x, y, lens, path):
    dir = os.path.dirname(os.path.abspath(path))
    if not os.path.exists(dir):
        os.makedirs(dir)

    with open(path, 'wb') as handle:
        pickle.dump({'x': x, 'y': y, 'lens': lens}, handle)

def balance(x, y, randomstate=None, **kwargs):
    sm = SMOTEENN(random_state=randomstate, n_jobs=3, n_neighbors=kwargs['neighbors'])  
    print('dataset shape {}'.format(Counter(y)))
    print('Resampling...')
    rx, ry= sm.fit_sample(x,y)
    print('Resampled dataset shape {}'.format(Counter(ry)))
    return rx, ry


def balancedataset(path, randomstate=None, **kwargs):
    if not os.path.exists(path + '.balance'):
        if os.path.exists(path):
            x, y = loaddataset(path)
            rx, ry = balance(x,y, randomstate, **kwargs)
            savedataset(rx, ry, np.array([], np.int), path + '.balance')

datasetdir = 'exps/'

features = ['raw', 'fft35', 'rrintervals', 'hos', 'waveletdb1lvl3', 'waveletdb1lvl3uniformlbp', 'hermite']
featuremodes =['flatten', 'leads']
randomstate = 123

#train
for feature in features:
    for featuremode in featuremodes:
        train_path = datasetdir + feature + '/' + featuremode + '/train.pickle'
        balancedataset(train_path, randomstate, neighbors=5)

#valid
for feature in features:
    for featuremode in featuremodes:
        valid_path = datasetdir + feature + '/' + featuremode + '/valid.pickle'
        balancedataset(valid_path, randomstate, neighbors=5)

#test
for feature in features:
    for featuremode in featuremodes:
        test_path = datasetdir + feature + '/' + featuremode + '/test.pickle'
        balancedataset(test_path, randomstate, neighbors=5)

