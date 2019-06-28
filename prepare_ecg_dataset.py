import numpy as np
import pickle
import os

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

def feature_fusion(dir, flag, feats):
    features = np.array([], np.float)
    labels = np.array([], np.int)
    
    for feat in feats:
        x, y = loaddataset(dir + flag + feat+ '.pickle')
        
        if not labels.size:
            labels = np.array(y, np.int)

        features = np.column_stack((features, x)) if features.size else x

    return features, labels


class feature_normalizer(object):
    def __init__(self, x, mode, **kwargs):
        self.mode = mode
        if mode == 'zero_mean_per_feature':
            self.std = np.std(x, axis=0)
            self.mean = np.mean(x, axis=0)
        elif mode == 'zero_mean_per_channel':
            self.std = np.std(np.reshape(x, (-1, np.shape(x)[2])), axis=0)
            self.mean = np.mean(np.reshape(x, (-1, np.shape(x)[2])), axis=0)
        elif mode == 'range_per_feature':
            self.min = np.min(x, axis=0)
            self.max = np.max(x, axis=0)
        elif mode == 'range_all_feature':
            self.min = np.min(x)
            self.max = np.max(x)
        return super().__init__(**kwargs)

    def transform(self, x):
        if self.mode == 'zero_mean_per_feature':
            return (x-self.std)/self.mean
        elif self.mode == 'zero_mean_per_channel':
            return (x-self.std)/self.mean
        elif self.mode == 'range_per_feature':
            return (x-self.min)/(self.max-self.min)
        elif self.mode == 'range_all_feature':
            return (x-self.min)/(self.max-self.min)

train_test_mode = 'intra'
datasetdir = 'dataset/' + train_test_mode + '/'

feats = {'rrintervals'}
print('rrintervals flatten ...')
if not os.path.exists('exps/rrintervals/flatten/train.pickle'):
    x, y = feature_fusion(datasetdir, 'train_', feats)

    savedataset(x, y, np.array([], np.int), 'exps/rrintervals/flatten/train.pickle')

    x, y = feature_fusion(datasetdir, 'valid_', feats)
    savedataset(x, y, np.array([], np.int), 'exps/rrintervals/flatten/valid.pickle')

    x, y = feature_fusion(datasetdir, 'test_', feats)
    savedataset(x, y, np.array([], np.int), 'exps/rrintervals/flatten/test.pickle')
    print('rrintervals flatten completed')

feats = {'raw'}
print('raw flatten ...')
if not os.path.exists('exps/raw/flatten/train.pickle'):
    x, y = feature_fusion(datasetdir, 'train_', feats)

    x = np.reshape(x, (-1, 2, 180)).transpose([0,2,1])
    #normalizer = feature_normalizer(x, 'zero_mean_per_channel')
    #x = normalizer.transform(x)
    x = np.reshape(x, (-1, 360))
    savedataset(x, y, np.array([], np.int), 'exps/raw/flatten/train.pickle')

    x, y = feature_fusion(datasetdir, 'valid_', feats)
    x = np.reshape(x, (-1, 2, 180)).transpose([0,2,1])
    #x = normalizer.transform(x)
    x = np.reshape(x, (-1, 360))
    savedataset(x, y, np.array([], np.int), 'exps/raw/flatten/valid.pickle')

    x, y = feature_fusion(datasetdir, 'test_', feats)
    x = np.reshape(x, (-1, 2, 180)).transpose([0,2,1])
    #x = normalizer.transform(x)
    x = np.reshape(x, (-1, 360))
    savedataset(x, y, np.array([], np.int), 'exps/raw/flatten/test.pickle')
    print('raw flatten completed')

feats = {'fft35'}
print('fft35 flatten ...')
if not os.path.exists('exps/fft35/flatten/train.pickle'):
    x, y = feature_fusion(datasetdir, 'train_', feats)

    normalizer = feature_normalizer(x, 'zero_mean_per_feature')
    x = normalizer.transform(x)
    savedataset(x, y, np.array([], np.int), 'exps/fft35/flatten/train.pickle')

    x, y = feature_fusion(datasetdir, 'valid_', feats)
    x = normalizer.transform(x)
    savedataset(x, y, np.array([], np.int), 'exps/fft35/flatten/valid.pickle')

    x, y = feature_fusion(datasetdir, 'test_', feats)
    x = normalizer.transform(x)
    savedataset(x, y, np.array([], np.int), 'exps/fft35/flatten/test.pickle')
    print('fft35 flatten completed')

feats = {'waveletdb1lvl3'}
print('waveletdb1lvl3 flatten ...')
if not os.path.exists('exps/waveletdb1lvl3/flatten/train.pickle'):
    x, y = feature_fusion(datasetdir, 'train_', feats)

    x = np.reshape(x, (-1, 2, 23)).transpose([0,2,1]).reshape((-1, 46))
    normalizer = feature_normalizer(x, 'zero_mean_per_feature')
    x = normalizer.transform(x)
    savedataset(x, y, np.array([], np.int), 'exps/waveletdb1lvl3/flatten/train.pickle')

    x, y = feature_fusion(datasetdir, 'valid_', feats)
    x = np.reshape(x, (-1, 2, 23)).transpose([0,2,1]).reshape((-1, 46))
    x = normalizer.transform(x)
    savedataset(x, y, np.array([], np.int), 'exps/waveletdb1lvl3/flatten/valid.pickle')

    x, y = feature_fusion(datasetdir, 'test_', feats)
    x = np.reshape(x, (-1, 2, 23)).transpose([0,2,1]).reshape((-1, 46))
    x = normalizer.transform(x)
    savedataset(x, y, np.array([], np.int), 'exps/waveletdb1lvl3/flatten/test.pickle')
    print('waveletdb1lvl3 flatten completed')

feats = {'hos'}
print('hos flatten ...')
if not os.path.exists('exps/hos/flatten/train.pickle'):
    x, y = feature_fusion(datasetdir, 'train_', feats)

    x = np.reshape(x, (-1, 2, 20)).transpose([0,2,1]).reshape((-1, 40))
    normalizer = feature_normalizer(x, 'zero_mean_per_feature')
    x = normalizer.transform(x)
    savedataset(x, y, np.array([], np.int), 'exps/hos/flatten/train.pickle')

    x, y = feature_fusion(datasetdir, 'valid_', feats)
    x = np.reshape(x, (-1, 2, 20)).transpose([0,2,1]).reshape((-1, 40))
    x = normalizer.transform(x)
    savedataset(x, y, np.array([], np.int), 'exps/hos/flatten/valid.pickle')

    x, y = feature_fusion(datasetdir, 'test_', feats)
    x = np.reshape(x, (-1, 2, 20)).transpose([0,2,1]).reshape((-1, 40))
    x = normalizer.transform(x)
    savedataset(x, y, np.array([], np.int), 'exps/hos/flatten/test.pickle')
    print('hos flatten completed')

feats ={'waveletdb1lvl3uniformlbp'}
print('waveletdb1lvl3uniformlbp flatten ...')
if not os.path.exists('exps/waveletdb1lvl3uniformlbp/flatten/train.pickle'):
    x, y = feature_fusion(datasetdir, 'train_', feats)

    x = np.reshape(x, (-1, 2, 59)).transpose([0,2,1]).reshape((-1, 118))
    normalizer = feature_normalizer(x, 'range_all_feature')
    x = normalizer.transform(x)
    savedataset(x, y, np.array([], np.int), 'exps/waveletdb1lvl3uniformlbp/flatten/train.pickle')

    x, y = feature_fusion(datasetdir, 'valid_', feats)
    print(np.isnan(x).any())
    x = np.reshape(x, (-1, 2, 59)).transpose([0,2,1]).reshape((-1, 118))
    x = normalizer.transform(x)
    savedataset(x, y, np.array([], np.int), 'exps/waveletdb1lvl3uniformlbp/flatten/valid.pickle')

    x, y = feature_fusion(datasetdir, 'test_', feats)
    print(np.isnan(x).any())
    x = np.reshape(x, (-1, 2, 59)).transpose([0,2,1]).reshape((-1, 118))
    x = normalizer.transform(x)
    savedataset(x, y, np.array([], np.int), 'exps/waveletdb1lvl3uniformlbp/flatten/test.pickle')
    print('waveletdb1lvl3uniformlbp flatten completed')

feats ={'hermite'}
print('hermite flatten ...')
if not os.path.exists('exps/hermite/flatten/train.pickle'):
    x, y = feature_fusion(datasetdir, 'train_', feats)

    x = np.reshape(x, (-1, 2, 22)).transpose([0,2,1]).reshape((-1, 44))
    normalizer = feature_normalizer(x, 'range_per_feature')
    x = normalizer.transform(x)
    savedataset(x, y, np.array([], np.int), 'exps/hermite/flatten/train.pickle')

    x, y = feature_fusion(datasetdir, 'valid_', feats)
    x = np.reshape(x, (-1, 2, 22)).transpose([0,2,1]).reshape((-1, 44))
    x = normalizer.transform(x)
    savedataset(x, y, np.array([], np.int), 'exps/hermite/flatten/valid.pickle')

    x, y = feature_fusion(datasetdir, 'test_', feats)
    x = np.reshape(x, (-1, 2, 22)).transpose([0,2,1]).reshape((-1, 44))
    x = normalizer.transform(x)
    savedataset(x, y, np.array([], np.int), 'exps/hermite/flatten/test.pickle')
    print('hermite flatten completed')


#feats = {'raw'}
#print('raw leads...')
#if not os.path.exists('exps/raw/leads/train.pickle'):
#    x, y = feature_fusion(datasetdir, 'train_', feats)
#    x = np.reshape(x, (-1, 2, 180)).transpose([0,2,1])

#    normalizer = feature_normalizer(x, 'zero_mean_per_channel')
#    x = normalizer.transform(x)

#    savedataset(x, y, np.array([], np.int), 'exps/raw/leads/train.pickle')

#    x, y = feature_fusion(datasetdir, 'valid_', feats)
#    x = np.reshape(x, (-1, 2, 180)).transpose([0,2,1])
#    x = normalizer.transform(x)
#    savedataset(x, y, np.array([], np.int), 'exps/raw/leads/valid.pickle')

#    x, y = feature_fusion(datasetdir, 'test_', feats)
#    x = np.reshape(x, (-1, 2, 180)).transpose([0,2,1])
#    x = normalizer.transform(x)
#    savedataset(x, y, np.array([], np.int), 'exps/raw/leads/test.pickle')






