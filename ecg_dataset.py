import numpy as np
import pickle
import os
from ecgfeatures import extractfeatures

def loaddataset(path):
    with open(path, 'rb') as handle:
        dict = pickle.load(handle)
        return dict['samples'], dict['labels'], dict['sindrs'], dict['rinds'], dict['rlocs']

def savedataset(x, y, path):
    with open(path, 'wb') as handle:
        pickle.dump({'x': x, 'y': y}, handle)

def preparedataset(path, train_test_mode, flag, forces):
    samples, labels, sindrs, rinds, rlocs = loaddataset(path)

    datasetdir = 'dataset/' + train_test_mode + '/'

    #raw signals
    datapath = datasetdir + flag + 'raw.pickle'
    if not os.path.exists(datapath):
        leads = {'MLII', 'V1'}
        feats = {'raw'}
        features = extractfeatures(samples, sindrs, rinds, rlocs, leads, feats)
        savedataset(features, labels, datapath)

    #fft 35
    datapath = datasetdir + flag + 'fft35.pickle'
    if not os.path.exists(datapath) or 'fft' in forces:
        leads = {'MLII', 'V1'}
        feats = {'fft'}
        features = extractfeatures(samples, sindrs, rinds, rlocs, leads, feats, fftn=35)
        savedataset(features, labels, datapath)

    #wavelet
    datapath = datasetdir + flag + 'waveletdb1lvl3.pickle'
    if not os.path.exists(datapath) or 'wavelet' in forces:
        leads = {'MLII', 'V1'}
        feats = {'wavelet'}
        features = extractfeatures(samples, sindrs, rinds, rlocs, leads, feats, wavelet_family='db1', wavelet_level=3)
        savedataset(features, labels, datapath)
    
    #rr_intervals
    datapath = datasetdir + flag+ 'rrintervals.pickle'
    if not os.path.exists(datapath) or 'rr_intervals' in forces:
        leads = {'MLII'}
        feats = {'rr_intervals'}
        features = extractfeatures(samples, sindrs, rinds, rlocs, leads, feats, normintervals=False)
        savedataset(features, labels, datapath)

    #hos
    datapath = datasetdir + flag+ 'hos.pickle'
    if not os.path.exists(datapath) or 'hos' in forces:    
        leads = {'MLII', 'V1'}
        feats = {'hos'}
        features = extractfeatures(samples, sindrs, rinds, rlocs, leads, feats, hos_region=4)
        savedataset(features, labels, datapath)

    #wavelet_uniform_lbp
    datapath = datasetdir + flag+ 'waveletdb1lvl3uniformlbp.pickle'
    if not os.path.exists(datapath) or 'wavelet_uniform_lbp' in forces:  
        leads = {'MLII', 'V1'}
        feats = {'wavelet_uniform_lbp'}
        features = extractfeatures(samples, sindrs, rinds, rlocs, leads, feats, wavelet_family='db1', wavelet_level=3)
        savedataset(features, labels, datapath)

    #hermite
    datapath = datasetdir + flag+ 'hermite.pickle'
    if not os.path.exists(datapath) or 'hermite' in forces:  
        leads = {'MLII', 'V1'}
        feats = {'hermite'}
        features = extractfeatures(samples, sindrs, rinds, rlocs, leads, feats, hermites=[3,4,5,6])
        savedataset(features, labels, datapath)

    #iceemd
    #leads = {'MLII', 'V1'}
    #feats = {'iceemd'}
    #features = extractfeatures(samples, sindrs, rinds, rlocs, leads, feats)
    #savedataset(features, labels, datasetdir + 'train_iceemd.pickle') 

    #iceemd_cumulants_sample_entrop




if __name__ == "__main__":

    train_test_mode = 'intra'

    forces = []

    preparedataset('data/'+ train_test_mode + '/train_data.pickle', train_test_mode, 'train_', forces)
    preparedataset('data/'+ train_test_mode + '/valid_data.pickle', train_test_mode, 'valid_', forces)
    preparedataset('data/'+ train_test_mode + '/test_data.pickle', train_test_mode, 'test_', forces)


    

        


    