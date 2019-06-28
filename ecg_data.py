import numpy as np
import wfdb
import os
from wfdb import processing
import scipy
import pickle
import operator
from collections import Counter
from imblearn import under_sampling
from sklearn import preprocessing
from imblearn import ensemble
from scipy import io as scio
from scipy.signal import medfilt
from tqdm import tqdm
import matplotlib.pyplot as plt

symtables = {'0':'N', '1':'S', '2':'V', '3':'F', '4':'Q'} 

symrefs ={'N': ['N', 'L', 'R'], 'S':['A', 'a', 'J', 'S', 'e', 'j'], 'V':['V', 'E'], 'F': ['F'], 'Q':['f', 'Q', 'P']}

labrefs ={'N': 0, 'S':1, 'V':2, 'F': 3, 'Q': 4}

def stat(y):
    yy =np.array(y).reshape(-1).tolist()
    counter = Counter(yy)
    for key, count in counter.items():
        print(symtables[str(key)], ':', count)

def statstr(y):
    yy =np.array(y).reshape(-1).tolist()
    counter = Counter(yy)
    for key, count in counter.items():
        print(str(key), ':', count)

def filterrds_byleads(rdnames, leads, verbose=False):
    filtered_rdnames = []
    for rdname in rdnames:
        print(rdname)
        sig, fields = wfdb.rdsamp('mitdb/'+rdname, channels='all')
        
        drop = False
        if verbose:
            print(fields['sig_name'])

        for lead in leads:
            if not lead in fields['sig_name']:
                drop = True

        if not drop:
            filtered_rdnames.append(rdname)

    return filtered_rdnames

def extractRpeaks(rdnames, rpeak_lead, samplefrom=0, sampleto='end', verbose=False):
    allsymbols = []
    for rdname in rdnames:
        print(rdname)
        sig, fields = wfdb.rdsamp('mitdb/'+rdname, channels='all', sampfrom=samplefrom, sampto=sampleto)
        ann_ref = wfdb.rdann('mitdb/'+rdname,'atr', sampfrom=samplefrom, sampto=None if sampleto=='end' else sampleto)

        peak_channel = 0
        if rpeak_lead in fields['sig_name']:
           peak_channel = fields['sig_name'].index(rpeak_lead)
        else:
            continue 

        xqrs = processing.XQRS(sig=sig[:,peak_channel], fs=fields['fs'])
        xqrs.detect()
        acts = xqrs.qrs_inds

        comparitor = processing.compare_annotations(ref_sample=ann_ref.sample[1:],
                                                    test_sample=xqrs.qrs_inds,
                                                    window_width=int(0.1 * fields['fs']),
                                                    signal=sig[:,peak_channel])
        if verbose:
            comparitor.print_summary()

        matched_inds = comparitor.matching_sample_nums

        outpath = 'mitdb/'+rdname + '.Rpeaks'
        with open(outpath, 'wb') as handle:
            pickle.dump({'acts':acts, 'matched_inds':matched_inds, 'anns': ann_ref.symbol[1:]}, handle)

def extractSamples(rdnames, winL=-90, winR=90, samplefrom=0, sampleto='end', verbose=False):
    allsymbols = []
    for rdname in rdnames:
        print(rdname)
        sig, fields = wfdb.rdsamp('mitdb/'+rdname, channels='all', sampfrom=samplefrom, sampto=sampleto)

        sigupper = len(sig)-1

        peakpath = 'mitdb/'+rdname + '.Rpeaks'
        with open(peakpath, 'rb') as handle:
            dict = pickle.load(handle)
            acts = dict['acts']
            matched_inds = dict['matched_inds']
            anns = dict['anns']

        actnum = len(matched_inds)
        actupper = len(acts)-1
    
        samples = []
        symbols = []
        rinds = []
        rlocs = np.copy(acts)

        for i in range(1, actnum):
            actind = matched_inds[i]
            if actind==-1 or actind==0 or actind==actupper:
                continue
            
            #prev_act = acts[actind-1]
            cur_act = acts[actind]
            #next_act = acts[actind+1]
            cur_sym = anns[i]
        
            start = cur_act + winL
            end = cur_act + winR

            if start<0:
                continue
            if end>sigupper:
                continue

            if cur_sym =='/':
                cur_sym ='P'

            samples.append((start, end))
            symbols.append(cur_sym)
            allsymbols.append(cur_sym)
            rinds.append(actind)

        assert len(samples)==len(symbols)

        outpath = 'mitdb/' + rdname + '.samples'
        with open(outpath, 'wb') as handle:
            pickle.dump({'samples':samples, 'symbols':symbols, 'rlocs' : rlocs, 'rinds': rinds}, handle)

    return allsymbols

def stat_labels(rdnames):
    for rdname in rdnames:
        peakpath = 'mitdb/'+rdname + '.samples'
        with open(peakpath, 'rb') as handle:
            dict = pickle.load(handle)
            symbols = dict['symbols']

        labels = []
        count = len(symbols)
        for i in range(count):
            cur_sym = symbols[i]

            cur_label = -1
            if cur_sym in symrefs['N']:
                cur_label = labrefs['N']
            elif cur_sym in symrefs['S']:
                cur_label = labrefs['S']
            elif cur_sym in symrefs['V']:
                cur_label = labrefs['V']
            elif cur_sym in symrefs['F']:
                cur_label = labrefs['F']
            elif cur_sym in symrefs['Q']:
                cur_label = labrefs['Q']
            else:
                continue
            labels.append(cur_label)

        print(rdname, ':', Counter(labels))

def statsamples(rdnames):
    allsymbols = []
    for rdname in rdnames:
        print(rdname)
        peakpath = 'mitdb/'+rdname + '.samples'
        with open(peakpath, 'rb') as handle:
            dict = pickle.load(handle)
            symbols = dict['symbols']
            allsymbols.extend(symbols)
    statstr(allsymbols)

def generatesamples(rdnames, feature ='raw'):
    samples = []
    labels = []
    sindrs = []
    rlocs = []
    rinds = []
    
    counts = {'0':0, '1':0, '2':0, '3':0, '4':0} 

    for rdname in rdnames:
        print(rdname)
        sig, fields = wfdb.rdsamp('mitdb/'+rdname, channels='all')

        peakpath = 'mitdb/'+rdname + '.samples'
        with open(peakpath, 'rb') as handle:
            dict = pickle.load(handle)
            windows = dict['samples']
            symbols = dict['symbols']
            srlocs= dict['rlocs']
            srinds= dict['rinds']

        count = len(symbols)

        MLII_channel = fields['sig_name'].index('MLII')
        V1_channel = fields['sig_name'].index('V1')

        sig_MLII = sig[:,MLII_channel]
        sig_V1 = sig[:,V1_channel]
    
        if feature=='remove_baseline':
            print('begin filtering on MLII ...')
            baseline = medfilt(sig_MLII, 71)
            baseline = medfilt(baseline, 215)
            for i in range(0, len(sig_MLII)):
                sig_MLII[i] = sig_MLII[i] - baseline[i]
                
            print('begin filtering on V1 ...')
            baseline = medfilt(sig_V1, 71) 
            baseline = medfilt(baseline, 215) 

            for i in range(0, len(sig_V1)):
                sig_V1[i] = sig_V1[i] - baseline[i] 

        ssrinds = []

        for i in range(count):
            start = windows[i][0]
            end = windows[i][1]
            cur_sym = symbols[i]
            
            MLII = sig_MLII[start:end]
            V1 = sig_V1[start:end]
            seq = {'MLII':MLII, 'V1': V1}

            cur_label = -1
            if cur_sym in symrefs['N']:
                cur_label = labrefs['N']
            elif cur_sym in symrefs['S']:
                cur_label = labrefs['S']
            elif cur_sym in symrefs['V']:
                cur_label = labrefs['V']
            elif cur_sym in symrefs['F']:
                cur_label = labrefs['F']
            elif cur_sym in symrefs['Q']:
                cur_label = labrefs['Q']
            else:
                continue

            counts[str(cur_label)] += 1

            samples.append(seq)
            labels.append(cur_label)
            ssrinds.append(srinds[i])
            

        assert len(samples)==len(labels)

        sindrs.append(len(samples))
        rinds.extend(ssrinds)
        rlocs.append(srlocs)

    return samples, labels, sindrs, rinds, rlocs, counts

def savedataset(samples, labels, sindrs, rinds, rlocs, path):
    with open(path, 'wb') as handle:
        pickle.dump({'samples':samples, 'labels': labels, 'sindrs': sindrs, 'rinds': rinds, 'rlocs': rlocs}, handle)

def loaddataset(path):
    with open(path, 'rb') as handle:
        dict = pickle.load(handle)
        return dict['samples'], dict['labels'], dict['sindrs'], dict['rinds'], dict['rlocs']

class sindrs_indexing(object):
    def __init__(self, sindrs,  **kwargs):
        self.refs = sindrs
        self.ref_count = len(self.refs)
        return super().__init__(**kwargs)

    def indexof(self, ind):
        refind = 0
        for ref in self.refs:
            if ind < ref:
                return refind
            refind+=1
        return -1
    
    def ranges(self, refind):
        return range(0 if refind==0 else self.refs[refind-1],  self.refs[refind])

def datasplit(labels, refinds, uniquelabels, trainprec, testprec, randstate):
    train_inds = []
    valid_inds = []
    test_inds = []

    for ulab in uniquelabels:
        uinds = np.where(np.equal(labels, ulab))[0]
        randstate.shuffle(uinds)
        uinds_len = len(uinds)
        
        train_uinds_len = int(uinds_len * trainprec)
        valid_uinds_upper = int(uinds_len * (1.0-testprec))

        train_uinds = uinds[:train_uinds_len]
        valid_uinds = uinds[train_uinds_len:valid_uinds_upper]
        test_uinds = uinds[valid_uinds_upper:]

        train_inds.extend([refinds[ind] for ind in train_uinds])
        valid_inds.extend([refinds[ind] for ind in valid_uinds])
        test_inds.extend([refinds[ind] for ind in test_uinds])

    return train_inds, valid_inds, test_inds

allrecords = [os.path.splitext(f)[0] for f in os.listdir('./mitdb') if f.endswith('.hea')]
#find out records with leads 'MLII', 'V1'
sel_records = filterrds_byleads(allrecords, ['MLII', 'V1'], True)
print(len(sel_records))
print(sel_records)

'extract R peaks'
#extractRpeaks(sel_records, rpeak_lead = 'MLII')

'extract samples'
winL = 90
winR = 90
#alllabels = extractSamples(sel_records, -winL, winR)
#statstr(alllabels)
#stat_labels(sel_records)

train_recordids = [106, 107, 108, 109, 112, 115, 116, 119, 121, 122, 212, 203, 205, 207, 208, 215, 232, 223, 230]
test_recordids = [105, 111, 113, 118, 200, 201, 202, 210, 213, 214, 217, 219, 221, 222, 228, 231, 220, 233, 234]

train_records = [str(rd) for rd in train_recordids]
test_records = [str(rd) for rd in test_recordids]

print('inter_training:')
statsamples(train_records)
print('inter_testing:')
statsamples(test_records)

feature = 'remove_baseline'

samples, labels, sindrs, rinds, rlocs, counts= generatesamples(train_records, feature)
savedataset(samples, labels, sindrs, rinds, rlocs, 'data/inter/train_data.pickle')
print('train dataset:')
stat(labels)

samples, labels, sindrs, rinds, rlocs, counts= generatesamples(test_records, feature)
savedataset(samples, labels, sindrs, rinds, rlocs, 'data/inter/test_data.pickle')
print('test dataset:')
stat(labels)

samples1, labels1, sindrs1, rinds1, rlocs1 = loaddataset('data/inter/train_data.pickle')
samples2, labels2, sindrs2, rinds2, rlocs2 = loaddataset('data/inter/test_data.pickle')

train_prec = 0.5
test_prec = 0.35

train_samples = []
train_labels = []
train_sindrs = []
train_rinds = []
train_rlocs = []

valid_samples = []
valid_labels = []
valid_sindrs = []
valid_rinds = []
valid_rlocs = []

test_samples = []
test_labels = []
test_sindrs = []
test_rinds = []
test_rlocs = []

sindrs_indexer1 = sindrs_indexing(sindrs1)
sindrs_indexer2 = sindrs_indexing(sindrs2)

sindr_count1 = sindrs_indexer1.ref_count
sindr_count2 = sindrs_indexer2.ref_count

randstate =  np.random.RandomState(123)

uniquelabels = [0,1,2,3,4]

for i in tqdm(range(sindr_count1), ncols=60):
    iinds = list(sindrs_indexer1.ranges(i))

    train_inds, valid_inds, test_inds = datasplit([labels1[j] for j in iinds], iinds, uniquelabels, train_prec, test_prec, randstate)

    train_samples.extend([samples1[j] for j in train_inds])
    valid_samples.extend([samples1[j] for j in valid_inds])
    test_samples.extend([samples1[j] for j in test_inds])

    train_labels.extend([labels1[j] for j in train_inds])
    valid_labels.extend([labels1[j] for j in valid_inds])
    test_labels.extend([labels1[j] for j in test_inds])

    train_rinds.extend([rinds1[j] for j in train_inds])
    valid_rinds.extend([rinds1[j] for j in valid_inds])
    test_rinds.extend([rinds1[j] for j in test_inds])

    train_rlocs.append(rlocs1[i])
    train_sindrs.append(len(train_samples))

    valid_rlocs.append(rlocs1[i])
    valid_sindrs.append(len(valid_samples))

    test_rlocs.append(rlocs1[i])
    test_sindrs.append(len(test_samples))

for i in tqdm(range(sindr_count2), ncols=60):
    iinds = list(sindrs_indexer2.ranges(i))

    train_inds, valid_inds, test_inds = datasplit([labels2[j] for j in iinds], iinds, uniquelabels, train_prec, test_prec, randstate)

    train_samples.extend([samples2[j] for j in train_inds])
    valid_samples.extend([samples2[j] for j in valid_inds])
    test_samples.extend([samples2[j] for j in test_inds])

    train_labels.extend([labels2[j] for j in train_inds])
    valid_labels.extend([labels2[j] for j in valid_inds])
    test_labels.extend([labels2[j] for j in test_inds])

    train_rinds.extend([rinds2[j] for j in train_inds])
    valid_rinds.extend([rinds2[j] for j in valid_inds])
    test_rinds.extend([rinds2[j] for j in test_inds])

    train_rlocs.append(rlocs2[i])
    train_sindrs.append(len(train_samples))

    valid_rlocs.append(rlocs2[i])
    valid_sindrs.append(len(valid_samples))

    test_rlocs.append(rlocs2[i])
    test_sindrs.append(len(test_samples))

savedataset(train_samples, train_labels, train_sindrs, train_rinds, train_rlocs, 'data/intra/train_data.pickle')
savedataset(valid_samples, valid_labels, valid_sindrs, valid_rinds, valid_rlocs, 'data/intra/valid_data.pickle')
savedataset(test_samples, test_labels, test_sindrs, test_rinds, test_rlocs, 'data/intra/test_data.pickle')


print('train dataset:')
stat(train_labels)

print('valid dataset:')
stat(valid_labels)

print('test dataset:')
stat(test_labels)

    






