import numpy as np
import pywt
from pyentrp import entropy as ent
import nolds
import scipy
from scipy import stats
from PyEMD import CEEMDAN
from tqdm import tqdm

def sig_fft(sig, fftn = 35):
    features = np.fft.fft(sig, fftn).real
    return features

def sig_spectrum(sig, fftn = 35):
    features = np.fft.fft(sig, fftn).real
    return features


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

def sig_rrintervals(sindrs, norm=False):
    slen = len(sindrs)
    RR = np.zeros((slen-1,), dtype='float64')
    pre_R = np.zeros((slen,), dtype=float)
    post_R = np.zeros((slen,), dtype=float)
    local_R = np.zeros((slen,), dtype=float)
    global_R = np.zeros((slen,), dtype=float)

    if norm:
        for i in range(1, len(sindrs)):
            RR[i-1] = float(sindrs[i] - sindrs[i-1])
        mean_RR = float(np.mean(RR))

    # Pre_R and Post_R
    post_R[0] = sindrs[1] - sindrs[0]

    for i in range(1, slen-1):
        pre_R[i] = sindrs[i] - sindrs[i-1]
        post_R[i] = sindrs[i+1] - sindrs[i]

    pre_R[0] = pre_R[1]
    pre_R[slen-1] = sindrs[-1] - sindrs[-2]

    post_R[slen-1] = post_R[-1]

    # Local_R: AVG from last 1 minutes = 10800 samples
    for i in range(0, slen):
        num = 0
        avg_val = 0
        for j in range(0, i):
            if (sindrs[i] - sindrs[j]) < 10800:
                avg_val = avg_val + pre_R[j]
                num = num + 1
        if num == 0:
            local_R[i] = 0
        else:
            local_R[i] = avg_val / float(num)

	# Global R AVG: from full past-signal
    # TODO: AVG from past 20 minutes = 216000 samples
    global_R = np.append(global_R, pre_R[0])    
    for i in range(1, len(sindrs)):
        num = 0
        avg_val = 0

        for j in range( 0, i):
            if (sindrs[i] - sindrs[j]) < 216000:
                avg_val = avg_val + pre_R[j]
                num = num + 1
        
        if num == 0:
            global_R[i] = 0
        else:
            global_R[i] = avg_val / float(num)

    if norm:
        return np.vstack((pre_R[0: slen], post_R[0: slen], local_R[0: slen], global_R[0: slen])).transpose([1,0])/mean_RR
    else:
        return np.vstack((pre_R[0: slen], post_R[0: slen], local_R[0: slen], global_R[0: slen])).transpose([1,0])

def sig_wavelet(sig, family, level):
    wave_family = pywt.Wavelet(family)
    coeffs = pywt.wavedec(sig, wave_family, level=level)
    return coeffs[0]


uniform_pattern_list = np.array([0, 1, 2, 3, 4, 6, 7, 8, 12, 14, 15, 16, 24, 28, 30, 31, 32, 48, 56, 60, 62, 63, 64, 96, 112, 120, 124, 126, 127, 128,
     129, 131, 135, 143, 159, 191, 192, 193, 195, 199, 207, 223, 224, 225, 227, 231, 239, 240, 241, 243, 247, 248, 249, 251, 252, 253, 254, 255])
# Compute the uniform LBP 1D from signal with neigh equal to number of neighbours
# and return the 59 histogram:
# 0-57: uniform patterns
# 58: the non uniform pattern
def sig_uniform_lbp(sig):
    hist_u_lbp = np.zeros(59, dtype=float)

    #avg_win_size = 2
    #Reduce sampling by half
    #sig_avg = scipy.sig.resample(sig, len(sig) / avg_win_size)

    for i in range(4, int(len(sig) - 4)):
        pattern = np.zeros(8)
        ind = 0
        for n in range(-3,5,2):
            if sig[i] > sig[i+n]:
                pattern[ind] = 1          
            ind += 1
        pattern_id = int("".join(str(c) for c in pattern.astype(int)), 2)

        if pattern_id in uniform_pattern_list:
            pattern_uniform_id = int(np.argwhere(uniform_pattern_list == pattern_id))
        else:
            pattern_uniform_id = 58 # Non uniforms patternsuse

        hist_u_lbp[pattern_uniform_id] += 1.0

    return hist_u_lbp

def sig_hos(sig, regions):
    period = int(round(len(sig)/ regions))
    hos = np.zeros((regions * 5,))
    for i in range(regions):
        interval = sig[period* i: (i+1)*period]
        
        # Skewness
        hos[i*5] = scipy.stats.skew(interval, 0, True)

        if np.isnan(hos[i*5]):
            hos[i*5] = 0.0
            
        # Kurtosis
        hos[i*5+1] = scipy.stats.kurtosis(interval, 0, False, True)
        if np.isnan(hos[i*5+1]):
            hos[i*5+1] = 0.0

        # Range
        hos[i*5+2] = max(interval) - min(interval)

        # Std
        hos[i*5+3] = np.std(interval)

        # Avg
        hos[i*5+4] = np.mean(interval)

    return hos

def sig_hermite(sig, hermites):
    sigrange = range(0, len(sig))
    coeffs = np.array([], dtype=np.float)
    for hermite in hermites:
        hcoeffs = np.polynomial.hermite.hermfit(sigrange, sig, hermite)
        coeffs = np.hstack((coeffs, hcoeffs))

    return coeffs


#need implementation
def sig_dfa(sig):
    return nolds.dfa(sig)

def sig_icemm(sig):
    components = []
    ceemdan = CEEMDAN()
    IMFs = ceemdan(sig)
    imfcount  = IMFs.shape[0]
    components.append(sig-np.sum(IMFs, axis=0))    
    for i in range(imfcount):
        components.append(IMFs[i])

    return components
    
def sig_cumulants(sig, orders):
    cumulants = np.array([],dtype= np.float)
    for order in orders:
        cumulants = np.hstack((cumulants,stats.kstat(sig,order)))
    return cumulants

def sig_sample_entropy(sig):
    sig_std = np.std(sig)
    return ent.multiscale_entropy(sig, 4, 0.2* sig_std)[0]


def extractfeatures(sigs, sindrs, rinds, rlocs, leads, feats, **kwargs):
    features = np.array([], dtype=np.float)
    siglen = len(sigs[0][next(iter(sigs[0]))])
    sigcount = len(sigs)
    leadcount = len(leads)
    print(sigcount, ' samples')
    print(siglen, ' length')

    if 'raw' in feats:
        print('raw...')
        subfeatures = np.empty((sigcount, siglen*leadcount), dtype=np.float)
        for i in tqdm(range(sigcount), ncols=60):
            sig = sigs[i]
            ifeatures = np.array([], dtype=np.float)
            for lead in leads:
                lfeatures = sig[lead]
                ifeatures = np.hstack((ifeatures, lfeatures))
            subfeatures[i] = ifeatures
        features = np.column_stack((features, subfeatures)) if features.size else subfeatures

    if 'fft' in feats:
        print('fftn...')
        subfeatures = np.empty((sigcount, kwargs['fftn']*leadcount), dtype=np.float)
        for i in tqdm(range(sigcount), ncols=60):
            sig = sigs[i]
            ifeatures = np.array([], dtype=np.float)
            for lead in leads:
                lsig = sig[lead]
                lfeatures = sig_fft(lsig, kwargs['fftn'])
                ifeatures = np.hstack((ifeatures, lfeatures))
            subfeatures[i] = ifeatures
        features = np.column_stack((features, subfeatures)) if features.size else subfeatures

    if 'wavelet' in feats:
        print('wavelet...')
        subfeatures = np.empty((sigcount, 23*leadcount), dtype=np.float)
        for i in tqdm(range(sigcount), ncols=60):
            sig = sigs[i]
            ifeatures = np.array([], dtype=np.float)
            for lead in leads:
                lsig = sig[lead]
                lfeatures = sig_wavelet(lsig, kwargs['wavelet_family'], kwargs['wavelet_level'])
                ifeatures = np.hstack((ifeatures, lfeatures))
            subfeatures[i] = ifeatures
        features = np.column_stack((features, subfeatures)) if features.size else subfeatures

    #paper: "Automatic Classification of Heartbeats Using ECG Morphology and Heartbeat Interval Features"
    if 'rr_intervals' in feats:
        print('rr_intervals...')
        sindr_indexer = sindrs_indexing(sindrs)
        sindr_count = sindr_indexer.ref_count

        subfeatures = np.empty((sigcount, 4), dtype=np.float)
        for i in tqdm(range(sindr_count), ncols=60):
            iinds = sindr_indexer.ranges(i)
            rr_sfeatures = sig_rrintervals(rlocs[i], kwargs['normintervals'])
            for j in iinds:
                cur_rind = rinds[j]
                subfeatures[j] = rr_sfeatures[cur_rind]

        features = np.column_stack((features, subfeatures)) if features.size else subfeatures

    #paper: "Effective ECG beat classification using higher order statistic features and genetic feature selection"
    if 'hos' in feats:
        print('higher order statistic ...')
        subfeatures = np.empty((sigcount, kwargs['hos_region']*5*leadcount), dtype=np.float)
        for i in tqdm(range(sigcount), ncols=60):
            sig = sigs[i]
            ifeatures = np.array([], dtype=np.float)
            for lead in leads:
                lsig = sig[lead]
                lfeatures = sig_hos(lsig, kwargs['hos_region'])
                ifeatures = np.hstack((ifeatures, lfeatures))
            subfeatures[i] = ifeatures
        features = np.column_stack((features, subfeatures)) if features.size else subfeatures

    #paper: "Pattern Recognition Application in ECG Arrhythmia Classification"
    if 'wavelet_uniform_lbp' in feats:
        print('wavelet uniform lbp...')
        subfeatures = np.empty((sigcount, 59*leadcount), dtype=np.float)
        for i in tqdm(range(sigcount), ncols=60):
            sig = sigs[i]
            ifeatures = np.array([], dtype=np.float)
            for lead in leads:
                lsig = sig[lead]
                wavelet_coeffs = sig_wavelet(lsig, kwargs['wavelet_family'], kwargs['wavelet_level'])
                lfeatures = sig_uniform_lbp(wavelet_coeffs)
                ifeatures = np.hstack((ifeatures, lfeatures))
            subfeatures[i] = ifeatures
        features = np.column_stack((features, subfeatures)) if features.size else subfeatures

    #paper: "Clustering ECG Complexes Using Hermite Functions and Self-Organizing Maps"
    if 'hermite' in feats:
        print('hermite...')
        subfeaturedim = sum(kwargs['hermites']) + len(kwargs['hermites'])
        subfeatures = np.empty((sigcount, subfeaturedim*leadcount), dtype=np.float)
        for i in tqdm(range(sigcount), ncols=60):
            sig = sigs[i]
            ifeatures = np.array([], dtype=np.float)
            for lead in leads:
                lsig = sig[lead]
                lfeatures = sig_hermite(lsig, kwargs['hermites'])
                ifeatures = np.hstack((ifeatures, lfeatures))
            subfeatures[i] = ifeatures
        features = np.column_stack((features, subfeatures)) if features.size else subfeatures

    #paper: "Detrended Fluctuation Analysis  A Suitable Long-term Measure of HRV Signals in Children with Sleep Disordered Breathing"
    #if 'dfa' in feats:
    #    print('detrended fluctuation analysis...')
    #    subfeatures = np.empty((sigcount, leadcount), dtype=np.float)
    #    for i in tqdm(range(sigcount), ncols=60):
    #        sig = sigs[i]
    #        ifeatures = np.array([], dtype=np.float)
    #        for lead in leads:
    #            lsig = sig[lead]
    #            lfeatures = sig_dfa(lsig)
    #            ifeatures = np.hstack((ifeatures, lfeatures))
    #        subfeatures[i] = ifeatures
    #    features = np.column_stack((features, subfeatures)) if features.size else subfeatures

    #paper: "Classification of imbalanced ECG beats using re-sampling techniques and AdaBoost ensemble classifier"
    if 'iceemd' in feats:
        print('iceemd...')
        subfeatures = np.empty((sigcount, siglen * 7 * leadcount), dtype=np.float)
        for i in tqdm(range(sigcount), ncols=60):
            sig = sigs[i]
            ifeatures = np.array([], dtype=np.float)
            for lead in leads:
                lsig = sig[lead]
                lfeatures = sig_icemm(lsig)
                ifeatures = np.hstack(lfeatures)
            subfeatures[i] = ifeatures
        features = np.column_stack((features, subfeatures)) if features.size else subfeatures

    #paper: "Classification of imbalanced ECG beats using re-sampling techniques and AdaBoost ensemble classifier"
    #paper: "Sample entropy analysis of neonatal heart rate variability"
    if "iceemd_cumulants_sample_entropy" in feats:
        print('iceemd sample entropy...')
        subfeatures = np.empty((sigcount, 7* leadcount), dtype=np.float)
        for i in tqdm(range(sigcount), ncols=60):
            sig = sigs[i]
            ifeatures = np.array([], dtype=np.float)
            for lead in leads:
                lsig = sig[lead]
                lfeatures = sig_icemm(lsig)
                for lsfeatures in lfeatures:
                   slsfeatures1 = sig_cumulants(lsfeatures,kwargs['iceemd_cumulants_orders'])
                   ifeatures = np.hstack((ifeatures, slsfeatures1))
                   slsfeatures2 = sig_sample_entropy(lsfeatures, kwargs['iceemd_hermites'])
                   ifeatures = np.hstack((ifeatures, slsfeatures2))
            subfeatures[i] = ifeatures
        features = np.column_stack((features, subfeatures)) if features.size else subfeatures

    return features
        