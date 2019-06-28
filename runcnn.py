import sys, getopt, os
sys.path.append(os.path.dirname(__file__) + '\\..')
from ECG import console


if __name__ == "__main__":
    
    numclasses = 5

    expsnum = 5
    batchsize = 1024
    maxepoch = 120
    patience = 12
    timesteps = None
    #imbalanced
    balanced = True
    print('cnn with rrintervals...')
    console.run('cnn', 'rrintervals', 'flatten', timesteps, numclasses, batchsize, maxepoch, patience, expsnum, balanced)
    print('cnn with raw...')
    console.run('cnn', 'raw', 'flatten', timesteps, numclasses, batchsize, maxepoch, patience, expsnum, balanced)
    print('cnn with fft35...')
    console.run('cnn', 'fft35', 'flatten', timesteps, numclasses, batchsize, maxepoch, patience, expsnum, balanced)
    print('cnn with waveletdb1lvl3...')
    console.run('cnn', 'waveletdb1lvl3', 'flatten', timesteps, numclasses, batchsize, maxepoch, patience, expsnum, balanced)
    print('cnn with hos...')
    console.run('cnn', 'hos', 'flatten', timesteps, numclasses, batchsize, maxepoch, patience, expsnum, balanced)
    print('cnn with waveletdb1lvl3uniformlbp...')
    console.run('cnn', 'waveletdb1lvl3uniformlbp', 'flatten', timesteps, numclasses, batchsize, maxepoch, patience, expsnum, balanced)
    print('cnn with hermite...')
    console.run('cnn', 'hermite', 'flatten', timesteps, numclasses, batchsize, maxepoch, patience, expsnum, balanced)
