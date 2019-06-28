import sys, getopt, os
sys.path.append(os.path.dirname(__file__) + '\\..')
from ECG import console


if __name__ == "__main__":
    
    numclasses = 5

    expsnum = 5
    batchsize = 512
    maxepoch = 60
    patience = 8
    timesteps = None
    #imbalanced

    balanced = True
    print('lstm with waveletdb1lvl3...')
    console.run('lstm', 'waveletdb1lvl3', 'leads', timesteps, numclasses, batchsize, maxepoch, patience, expsnum, balanced)

    balanced = False
    print('lstm with raw...')
    console.run('lstm', 'raw', 'leads', timesteps, numclasses, batchsize, maxepoch, patience, expsnum, balanced)
    print('lstm with waveletdb1lvl3...')
    console.run('lstm', 'waveletdb1lvl3', 'leads', timesteps, numclasses, batchsize, maxepoch, patience, expsnum, balanced)
