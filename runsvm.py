import sys, getopt, os
sys.path.append(os.path.dirname(__file__) + '\\..')
from ECG import console


if __name__ == "__main__":
    
    numclasses = 5

    
    expsnum = 1
    batchsize = 1
    maxepoch = 1
    patience = 4
    timesteps = 1
    #imbalanced
    balanced = False
    svmkernel = 'linear'
    print('svm with rrintervals...')
    console.run('svm', 'rrintervals', 'flatten', timesteps, numclasses, batchsize, maxepoch, patience, expsnum, balanced, svm_kernel=svmkernel)
    print('svm with raw...')
    console.run('svm', 'raw', 'flatten', timesteps, numclasses, batchsize, maxepoch, patience, expsnum, balanced, svm_kernel=svmkernel)
    print('svm with fft35...')
    console.run('svm', 'fft35', 'flatten', timesteps, numclasses, batchsize, maxepoch, patience, expsnum, balanced, svm_kernel=svmkernel)
    print('svm with waveletdb1lvl3...')
    console.run('svm', 'waveletdb1lvl3', 'flatten', timesteps, numclasses, batchsize, maxepoch, patience, expsnum, balanced, svm_kernel=svmkernel)
    print('svm with hos...')
    console.run('svm', 'hos', 'flatten', timesteps, numclasses, batchsize, maxepoch, patience, expsnum, balanced, svm_kernel=svmkernel)
    print('svm with waveletdb1lvl3uniformlbp...')
    console.run('svm', 'waveletdb1lvl3uniformlbp', 'flatten', timesteps, numclasses, batchsize, maxepoch, patience, expsnum, balanced, svm_kernel=svmkernel)
    print('svm with hermite...')
    console.run('svm', 'hermite', 'flatten', timesteps, numclasses, batchsize, maxepoch, patience, expsnum, balanced, svm_kernel=svmkernel)

    ##balanced
    #balanced = True
    #svmkernel = 'linear'
    #print('svm with rrintervals...')
    #console.run('svm', 'rrintervals', 'flatten', timesteps, numclasses, batchsize, maxepoch, patience, expsnum, balanced, svm_kernel=svmkernel)
    #print('svm with raw...')
    #console.run('svm', 'raw', 'flatten', timesteps, numclasses, batchsize, maxepoch, patience, expsnum, balanced, svm_kernel=svmkernel)
    #print('svm with fft35...')
    #console.run('svm', 'fft35', 'flatten', timesteps, numclasses, batchsize, maxepoch, patience, expsnum, balanced, svm_kernel=svmkernel)
    #print('svm with waveletdb1lvl3...')
    #console.run('svm', 'waveletdb1lvl3', 'flatten', timesteps, numclasses, batchsize, maxepoch, patience, expsnum, balanced, svm_kernel=svmkernel)
    #print('svm with hos...')
    #console.run('svm', 'hos', 'flatten', timesteps, numclasses, batchsize, maxepoch, patience, expsnum, balanced, svm_kernel=svmkernel)
    #print('svm with waveletdb1lvl3uniformlbp...')
    #console.run('svm', 'waveletdb1lvl3uniformlbp', 'flatten', timesteps, numclasses, batchsize, maxepoch, patience, expsnum, balanced, svm_kernel=svmkernel)
    #print('svm with hermite...')
    #console.run('svm', 'hermite', 'flatten', timesteps, numclasses, batchsize, maxepoch, patience, expsnum, balanced, svm_kernel=svmkernel)


    #expsnum = 5
    ##imbalanced
    #balanced = False
    #print('random forest with rrintervals...')
    #console.run('rf', 'rrintervals', 'flatten', timesteps, numclasses, batchsize, maxepoch, patience, expsnum, balanced)
    #print('random forest with raw...')
    #console.run('rf', 'raw', 'flatten', timesteps, numclasses, batchsize, maxepoch, patience, expsnum, balanced)
    #print('random forest with fft35...')
    #console.run('rf', 'fft35', 'flatten', timesteps, numclasses, batchsize, maxepoch, patience, expsnum, balanced)
    #print('random forest with waveletdb1lvl3...')
    #console.run('rf', 'waveletdb1lvl3', 'flatten', timesteps, numclasses, batchsize, maxepoch, patience, expsnum, balanced)
    #print('random forest with hos...')
    #console.run('rf', 'hos', 'flatten', timesteps, numclasses, batchsize, maxepoch, patience, expsnum, balanced)
    #print('random forest with waveletdb1lvl3uniformlbp...')
    #console.run('rf', 'waveletdb1lvl3uniformlbp', 'flatten', timesteps, numclasses, batchsize, maxepoch, patience, expsnum, balanced)
    #print('random forest with hermite...')
    #console.run('rf', 'hermite', 'flatten', timesteps, numclasses, batchsize, maxepoch, patience, expsnum, balanced)

    ##balanced
    #balanced = True
    #print('random forest with rrintervals...')
    #console.run('rf', 'rrintervals', 'flatten', timesteps, numclasses, batchsize, maxepoch, patience, expsnum, balanced)
    #print('random forest with raw...')
    #console.run('rf', 'raw', 'flatten', timesteps, numclasses, batchsize, maxepoch, patience, expsnum, balanced)
    #print('random forest with fft35...')
    #console.run('rf', 'fft35', 'flatten', timesteps, numclasses, batchsize, maxepoch, patience, expsnum, balanced)
    #print('random forest with waveletdb1lvl3...')
    #console.run('rf', 'waveletdb1lvl3', 'flatten', timesteps, numclasses, batchsize, maxepoch, patience, expsnum, balanced)
    #print('random forest with hos...')
    #console.run('rf', 'hos', 'flatten', timesteps, numclasses, batchsize, maxepoch, patience, expsnum, balanced)
    #print('random forest with waveletdb1lvl3uniformlbp...')
    #console.run('rf', 'waveletdb1lvl3uniformlbp', 'flatten', timesteps, numclasses, batchsize, maxepoch, patience, expsnum, balanced)
    #print('random forest with hermite...')
    #console.run('rf', 'hermite', 'flatten', timesteps, numclasses, batchsize, maxepoch, patience, expsnum, balanced)


    #expsnum = 5
    #batchsize = 1024
    #maxepoch = 120
    #patience = 4
    #timesteps = None
    ##imbalanced
    #balanced = False
    #print('dnn with rrintervals...')
    #console.run('dnn', 'rrintervals', 'flatten', timesteps, numclasses, batchsize, maxepoch, patience, expsnum, balanced)
    #print('dnn with raw...')
    #console.run('dnn', 'raw', 'flatten', timesteps, numclasses, batchsize, maxepoch, patience, expsnum, balanced)
    #print('dnn with fft35...')
    #console.run('dnn', 'fft35', 'flatten', timesteps, numclasses, batchsize, maxepoch, patience, expsnum, balanced)
    #print('dnn with waveletdb1lvl3...')
    #console.run('dnn', 'waveletdb1lvl3', 'flatten', timesteps, numclasses, batchsize, maxepoch, patience, expsnum, balanced)
    #print('dnn with hos...')
    #console.run('dnn', 'hos', 'flatten', timesteps, numclasses, batchsize, maxepoch, patience, expsnum, balanced)
    #print('dnn with waveletdb1lvl3uniformlbp...')
    #console.run('dnn', 'waveletdb1lvl3uniformlbp', 'flatten', timesteps, numclasses, batchsize, maxepoch, patience, expsnum, balanced)
    #print('dnn with hermite...')
    #console.run('dnn', 'hermite', 'flatten', timesteps, numclasses, batchsize, maxepoch, patience, expsnum, balanced)

    ##balanced
    #balanced = True
    #print('dnn with rrintervals...')
    #console.run('dnn', 'rrintervals', 'flatten', timesteps, numclasses, batchsize, maxepoch, patience, expsnum, balanced)
    #print('dnn with raw...')
    #console.run('dnn', 'raw', 'flatten', timesteps, numclasses, batchsize, maxepoch, patience, expsnum, balanced)
    #print('dnn with fft35...')
    #console.run('dnn', 'fft35', 'flatten', timesteps, numclasses, batchsize, maxepoch, patience, expsnum, balanced)
    #print('dnn with waveletdb1lvl3...')
    #console.run('dnn', 'waveletdb1lvl3', 'flatten', timesteps, numclasses, batchsize, maxepoch, patience, expsnum, balanced)
    #print('dnn with hos...')
    #console.run('dnn', 'hos', 'flatten', timesteps, numclasses, batchsize, maxepoch, patience, expsnum, balanced)
    #print('dnn with waveletdb1lvl3uniformlbp...')
    #console.run('dnn', 'waveletdb1lvl3uniformlbp', 'flatten', timesteps, numclasses, batchsize, maxepoch, patience, expsnum, balanced)
    #print('dnn with hermite...')
    #console.run('dnn', 'hermite', 'flatten', timesteps, numclasses, batchsize, maxepoch, patience, expsnum, balanced)

