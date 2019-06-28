import sys, getopt, os
sys.path.append(os.path.dirname(__file__) + '\\..')
from ECG import console

os.environ['MP_CPUJOBS']='1'
os.environ["CUDA_VISIBLE_DEVICES"]="-1"


if __name__ == "__main__":
    numclasses = 5
    balanced = True

    expsnum = 1
    batchsize = 1
    timesteps = 1

    svmkernel = 'rbf'
    #print('svm with rrintervals...')
    #console.bench('svm', 'rrintervals', 'flatten', timesteps, numclasses, batchsize, expsnum, balanced, svm_kernel=svmkernel)
    #print('svm with raw...')
    #console.bench('svm', 'raw', 'flatten', timesteps, numclasses, batchsize, expsnum, balanced, svm_kernel=svmkernel)
    #print('svm with fft35...')
    #console.bench('svm', 'fft35', 'flatten', timesteps, numclasses, batchsize, expsnum, balanced, svm_kernel=svmkernel)
    #print('svm with waveletdb1lvl3...')
    #console.bench('svm', 'waveletdb1lvl3', 'flatten', timesteps, numclasses, batchsize, expsnum, balanced, svm_kernel=svmkernel)
    #print('svm with hos...')
    #console.bench('svm', 'hos', 'flatten', timesteps, numclasses, batchsize, expsnum, balanced, svm_kernel=svmkernel)
    #print('svm with waveletdb1lvl3uniformlbp...')
    #console.bench('svm', 'waveletdb1lvl3uniformlbp', 'flatten', timesteps, numclasses, batchsize, expsnum, balanced, svm_kernel=svmkernel)
    #print('svm with hermite...')
    #console.bench('svm', 'hermite', 'flatten', timesteps, numclasses, batchsize, expsnum, balanced, svm_kernel=svmkernel)


    #expsnum = 5

    #print('random forest with rrintervals...')
    #console.bench('rf', 'rrintervals', 'flatten', timesteps, numclasses, batchsize, expsnum, balanced)
    #print('random forest with raw...')
    #console.bench('rf', 'raw', 'flatten', timesteps, numclasses, batchsize, expsnum, balanced)
    #print('random forest with fft35...')
    #console.bench('rf', 'fft35', 'flatten', timesteps, numclasses, batchsize, expsnum, balanced)
    #print('random forest with waveletdb1lvl3...')
    #console.bench('rf', 'waveletdb1lvl3', 'flatten', timesteps, numclasses, batchsize, expsnum, balanced)
    #print('random forest with hos...')
    #console.bench('rf', 'hos', 'flatten', timesteps, numclasses, batchsize, expsnum, balanced)
    #print('random forest with waveletdb1lvl3uniformlbp...')
    #console.bench('rf', 'waveletdb1lvl3uniformlbp', 'flatten', timesteps, numclasses, batchsize, expsnum, balanced)
    #print('random forest with hermite...')
    #console.bench('rf', 'hermite', 'flatten', timesteps, numclasses, batchsize, expsnum, balanced)

    expsnum = 5
    batchsize = 1024
    timesteps = None

    #print('dnn with rrintervals...')
    #console.bench('dnn', 'rrintervals', 'flatten', timesteps, numclasses, batchsize, expsnum, balanced)
    #print('dnn with raw...')
    #console.bench('dnn', 'raw', 'flatten', timesteps, numclasses, batchsize, expsnum, balanced)
    #print('dnn with fft35...')
    #console.bench('dnn', 'fft35', 'flatten', timesteps, numclasses, batchsize, expsnum, balanced)
    #print('dnn with waveletdb1lvl3...')
    #console.bench('dnn', 'waveletdb1lvl3', 'flatten', timesteps, numclasses, batchsize, expsnum, balanced)
    #print('dnn with hos...')
    #console.bench('dnn', 'hos', 'flatten', timesteps, numclasses, batchsize, expsnum, balanced)
    #print('dnn with waveletdb1lvl3uniformlbp...')
    #console.bench('dnn', 'waveletdb1lvl3uniformlbp', 'flatten', timesteps, numclasses, batchsize, expsnum, balanced)
    #print('dnn with hermite...')
    #console.bench('dnn', 'hermite', 'flatten', timesteps, numclasses, batchsize, expsnum, balanced)


    #print('cnn with rrintervals...')
    #console.bench('cnn', 'rrintervals', 'flatten', timesteps, numclasses, batchsize, expsnum, balanced)
    #print('cnn with raw...')
    #console.bench('cnn', 'raw', 'flatten', timesteps, numclasses, batchsize, expsnum, balanced)
    #print('cnn with fft35...')
    #console.bench('cnn', 'fft35', 'flatten', timesteps, numclasses, batchsize, expsnum, balanced)
    #print('cnn with waveletdb1lvl3...')
    #console.bench('cnn', 'waveletdb1lvl3', 'flatten', timesteps, numclasses, batchsize, expsnum, balanced)
    #print('cnn with hos...')
    #console.bench('cnn', 'hos', 'flatten', timesteps, numclasses, batchsize, expsnum, balanced)
    #print('cnn with waveletdb1lvl3uniformlbp...')
    #console.bench('cnn', 'waveletdb1lvl3uniformlbp', 'flatten', timesteps, numclasses, batchsize, expsnum, balanced)
    #print('cnn with hermite...')
    #console.bench('cnn', 'hermite', 'flatten', timesteps, numclasses, batchsize, expsnum, balanced)

    #print('gru with raw...')
    #console.bench('gru', 'raw', 'leads', timesteps, numclasses, batchsize, expsnum, balanced, test_usebalance=True)
    #print('gru with waveletdb1lvl3...')
    #console.bench('gru', 'waveletdb1lvl3', 'leads', timesteps, numclasses, batchsize, expsnum, balanced)

    #print('lstm with raw...')
    #console.bench('lstm', 'raw', 'leads', timesteps, numclasses, batchsize, expsnum, balanced, test_usebalance=True)
    #print('lstm with waveletdb1lvl3...')
    #console.bench('lstm', 'waveletdb1lvl3', 'leads', timesteps, numclasses, batchsize, expsnum, balanced)
