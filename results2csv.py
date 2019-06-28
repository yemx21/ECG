import pickle as pickle
import numpy as np
from os import listdir
import warnings

symtables = ['N', 'S', 'V', 'F', 'Q']
symlabels = [0,1,2,3,4]

def print_cm(cm, labels, hide_zeroes=False, hide_diagonal=False, hide_threshold=None):
    columnwidth = max([len(x) for x in labels] + [5])  # 5 is value length
    empty_cell = " " * columnwidth
    # Print header
    print("    " + empty_cell, end=" ")
    for label in labels:
        print("%{0}s".format(columnwidth) % label, end=" ")
    print()
    # Print rows
    for i, label1 in enumerate(labels):
        print("    %{0}s".format(columnwidth) % label1, end=" ")
        for j in range(len(labels)):
            cell = "%{0}.0f".format(columnwidth) % cm[i, j]
            if hide_zeroes:
                cell = cell if float(cm[i, j]) != 0 else empty_cell
            if hide_diagonal:
                cell = cell if i != j else empty_cell
            if hide_threshold:
                cell = cell if cm[i, j] > hide_threshold else empty_cell
            print(cell, end=" ")
        print()

def print_rp(r,p,f1,labels, target_names, digits=2):

    if target_names is not None and len(labels) != len(target_names):
        warnings.warn(
            "labels size, {0}, does not match size of target_names, {1}"
            .format(len(labels), len(target_names))
        )


    if target_names is None:
        target_names = [u'%s' % l for l in labels]
    name_width = max(len(cn) for cn in target_names)
    width = name_width

    headers = ["precision", "recall", "f1-score"]
    head_fmt = u'{:>{width}s} ' + u' {:>9}' * len(headers)
    report = head_fmt.format(u'', *headers, width=width)
    report += u'\n\n'

    row_fmt = u'{:>{width}s} ' + u' {:>9.{digits}f}' * 3 + u'\n'
    rows = zip(target_names, p, r, f1)
    for row in rows:
        report += row_fmt.format(*row, width=width, digits=digits)

    print(report)

def printresults(method, valstr, expnum, numclasses):
    modeldir = "models/" + method + "/" + valstr + "/"
    logpath = modeldir + method + ".log"

    with open(logpath, 'rb') as handle:
        epochs = np.fromfile(handle, dtype='int32', count = expnum)
        print('epochs:', epochs)
        avgepochs = np.fromfile(handle, dtype='int32', count = 1)
        avgcsr = np.fromfile(handle, dtype='float64', count = 1)
        confmat = np.fromfile(handle, dtype='float64', count= numclasses * numclasses).reshape((numclasses,numclasses))
        recall = np.fromfile(handle, dtype='float64', count= numclasses)
        precision = np.fromfile(handle, dtype='float64', count= numclasses)
        f1score = np.fromfile(handle, dtype='float64', count= numclasses)
    print('average classificaiton error:',  (1.0-avgcsr)*100.0, '%')
    print_cm(confmat, symtables)
    print_rp(recall, precision, f1score, symlabels, symtables)
    

numclasses=5

expnum = 1
#print('svm with rrintervals...')
#printresults('svm', 'rrintervals', expnum, numclasses)
#print('svm with raw...')
#printresults('svm', 'raw', expnum, numclasses)
#print('svm with fft35...')
#printresults('svm', 'fft35', expnum, numclasses)
#print('svm with waveletdb1lvl3...')
#printresults('svm', 'waveletdb1lvl3', expnum, numclasses)
#print('svm with hos...')
#printresults('svm', 'hos', expnum, numclasses)
#print('svm with waveletdb1lvl3uniformlbp...')
#printresults('svm', 'waveletdb1lvl3uniformlbp', expnum, numclasses)
#print('svm with hermite...')
#printresults('svm', 'hermite', expnum, numclasses)

expnum = 5
print('rf with rrintervals...')
printresults('rf', 'rrintervals', expnum, numclasses)
print('rf with raw...')
printresults('rf', 'raw', expnum, numclasses)
print('rf with fft35...')
printresults('rf', 'fft35', expnum, numclasses)
print('rf with waveletdb1lvl3...')
printresults('rf', 'waveletdb1lvl3', expnum, numclasses)
print('rf with hos...')
printresults('rf', 'hos', expnum, numclasses)
print('rf with waveletdb1lvl3uniformlbp...')
printresults('rf', 'waveletdb1lvl3uniformlbp', expnum, numclasses)
print('rf with hermite...')
printresults('rf', 'hermite', expnum, numclasses)

expnum = 1
print('dnn with rrintervals...')
printresults('dnn', 'rrintervals', expnum, numclasses)
#print('dnn with raw...')
#printresults('dnn', 'raw', expnum, numclasses)
#print('dnn with fft35...')
#printresults('dnn', 'fft35', expnum, numclasses)
#print('dnn with waveletdb1lvl3...')
#printresults('dnn', 'waveletdb1lvl3', expnum, numclasses)
#print('dnn with hos...')
#printresults('dnn', 'hos', expnum, numclasses)
#print('dnn with waveletdb1lvl3uniformlbp...')
#printresults('dnn', 'waveletdb1lvl3uniformlbp', expnum, numclasses)
#print('dnn with hermite...')
#printresults('dnn', 'hermite', expnum, numclasses)


