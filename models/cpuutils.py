import os
import numpy as np
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support, classification_report
from sklearn.utils.multiclass import unique_labels


def get_muticpujobs():
    num_cpujobs = os.environ.get('MP_CPUJOBS')
    if num_cpujobs is None:
        return 1
    return num_cpujobs

def encode_target(y, n = 5):
    return np.eye(n)[np.array(y, dtype=np.int32)]


def savemetrics(path, epochs, predicts, groundtruth, target_names):
    avgcsr = np.mean(groundtruth == predicts)
    avgepochs = np.mean(epochs)
    confmat = confusion_matrix(groundtruth, predicts)

    p, r, f1, _ = precision_recall_fscore_support(groundtruth, predicts,
                                                  labels= unique_labels(groundtruth, predicts),
                                                  average=None,
                                                  sample_weight=None)
    
    with open(path, "wb") as file:
        np.array(epochs).astype('int32').tofile(file)
        np.array(avgepochs).astype('int32').tofile(file)
        np.array(avgcsr).astype('float64').tofile(file)
        np.array(confmat).astype('float64').tofile(file)
        np.array(r).astype('float64').tofile(file)
        np.array(p).astype('float64').tofile(file)
        np.array(f1).astype('float64').tofile(file)

    with open(path + '.txt', 'w') as txtfile:
        txtfile.write(str(avgcsr))
        txtfile.write('\n')
        txtfile.write(str(confmat))
        txtfile.write('\n')
        txtfile.write(classification_report(groundtruth, predicts, target_names=target_names, digits=3))
     
    print(avgcsr)
    print(confmat)
    print(classification_report(groundtruth, predicts, target_names=target_names, digits=3))

def savebenchmark(path, paramnums, durations, predicts, groundtruth, target_names):
    avgcsr = np.mean(groundtruth == predicts)
    duration = np.mean(durations)
    paramnum = np.mean(paramnums)
    confmat = confusion_matrix(groundtruth, predicts)

    p, r, f1, _ = precision_recall_fscore_support(groundtruth, predicts,
                                                  labels= unique_labels(groundtruth, predicts),
                                                  average=None,
                                                  sample_weight=None)
    
    with open(path, "wb") as file:
        np.array(duration).astype('int32').tofile(file)
        np.array(paramnum).astype('int32').tofile(file)
        np.array(durations).astype('float64').tofile(file)
        np.array(paramnums).astype('float64').tofile(file)
        np.array(avgcsr).astype('float64').tofile(file)
        np.array(confmat).astype('float64').tofile(file)
        np.array(r).astype('float64').tofile(file)
        np.array(p).astype('float64').tofile(file)
        np.array(f1).astype('float64').tofile(file)

    print(duration, ' seconds')
    print(paramnum, ' parameters')
    print(avgcsr)
    print(confmat)
    print(classification_report(groundtruth, predicts, target_names=target_names, digits=3))

