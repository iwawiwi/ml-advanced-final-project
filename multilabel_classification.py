__author__ = 'iwawiwi'
import numpy as np


# TRANSFORM DATASET
def multilabel_dataset_transform(X, Y):
    # Dataset Transformation
    ii = 0 # Pointer to all data sample, X.shape[0] or labels.shape[0]
    temp = np.zeros(shape=(1,X.shape[1]))
    temp2 = [0]
    #print Y
    for labels in Y:
        #print 'labels: ', labels
        idx_class = 0 # Pointer to determine class nominal, ignore 0 value on labels
        for atom in labels:
            #print 'atom: ', atom
            if atom == 1:
                temp = np.vstack((temp,X[ii]))
                temp2 = np.hstack((temp2,idx_class))
            idx_class += 1
        ii += 1

    # Remove first column
    X_transformed = temp[1:,:]
    y_transformed = temp2[1:]
    #print np.array(y_transformed)
    #print np.array(X_transformed)
    return np.array(X_transformed), np.array(y_transformed)


# TRANSFORM OBJECTIVE TARGETS
def target_transform(Y):
    temp = []
    #print Y
    for labels in Y:
        #print 'labels: ', labels
        temp2 = []
        idx_class = 0 # Pointer to determine class nominal, ignore 0 value on labels
        for atom in labels:
            #print 'atom: ', atom
            if atom == 1:
                temp2.append(idx_class)
            idx_class += 1
        temp.append(temp2)

    #print 'TARGET TRANSFORM: ', np.array(temp)
    return np.array(temp)


def load_mulan_data(filename,data_range,target_range):
    data = np.loadtxt(fname=filename,delimiter=',',skiprows=1,usecols=data_range)
    targets = np.loadtxt(fname=filename,delimiter=',',skiprows=1,usecols=target_range)

    return data, targets