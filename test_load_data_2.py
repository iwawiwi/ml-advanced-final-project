import numpy as np
from enselm import BaggingELMClassifier
from sklearn import cross_validation

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

# LOAD DATA
data = np.loadtxt(fname='emotions-train.csv',delimiter=',',skiprows=1,usecols=range(0,71,1))
targets = np.loadtxt(fname='emotions-train.csv',delimiter=',',skiprows=1,usecols=range(72,78,1))

# TODO: PROBLEM TRANSFORMATION
#data_temp = data[:5,:]
#targets_temp = targets[:5,:]
#print 'DATA: ', data_temp
#print 'TARGETS: ', targets_temp
#X_train_transformed, y_train_transformed = multilabel_dataset_transform(data_temp, targets_temp)
#print X_train_transformed, y_train_transformed
#print X_train_transformed.shape[0], y_train_transformed.size

#DO: K-FOLD
from sklearn.cross_validation import KFold
#n_fold = 10
#kf = KFold(len(targets), n_folds=n_fold)
#bag_elm = BaggingELMClassifier(n_estimators=10)
##print 'Y_TRAIN_TRANSFORMED: : ', y_train_transformed
#bag_elm.fit(X_train_transformed,y_train_transformed)
#
##print 'targets: ', targets[6:10,:]
#test_targets_temp = target_transform(targets[6:10,:])
##print 'TEST TARGET TEMP: ', test_targets_temp
#acc, prec, recall = bag_elm.score_multilabel(data[6:10,:], test_targets_temp, confidence=0)
#print 'ACCURACY: ', acc
#print 'PRECISION: ', prec
#print 'RECALL: ', recall

n_fold = 10
kf = KFold(len(targets), n_folds=n_fold)
accuracy_profile = np.zeros(shape=(200,1)) # 200 entry error profile
precision_profile = np.zeros(shape=(200,1)) # 200 entry error profile
recall_profile = np.zeros(shape=(200,1)) # 200 entry error profile
for i in range(2,200+1,1):
    accuracy = np.zeros(shape=(n_fold,1)) # Because 10-Fold classification
    precision = np.zeros(shape=(n_fold,1)) # Because 10-Fold classification
    recall = np.zeros(shape=(n_fold,1)) # Because 10-Fold classification
    bag_elm = BaggingELMClassifier(n_estimators=i)
    jj = 0
    for train, test in kf:
        data_temp = data[train,:]
        targets_temp = targets[train,:]
        X_train_transformed, y_train_transformed = multilabel_dataset_transform(data_temp, targets_temp)
        bag_elm.fit(X_train_transformed, y_train_transformed)
        test_targets_temp = target_transform(targets[test,:])
        accuracy[jj], precision[jj], recall[jj] = bag_elm.score_multilabel(data[test,:], test_targets_temp)
        #print 'ACC: ', accuracy[jj], 'PREC: ', precision[jj], 'REC: ', recall[jj]
        jj += 1
    #print 'ACCURACY: ', np.average(accuracy)
    #print 'PRECISION: ', np.average(precision)
    #print 'RECALL: ', np.average(recall)
    accuracy_profile[i-1] = np.average(accuracy)

# TODO: DO K-FOLD
#from sklearn.cross_validation import KFold
##print len(targets)
#kf = KFold(len(targets), n_folds=10) # Banyaknya data
#performace_profile = np.ones(shape=(200,1)) # 200 entry error profile
#for i in range(1,200+1,1):
#    jj = 1
#    accuracy = np.zeros(shape=(10,1)) # Because 10-Fold classification
#    for train, test in kf:
#        bag_elm = BaggingELMClassifier(n_estimators=i) # Create various ensemble strategy
#        X_train_transformed, y_train_transformed = multilabel_dataset_transform(data[train], targets[train])
#        bag_elm.fit(digits.data[train],digits.target[train])
#
#        #predicted_class = bag_elm.predict(test_data)
#        #error_profile[i-1] = 1.0 - bag_elm.score(test_data, test_target)
#        accuracy[jj-1] = bag_elm.score(digits.data[test], digits.target[test])
#        jj += 1
#    performace_profile[i-1] = np.average(accuracy)
#
import pylab as pl
fig = pl.figure('Bagging ELM Accuracy profile @10-Fold CV')
ax = fig.add_subplot(111)
ax.plot(np.arange(200) + 1, accuracy_profile,
        label='Accuracy',
        color='red', linewidth=2)
#ax.set_ylim((0.0, 0.5))
ax.set_xlabel('n_estimators')
ax.set_ylabel('accuracy rate')
leg = ax.legend(loc='upper right', fancybox=True)
leg.get_frame().set_alpha(0.7)
pl.show()

