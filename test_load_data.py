import numpy as np
from enselm import BaggingELMClassifier
from sklearn import cross_validation

# TRANSFORM DATASET
def multilabel_dataset_transform(X, Y):
    # Dataset Transformation
    ii = 0
    temp = np.zeros(shape=(1,X.shape[1]))
    y_transformed = []
    #print Y
    for labels in Y:
        #print labels
        for atom in labels:
            #print X[ii]
            temp = np.vstack((temp,X[ii]))
            y_transformed.append(atom)
            #print atom
        ii += 1

    # Remove first column
    X_transformed = temp[1:,:]
    #print np.array(y_transformed)
    #print np.array(X_transformed)
    return np.array(X_transformed), np.array(y_transformed)

# LOAD DATA
data = np.loadtxt(fname='emotions-train.csv',delimiter=',',skiprows=1,usecols=range(0,71,1))
targets = np.loadtxt(fname='emotions-train.csv',delimiter=',',skiprows=1,usecols=range(72,78,1))

targets = np.array(targets, dtype='int32')

#print 'DATA: ', data
#print 'TARGETS: ', targets

# CONVERT LABEL
#print np.shape(targets)
#temp = np.zeros(shape=(1,data.shape[1])) # jumlah fitur data
y_new = []
for ii in range(0,np.shape(targets)[0]):
    target_minimized = targets[ii,:]
    pointer = 0
    #print 'target_minimized: ', target_minimized
    for jj in range (0,np.shape(targets)[1]):
        #print 'jj', jj
        #print 'target_minimized in loop: ', target_minimized
        if targets[ii,jj] == 1:
            targets[ii,jj] = jj
            target_minimized[pointer] = jj
            pointer += 1
            #print 'pointer: ', pointer
        else:
            target_minimized = np.delete(target_minimized,pointer)
    y_new.append(target_minimized)
    #print 'y_new: ', y_new
#y_new_tuple = tuple(map(tuple, y_new))
#print 'y_new: tuple?',

#print 'DATA: ', data
#print 'Y_NEW: ', y_new

#print 'TRANSFORMED TARGET: ', np.array(y_transformed)
#X, Y = multilabel_dataset_transform(data, y_new)
#print 'Y: ',  Y

# Split training data and
X_train, X_test, y_train, y_test = cross_validation.train_test_split(data, y_new, test_size=0, random_state=0)
#print y_train
X_train_transformed, y_train_transformed = multilabel_dataset_transform(X_train, y_train)
#print X_train_transformed
#print y_train_transformed

# Train ELM ensemble
bag_elm = BaggingELMClassifier()
bag_elm.fit(X_train_transformed,y_train_transformed)

# TODO: Test ELM
predicted = bag_elm.predict_multilabel(X_test)
predicted2 = bag_elm.predict_multilabel(X_test,confidence=0)

#print 'TARGET --> PREDICITION'
#for ii in range(0,X_test.shape[0],1):
#    print ('%s -->  %s' % (np.array(y_test[ii], dtype='int32'), predicted[ii]))
#    #print predicted[1]
#    #print y_test[1]

## EVALUATION METRICES (Tsoumakas 2007)
print '##############################################################\nEMOTIONS DATASET (MULAN)\n##############################################################'
# Accuracy, Precision, and Recall (Godbole and Sarawagi)
n_D = X_test.shape[0]
sigma_acc = 0
sigma_prec = 0
sigma_recall = 0

sigma_acc_2 = 0
sigma_prec_2 = 0
sigma_recall_2 = 0

for ii in range(0,n_D,1):
    print ('%s -->  %s' % (np.array(y_test[ii], dtype='int32'), predicted[ii]))
    n_intersect = np.intersect1d(np.array(y_test[ii], dtype='int32'), predicted[ii])
    #print 'Intersect: ', n_intersect
    #print 'LEN: ', len(n_intersect)
    n_union = np.union1d(np.array(y_test[ii], dtype='int32'), predicted[ii])
    #print 'Union: ', n_union
    #print 'LEN: ', len(n_union)
    sigma_acc += len(n_intersect)/len(n_union)
    sigma_prec += len(n_intersect)/len(predicted[ii])
    sigma_recall += len(n_intersect)/len(y_test[ii])

    #print ('%s -->  %s' % (np.array(y_test[ii], dtype='int32'), predicted[ii]))
    n_intersect_2 = np.intersect1d(np.array(y_test[ii], dtype='int32'), predicted2[ii])
    #print 'Intersect: ', n_intersect
    #print 'LEN: ', len(n_intersect)
    n_union_2 = np.union1d(np.array(y_test[ii], dtype='int32'), predicted2[ii])
    #print 'Union: ', n_union
    #print 'LEN: ', len(n_union)
    sigma_acc_2 += len(n_intersect_2)/len(n_union_2)
    sigma_prec_2 += len(n_intersect_2)/len(predicted2[ii])
    sigma_recall_2 += len(n_intersect_2)/len(y_test[ii])

acc = (1./n_D) * sigma_acc
prec = (1./n_D) * sigma_prec
recall = (1./n_D) * sigma_recall

acc_2 = (1./n_D) * sigma_acc_2
prec_2 = (1./n_D) * sigma_prec_2
recall_2 = (1./n_D) * sigma_recall_2


#print 'Constant: ', 1./n_D
#print 'Sigma ACC: ', sigma_acc
print 'Bagging ELM accuracy, Confidence Default (0.1)'
print 'Accuracy\t: ', acc
print 'Precision\t: ', prec
print 'Recall\t\t: ', recall

print 'Bagging ELM accuracy, Confidence = 0'
print 'Accuracy\t: ', acc_2
print 'Precision\t: ', prec_2
print 'Recall\t\t: ', recall_2



#print '##############################################################\nPERFORMANCE PROFILE ON EMOTIONS DATASET (MULAN)\n##############################################################'
#from sklearn.cross_validation import KFold
#kf = KFold(data.shape[0], n_folds=10)
#performace_profile = np.ones(shape=(200,1)) # Max 200 ELMs
##y_new = np.array(y_new, dtype='int32') # Convert y_new array to integer
#temp99, temp98, temp97, temp96 = cross_validation.train_test_split(data, y_new, test_size=0, random_state=0)
#print temp97
#for i in range(1,200+1,1):
#    jj = 1
#    accuracy = np.zeros(shape=(10,1)) # Because 10-Fold Classification
#    for train, test in kf:
#        bag_elm = BaggingELMClassifier(n_estimators=i) # Create various ensemble strategy
#        # Transform training data for each Cross Valiadation
#        X_train_transformed, y_train_transformed = multilabel_dataset_transform(data[train], targets[train])
#        bag_elm.fit(X_train_transformed,y_train_transformed) # Train current Bagging ELM
#
#        #predicted_class = bag_elm.predict(test_data)
#        #error_profile[i-1] = 1.0 - bag_elm.score(test_data, test_target)
#        accuracy[jj-1] = bag_elm.score_multilabel(data[test], temp97[test])
#        jj += 1
#    performace_profile[i-1] = np.average(accuracy)
#
#import pylab as pl
#fig = pl.figure('Bagging ELM Performace for Emotion Dataset (MULAN)')
#ax = fig.add_subplot(111)
#ax.plot(np.arange(1000) + 1, performace_profile,
#        label='Bagging ELM Performance @10-fold CV',
#        color='red', linewidth=2)
##ax.set_ylim((0.0, 0.5))
#ax.set_xlabel('n_estimators')
#ax.set_ylabel('accuracy rate')
#leg = ax.legend(loc='upper right', fancybox=True)
#leg.get_frame().set_alpha(0.7)
#pl.show()