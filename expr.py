__author__ = 'iwawiwi'
from sklearn.cross_validation import KFold
import multilabel_classification as mc
import weak_bagging as wb
import numpy as np
from enselm import BaggingELMClassifier


#data, targets = mc.load_mulan_data('emotion.csv',range(0,71,1),range(72,78,1)) # Dataset labelnya sedikit
data, targets = mc.load_mulan_data('CAL500.csv',range(0,67,1),range(68,242,1)) # Dataset labelnya banyak

n_fold = 10
kf = KFold(len(targets), n_folds=n_fold)
accuracy_profile = np.zeros(shape=(200,1)) # 200 entry error profile
precision_profile = np.zeros(shape=(200,1)) # 200 entry error profile
recall_profile = np.zeros(shape=(200,1)) # 200 entry error profile

accuracy_profile_elm = np.zeros(shape=(200,1)) # 200 entry error profile
precision_profile_elm = np.zeros(shape=(200,1)) # 200 entry error profile
recall_profile_elm = np.zeros(shape=(200,1)) # 200 entry error profile

for i in range(2,200+1,1):
    accuracy = np.zeros(shape=(n_fold,1)) # Because 10-Fold classification
    precision = np.zeros(shape=(n_fold,1)) # Because 10-Fold classification
    recall = np.zeros(shape=(n_fold,1)) # Because 10-Fold classification
    accuracy_elm = np.zeros(shape=(n_fold,1)) # Because 10-Fold classification
    precision_elm = np.zeros(shape=(n_fold,1)) # Because 10-Fold classification
    recall_elm = np.zeros(shape=(n_fold,1)) # Because 10-Fold classification


    classifier = wb.BaggingWeakClassifier('DecisionStump',n_estimator=i)
    classifier_elm = BaggingELMClassifier(n_estimators=i)

    jj = 0
    for train, test in kf:
        data_temp = data[train,:]
        targets_temp = targets[train,:]
        X_train_transformed, y_train_transformed = mc.multilabel_dataset_transform(data_temp, targets_temp)

        classifier.fit(X_train_transformed, y_train_transformed)
        classifier_elm.fit(X_train_transformed, y_train_transformed)

        test_targets_temp = mc.target_transform(targets[test,:])

        accuracy[jj], precision[jj], recall[jj] = classifier.score_multilabel(data[test,:], test_targets_temp)
        accuracy_elm[jj], precision_elm[jj], recall_elm[jj] = classifier_elm.score_multilabel(data[test,:], test_targets_temp)

        #print 'ACC: ', accuracy[jj], 'PREC: ', precision[jj], 'REC: ', recall[jj]
        jj += 1
    #print 'ACCURACY: ', np.average(accuracy)
    #print 'PRECISION: ', np.average(precision)
    #print 'RECALL: ', np.average(recall)

    accuracy_profile[i-1] = np.average(accuracy)
    precision_profile[i-1] = np.average(precision)
    recall_profile[i-1] = np.average(recall)
    accuracy_profile_elm[i-1] = np.average(accuracy_elm)
    precision_profile_elm[i-1] = np.average(precision_elm)
    recall_profile_elm[i-1] = np.average(recall_elm)

import pylab as pl
#fig = pl.figure('Bagging Decision Stump Accuracy, Precision, and Recall profile @10-Fold CV')
#ax = fig.add_subplot(111)
#ax.plot(np.arange(200) + 1, accuracy_profile,
#        label='Accuracy',
#        color='red', linewidth=2)
#ax.plot(np.arange(200) + 1, precision_profile,
#        label='Precision',
#        color='blue', linewidth=2)
#ax.plot(np.arange(200) + 1, recall_profile,
#        label='Recall',
#        color='yellow', linewidth=2)
##ax.set_ylim((0.0, 0.5))
#ax.set_xlabel('n_estimators')
#ax.set_ylabel('rate')
#leg = ax.legend(loc='upper right', fancybox=True)
#leg.get_frame().set_alpha(0.7)
#pl.show()
fig = pl.figure('Bagging Decision Stump vs Bagging ELM Accuracy profile @10-Fold CV')
ax = fig.add_subplot(111)
ax.plot(np.arange(200) + 1, accuracy_profile,
        label='Bag. ELM',
        color='red', linewidth=2)
ax.plot(np.arange(200) + 1, accuracy_profile_elm,
        label='Bag. Decision Stump',
        color='blue', linewidth=2)
#ax.set_ylim((0.0, 0.5))
ax.set_xlabel('n_estimators')
ax.set_ylabel('accuracy rate')
leg = ax.legend(loc='upper right', fancybox=True)
leg.get_frame().set_alpha(0.7)
#pl.show()

fig = pl.figure('Bagging Decision Stump vs Bagging ELM Precision profile @10-Fold CV')
ax = fig.add_subplot(111)
ax.plot(np.arange(200) + 1, precision_profile,
        label='Bag. ELM',
        color='red', linewidth=2)
ax.plot(np.arange(200) + 1, precision_profile_elm,
        label='Bag. Decision Stump',
        color='blue', linewidth=2)
#ax.set_ylim((0.0, 0.5))
ax.set_xlabel('n_estimators')
ax.set_ylabel('precision rate')
leg = ax.legend(loc='upper right', fancybox=True)
leg.get_frame().set_alpha(0.7)
#pl.show()

fig = pl.figure('Bagging Decision Stump vs Bagging ELM Recall profile @10-Fold CV')
ax = fig.add_subplot(111)
ax.plot(np.arange(200) + 1, recall_profile,
        label='Bag. ELM',
        color='red', linewidth=2)
ax.plot(np.arange(200) + 1, recall_profile_elm,
        label='Bag. Decision Stump',
        color='blue', linewidth=2)
#ax.set_ylim((0.0, 0.5))
ax.set_xlabel('n_estimators')
ax.set_ylabel('recall rate')
leg = ax.legend(loc='upper right', fancybox=True)
leg.get_frame().set_alpha(0.7)
pl.show()