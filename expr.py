__author__ = 'iwawiwi'

from 	sklearn.cross_validation import KFold
import 	multilabel_classification as mc
import 	weak_bagging as wb
import 	numpy as np
from 	enselm import BaggingELMClassifier
import 	time


data_range 	= range(0,294,1)
label_range = range(294,300,1)
#data, targets = mc.load_mulan_data('emotion.csv',range(0,72,1),range(72,78,1)) # Dataset labelnya sedikit
#data, targets = mc.load_mulan_data('scene.csv',data_range,label_range) # Dataset labelnya sedikit
data, targets = mc.load_mulan_data('yeast.csv',range(0,103,1),range(103,117,1)) # Dataset labelnya sedikit
#data, targets = mc.load_mulan_data('CAL500.csv',range(0,67,1),range(68,242,1)) # Dataset labelnya banyak

n_fold 			= 10
max_estimators 	= 20
init_estimators = 2
kf 				= KFold(len(targets), n_folds=n_fold)

accuracy_profile_tree 	= np.zeros(shape=(max_estimators,1)) # accuracy profile
precision_profile_tree 	= np.zeros(shape=(max_estimators,1)) # precision profile
recall_profile_tree 	= np.zeros(shape=(max_estimators,1)) # recall profile

accuracy_profile_elm 	= np.zeros(shape=(max_estimators,1)) 
precision_profile_elm 	= np.zeros(shape=(max_estimators,1)) 
recall_profile_elm 		= np.zeros(shape=(max_estimators,1)) 

accuracy_profile_svm 	= np.zeros(shape=(max_estimators,1)) 
precision_profile_svm 	= np.zeros(shape=(max_estimators,1)) 
recall_profile_svm 		= np.zeros(shape=(max_estimators,1)) 

for i in range(init_estimators,max_estimators+1,1):
    accuracy_tree 	= np.zeros(shape=(n_fold,1)) # Because 10-Fold classification
    precision_tree 	= np.zeros(shape=(n_fold,1))
    recall_tree 	= np.zeros(shape=(n_fold,1)) 
	
    accuracy_elm 	= np.zeros(shape=(n_fold,1)) 
    precision_elm 	= np.zeros(shape=(n_fold,1))
    recall_elm 		= np.zeros(shape=(n_fold,1))
	
    accuracy_svm 	= np.zeros(shape=(n_fold,1)) 
    precision_svm 	= np.zeros(shape=(n_fold,1)) 
    recall_svm 		= np.zeros(shape=(n_fold,1)) 

    classifier_tree = wb.BaggingWeakClassifier('DecisionStump',n_estimator=i)
    classifier_elm 	= BaggingELMClassifier(n_estimators=i)
    classifier_svm 	= wb.BaggingWeakClassifier('SVM', n_estimator=i)
    
    start = time.time()
    print '# Evaluating Ensemble for Num-Estimator: ', i

    jj = 0
    for train, test in kf:
        data_temp 		= data[train,:]
        targets_temp 	= targets[train,:]
		
        X_train_transformed, y_train_transformed = mc.multilabel_dataset_transform(data_temp, targets_temp)

        classifier_tree.fit(X_train_transformed, y_train_transformed)
        classifier_elm.fit(X_train_transformed, y_train_transformed)
        classifier_svm.fit(X_train_transformed, y_train_transformed)

        test_targets_temp = mc.target_transform(targets[test,:])

        accuracy_tree[jj], precision_tree[jj], recall_tree[jj] 	= classifier_tree.score_multilabel(data[test,:], test_targets_temp)
        accuracy_elm[jj], precision_elm[jj], recall_elm[jj] 	= classifier_elm.score_multilabel(data[test,:], test_targets_temp)
        accuracy_svm[jj], precision_svm[jj], recall_svm[jj] 	= classifier_svm.score_multilabel(data[test,:], test_targets_temp)

        #print 'ACC: ', accuracy[jj], 'PREC: ', precision[jj], 'REC: ', recall[jj]
        jj += 1
    #print '>>> ACCURACY  : ', np.average(accuracy)
    #print '>>> PRECISION : ', np.average(precision)
    #print '>>> RECALL    : ', np.average(recall)

    accuracy_profile_tree[i-1] 	= np.average(accuracy_tree)
    precision_profile_tree[i-1] = np.average(precision_tree)
    recall_profile_tree[i-1] 	= np.average(recall_tree)
    print '>>> Accuracy  @TREE: ', accuracy_profile_tree[i-1]
    print '>>> Precision @TREE: ', precision_profile_tree[i-1]
    print '>>> Recall    @TREE: ', recall_profile_tree[i-1]
    
    accuracy_profile_elm[i-1] 	= np.average(accuracy_elm)
    precision_profile_elm[i-1] 	= np.average(precision_elm)
    recall_profile_elm[i-1] 	= np.average(recall_elm)
    print '>>> Accuracy  @ELM : ', accuracy_profile_elm[i-1]
    print '>>> Precision @ELM : ', precision_profile_elm[i-1]
    print '>>> Recall    @ELM : ', recall_profile_elm[i-1]
    
    accuracy_profile_svm[i-1] 	= np.average(accuracy_svm)
    precision_profile_svm[i-1] 	= np.average(precision_svm)
    recall_profile_svm[i-1] 	= np.average(recall_svm)
    print '>>> Accuracy  @SVM : ', accuracy_profile_svm[i-1]
    print '>>> Precision @SVM : ', precision_profile_svm[i-1]
    print '>>> Recall    @SVM : ', recall_profile_svm[i-1]
	
    end = time.time()
    print 'Finished in: ', end - start, 'second.'

import pylab as pl

fig = pl.figure('Accuracy profile @10-Fold CV')
ax = fig.add_subplot(111)
ax.plot(np.arange(max_estimators) + init_estimators, accuracy_profile_elm,
        label='Bag. ELM',
        color='red', linewidth=2)
ax.plot(np.arange(max_estimators) + init_estimators, accuracy_profile_tree,
        label='Bag. Decision Stump',
        color='blue', linewidth=2)
ax.plot(np.arange(max_estimators) + init_estimators, accuracy_profile_svm,
        label='Bag. SVM',
        color='green', linewidth=2)
ax.set_xlabel('n_estimators')
ax.set_ylabel('accuracy rate')
leg = ax.legend(loc='upper right', fancybox=True)
leg.get_frame().set_alpha(0.7)
#pl.show()

fig = pl.figure('Precision profile @10-Fold CV')
ax = fig.add_subplot(111)
ax.plot(np.arange(max_estimators) + init_estimators, precision_profile_elm,
        label='Bag. ELM',
        color='red', linewidth=2)
ax.plot(np.arange(max_estimators) + init_estimators, precision_profile_tree,
        label='Bag. Decision Stump',
        color='blue', linewidth=2)
ax.plot(np.arange(max_estimators) + init_estimators, accuracy_profile_svm,
        label='Bag. SVM',
        color='green', linewidth=2)
ax.set_xlabel('n_estimators')
ax.set_ylabel('precision rate')
leg = ax.legend(loc='upper right', fancybox=True)
leg.get_frame().set_alpha(0.7)
#pl.show()

fig = pl.figure('Recall profile @10-Fold CV')
ax = fig.add_subplot(111)
ax.plot(np.arange(max_estimators) + init_estimators, recall_profile_elm,
        label='Bag. ELM',
        color='red', linewidth=2)
ax.plot(np.arange(max_estimators) + init_estimators, recall_profile_tree,
        label='Bag. Decision Stump',
        color='blue', linewidth=2)
ax.plot(np.arange(max_estimators) + init_estimators, accuracy_profile_svm,
        label='Bag. SVM',
        color='green', linewidth=2)
ax.set_xlabel('n_estimators')
ax.set_ylabel('recall rate')
leg = ax.legend(loc='upper right', fancybox=True)
leg.get_frame().set_alpha(0.7)
pl.show()