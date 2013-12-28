__author__ = 'iwawiwi'
import  multilabel_classification as mc
import  weak_bagging as wb
import  numpy as np
from    enselm import BaggingELMClassifier
import  time

data_range      = range(0,294,1)
label_range     = range(294,300,1)
data, targets   = mc.load_mulan_data('scene.csv.csv',data_range,label_range)

from    sklearn.cross_validation import KFold

n_fold  = 10
kf      = KFold(len(targets), n_folds=n_fold)

accuracy        = np.zeros(shape=(n_fold,1)) # Because 10-Fold classification
precision       = np.zeros(shape=(n_fold,1)) # Because 10-Fold classification
recall          = np.zeros(shape=(n_fold,1)) # Because 10-Fold classification

accuracy_elm    = np.zeros(shape=(n_fold,1)) # Because 10-Fold classification
precision_elm   = np.zeros(shape=(n_fold,1)) # Because 10-Fold classification
recall_elm      = np.zeros(shape=(n_fold,1)) # Because 10-Fold classification


classifier      = wb.BaggingWeakClassifier('DecisionStump',n_estimator=len(label_range)+10)
classifier_elm  = BaggingELMClassifier(n_estimators=len(label_range)+10)

start = time.time()
print 'Evaluating Ensemble for Num-Estimator: ', len(label_range)+10

jj = 0
for train, test in kf:
    data_temp       = data[train,:]
    targets_temp    = targets[train,:]

    X_train_transformed, y_train_transformed \
        = mc.multilabel_dataset_transform(data_temp, targets_temp)

    classifier.fit(X_train_transformed, y_train_transformed)
    classifier_elm.fit(X_train_transformed, y_train_transformed)

    test_targets_temp = mc.target_transform(targets[test,:])

    accuracy[jj], precision[jj], recall[jj] \
        = classifier.score_multilabel(data[test,:], test_targets_temp, confidence=0.99)
    accuracy_elm[jj], precision_elm[jj], recall_elm[jj] \
        = classifier_elm.score_multilabel(data[test,:], test_targets_temp, confidence=0.99)

    #print 'ACC: ', accuracy[jj], 'PREC: ', precision[jj], 'REC: ', recall[jj]
    jj += 1
    break # TODO: 1-LOOP ONLY
#print 'ACCURACY: ', np.average(accuracy)
#print 'PRECISION: ', np.average(precision)
#print 'RECALL: ', np.average(recall)

accuracy_profile        = np.average(accuracy)
precision_profile       = np.average(precision)
recall_profile          = np.average(recall)

accuracy_profile_elm    = np.average(accuracy_elm)
precision_profile_elm   = np.average(precision_elm)
recall_profile_elm      = np.average(recall_elm)

print 'ACC: ', accuracy_profile, 'PREC: ', precision_profile, 'REC: ', recall_profile
print 'ACC: ', accuracy_profile_elm, 'PREC: ', precision_profile_elm, 'REC: ', recall_profile_elm

end = time.time()
print 'Finished in: ', end - start, 'second.'

