from sklearn.datasets import load_iris
from enselm import BaggingELMClassifier
from elm import ELMClassifier
import numpy as np

##########################################################################
######################## SINGLE ELM TEST #################################
##########################################################################
## Define number of weak classifier
#n_estimator = 10
## Load Fisher IRIS dataset
#loaded_data = load_iris()
## Generate test and train data
#data_size = np.size(loaded_data.data,0)
#rand_perm = np.random.permutation(data_size) # generate random permutation
## Shuffle target and data
#shfl_data = loaded_data.data[rand_perm]
#shfl_target = loaded_data.target[rand_perm]
## divide train data and test data
#offset = np.floor(data_size*0.8) # 80 percent data as training
#train_data = shfl_data[:offset-1,:]
#train_target = shfl_target[:offset-1]
#test_data = shfl_data[offset:,:]
#test_target = shfl_target[offset:]
#
## Build ELM Classifier using default value
#elm_classifier = ELMClassifier()
#elm_classifier.fit(train_data,train_target)
#
#predicted_class = elm_classifier.predict(test_data)
#error_elm = 1.0 - elm_classifier.score(test_data, test_target)
#print 'Predicted Class\t\t', predicted_class
#print 'Original Class\t\t', test_target
#print 'Error Score\t\t\t', error_elm

#print '##############################################################\nBAGGING ELM RESULT\n##############################################################'
## Build ELM Classifier using default value
#elm_classifier = ELMClassifier()
## Train ELM
#elm_classifier.fit(train_data,train_target)
#
#predicted_class = elm_classifier.predict(test_data)
#error_elm = 1.0 - elm_classifier.score(test_data, test_target)
#print 'Single ELM Hypotheses'
#print 'Predicted Class\t\t', predicted_class
#print 'Original Class\t\t', test_target
#print 'Error Score\t\t\t', error_elm
#print ''
#
#
#bag_elm = BaggingELMClassifier()
#bag_elm.fit(train_data,train_target)
#
#predicted_class = bag_elm.predict(test_data)
#error_bag_elm = 1.0 - bag_elm.score(test_data, test_target)
#print 'Bagging ELM Hypotheses'
#print 'Predicted Class\t\t', predicted_class
#print 'Original Class\t\t', test_target
#print 'Error Score\t\t\t', error_bag_elm
#print ''

## Check Error Profile Ensemble ELM for varying num_estimator
#error_profile = np.ones(shape=(100,1))
#for i in range(1,100+1,1):
#    bag_elm = BaggingELMClassifier(n_estimators=i) # Create various ensemble strategy
#    bag_elm.fit(train_data,train_target)
#
#    predicted_class = bag_elm.predict(test_data)
#    error_profile[i-1] = 1.0 - bag_elm.score(test_data, test_target)
#
##print 'Error Profile', error_profile
#
#import pylab as pl
#fig = pl.figure()
#ax = fig.add_subplot(111)
#ax.plot(np.arange(100) + 1, error_profile,
#        label='Bagging ELM Error Profile',
#        color='red', linewidth=2)
#ax.set_ylim((0.0, 0.5))
#ax.set_xlabel('n_estimators')
#ax.set_ylabel('error rate')
#leg = ax.legend(loc='upper right', fancybox=True)
#leg.get_frame().set_alpha(0.7)
#pl.show()


#print '##############################################################\nBAGGING ELM RESULT\n##############################################################'
from sklearn.cross_validation import KFold
from sklearn.datasets import load_digits

digits = load_digits()
kf = KFold(len(digits.target), n_folds=10)

#print 'Single ELM Hypotheses\n--------------------------------------------------------------'
#ii = 1
#accuracy = np.zeros(shape=(10,1))
#for train, test in kf:
#    # Build ELM Classifier using default value
#    elm_classifier = ELMClassifier()
#    # Train ELM
#    elm_classifier.fit(digits.data[train],digits.target[train])
#
#    #predicted_class = elm_classifier.predict(test_data)
#    #error_elm = 1.0 - elm_classifier.score(test_data, test_target)
#    accuracy[ii-1] = elm_classifier.score(digits.data[test], digits.target[test])
#    print('Fold num %d\t Accuracy: %f' % (ii, accuracy[ii-1]))
#    ii += 1
#print('Average \t Accuracy: %f' % np.average(accuracy))
#print ''
#
#print 'Bagging ELM Hypotheses\n--------------------------------------------------------------'
#jj = 1
#for train, test in kf:
#    # Build ELM Classifier using default value
#    bag_elm = BaggingELMClassifier()
#    # Train ELM
#    bag_elm.fit(digits.data[train],digits.target[train])
#
#    #predicted_class = elm_classifier.predict(test_data)
#    #error_elm = 1.0 - elm_classifier.score(test_data, test_target)
#    accuracy[jj-1] = bag_elm.score(digits.data[test], digits.target[test])
#    print('Fold num %d\t Accuracy: %f' % (jj, accuracy[jj-1]))
#    jj += 1
#print('Average \t Accuracy: %f' % np.average(accuracy))
#print '##############################################################\nEvaluated using 10-fold Cross Validation'


#print '##############################################################\nBAGGING ELM RESULT\n##############################################################'

# Check Error Profile Ensemble ELM for varying num_estimator
performace_profile = np.ones(shape=(100,1))
for i in range(1,100+1,1):
    jj = 1
    accuracy = np.zeros(shape=(10,1))
    for train, test in kf:
        bag_elm = BaggingELMClassifier(n_estimators=i) # Create various ensemble strategy
        bag_elm.fit(digits.data[train],digits.target[train])

        #predicted_class = bag_elm.predict(test_data)
        #error_profile[i-1] = 1.0 - bag_elm.score(test_data, test_target)
        accuracy[jj-1] = bag_elm.score(digits.data[test], digits.target[test])
        jj += 1
    performace_profile[i-1] = np.average(accuracy)

import pylab as pl
fig = pl.figure()
ax = fig.add_subplot(111)
ax.plot(np.arange(100) + 1, performace_profile,
        label='Bagging ELM Performance @10-fold CV',
        color='red', linewidth=2)
#ax.set_ylim((0.0, 0.5))
ax.set_xlabel('n_estimators')
ax.set_ylabel('accuracy rate')
leg = ax.legend(loc='upper right', fancybox=True)
leg.get_frame().set_alpha(0.7)
pl.show()