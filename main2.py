from sklearn.datasets import load_iris
from elm import ELMClassifier
import numpy as np

##########################################################################
######################## SINGLE ELM TEST #################################
##########################################################################
print '##############################################################\nSINGLE ELM RESULT\n##############################################################'
# Define number of weak classifier
n_estimator = 100
# Load Fisher IRIS dataset
loaded_data = load_iris()
# Generate test and train data
data_size = np.size(loaded_data.data,0)
rand_perm = np.random.permutation(data_size) # generate random permutation
# Shuffle target and data
shfl_data = loaded_data.data[rand_perm]
shfl_target = loaded_data.target[rand_perm]
# divide train data and test data
offset = np.floor(data_size*0.8) # 80 percent data as training
train_data = shfl_data[:offset,:]
train_target = shfl_target[:offset]
test_data = shfl_data[offset+1:,:]
test_target = shfl_target[offset+1:]

# Build ELM Classifier using default value
elm_classifier = ELMClassifier()
# Train ELM
elm_classifier.fit(train_data,train_target)

predicted_class = elm_classifier.predict(test_data)
error_elm = 1.0 - elm_classifier.score(test_data, test_target)
print 'Predicted Class\t\t', predicted_class
print 'Original Class\t\t', test_target
print 'Error Score\t\t\t', error_elm
print ''

#print '##############################################################\nSINGLE GEN-ELM RESULT\n##############################################################'
from elm import GenELMClassifier
## Build GEN_ELM Classifier using default value
#genelm_classifier = GenELMClassifier()
## Train the ELM
#genelm_classifier.fit(train_data,train_target)
#
#predicted_class = genelm_classifier.predict(test_data)
#error_elm = 1.0 - genelm_classifier.score(test_data, test_target)
#print 'Predicted Class\t\t', predicted_class
#print 'Original Class\t\t', test_target
#print 'Error Score\t\t\t', error_elm
#print ''

print '##############################################################\nADABOOST DECISION STUMP RESULT\n##############################################################'
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier

# Initialize Single Decision Stump Classifier
dstump = DecisionTreeClassifier(max_depth=1, min_samples_leaf=1)
# Train this Single Decision Stump Classifier
dstump.fit(train_data,train_target)
predicted_class = dstump.predict(test_data)
error_dstump = 1.0 - dstump.score(test_data, test_target)
print 'Single Decision Stump Hypotheses'
print 'Predicted Class\t\t', predicted_class
print 'Original Class\t\t', test_target
print 'Error Score\t\t\t', error_dstump

print 'AdaBoost Decision Stump Hypotheses'
# Initialize AdaBoost classifier using Default Decision Stump Classifier
ada_dstump = AdaBoostClassifier(n_estimators=n_estimator)
# Train the classifer
ada_dstump.fit(train_data,train_target)

predicted_class = ada_dstump.predict(test_data)
error_ada_dstump = 1.0 - ada_dstump.score(test_data, test_target)
print 'Predicted Class\t\t', predicted_class
print 'Original Class\t\t', test_target
print 'Error Score\t\t\t', error_ada_dstump
print ''

## Plot AdaBoost Error profile
#from sklearn.metrics import zero_one_loss
#import pylab as pl
#
#ada_dstump_err = np.zeros((n_estimator,))
## Record each iteration error
#for i, y_pred in enumerate(ada_dstump.staged_predict(test_data)):
#    ada_dstump_err[i] = zero_one_loss(y_pred, test_target)
#fig = pl.figure()
#ax = fig.add_subplot(111)
#ax.plot(np.arange(n_estimator) + 1, ada_dstump_err,
#        label='Decision Stump Error Profile',
#        color='red', linewidth=2)
#ax.set_ylim((0.0, 0.5))
#ax.set_xlabel('n_estimators')
#ax.set_ylabel('error rate')
#leg = ax.legend(loc='upper right', fancybox=True)
#leg.get_frame().set_alpha(0.7)
#pl.show()

print '##############################################################\nADABOOST ELM RESULT\n##############################################################'
elms = GenELMClassifier()
ada_elms = AdaBoostClassifier(base_estimator=elms, n_estimators=n_estimator, algorithm='SAMME')
ada_elms.fit(train_data,train_target)