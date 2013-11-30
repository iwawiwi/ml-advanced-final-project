from sklearn.datasets import load_iris
from elm import ELMClassifier
import numpy as np

##########################################################################
######################## SINGLE ELM TEST #################################
##########################################################################
# Define number of weak classifier
n_estimator = 10
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
elm_classifier.fit(train_data,train_target)

predicted_class = elm_classifier.predict(test_data)
error_elm = 1.0 - elm_classifier.score(test_data, test_target)
print 'Predicted Class\t\t', predicted_class
print 'Original Class\t\t', test_target
print 'Error Score\t\t\t', error_elm