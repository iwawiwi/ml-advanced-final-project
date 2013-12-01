from sklearn.datasets import make_multilabel_classification
import numpy as np


# GENERATE DATASET
def make_multilabel_dataset(num_sample=10, num_features=5, num_classes=3):
    X, Y = make_multilabel_classification(n_samples=num_sample, n_features=num_features,
                                          n_classes=num_classes, random_state=0)

    return X, Y


# TRANSFORM DATASET
def multilabel_dataset_transform(X, Y):
    # Dataset Transformation
    ii = 0
    temp = np.zeros(shape=(1,X.shape[1]))
    y_transformed = []
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



from sklearn import cross_validation
from enselm import BaggingELMClassifier

# GENERATE DATASET
X, Y = make_multilabel_dataset(num_classes=5, num_features=3, num_sample=30)
# Split training data and
X_train, X_test, y_train, y_test = cross_validation.train_test_split(X, Y, test_size=0.3, random_state=0)

X_train_transformed, y_train_transformed = multilabel_dataset_transform(X_train, y_train)
#print X_train_transformed
#print y_train_transformed

# Train ELM ensemble
bag_elm = BaggingELMClassifier()
bag_elm.fit(X_train_transformed,y_train_transformed)

# TODO: Test ELM
predicted = bag_elm.predict_multilabel(X_test)

print 'TARGET --> PREDICITION'
for ii in range(0,X_test.shape[0],1):
    print ('%s -->  %s' % (y_test[ii], predicted[ii]))
    #print predicted[1]
    #print y_test[1]

