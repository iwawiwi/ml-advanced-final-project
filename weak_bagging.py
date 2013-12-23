import numpy as np
from sklearn.tree.tree import DecisionTreeClassifier


def make_classifier(class_name):
    if class_name == 'DecisionStump':
        return DecisionTreeClassifier(max_depth=1, min_samples_leaf=1)
    else:
        raise Exception

class BaggingWeakClassifier():
    def __init__(self, classifier_name, n_estimator=10):
        self.classifier = []
        self.n_estimator = n_estimator

        for i in range(0,n_estimator,1):
            classifier = make_classifier(classifier_name)
            self.classifier.append(classifier)

    # BAGGING
    def bagging(self, X, y):
        m = np.shape(X)[0]
        idx_rand = np.random.randint(m, size=m) # With replacement (may be duplicate)
        X_bagging = X[idx_rand]
        y_bagging = y[idx_rand]

        return X_bagging, y_bagging


    # TRAIN Ensemble
    def fit(self, X, y):
        # Do Bagging for training each ELM
        for i in range(0,self.n_estimator,1):
            # Resampling dataset for each elm
            X_bagging, y_bagging = self.bagging(X, y)
            # Train Each ELMs
            self.classifier[i].fit(X_bagging, y_bagging)

        return self


    # PREDICTION
    def predict(self, X):
        # If num_estimator = 1, do traditional predicition
        if self.n_estimator == 1:
            y_predicted = self.classifier[0].predict(X)
            return y_predicted
        # Ensemble predicition
        else:
            # Similar to do ... while ...
            predictions = self.classifier[0].predict(X)
            for i in range(1,self.n_estimator,1):
                y_predicted = self.classifier[i].predict(X)
                predictions = np.vstack((predictions, y_predicted)) # Stack predicition

            #print 'Each ELM predicition: ', np.array(predictions[:,0])
            ens_predictions = []
            for i in range(0,np.shape(X)[0],1):
                vote_count = np.bincount(predictions[:,i]) # Count prediction each ensemble vote
                ens_predictions.append(np.argmax(vote_count)) # Return maximum value in voting

            #print 'ELM Ensemble Predicition', ens_predictions

            return np.array(ens_predictions)


    # PREDICT MULTI LABEL
    def predict_multilabel(self, X, confidence=0):
        # If num_estimator = 1, do traditional predicition
        if self.n_estimator == 1:
            y_predicted = self.classifier[0].predict(X)
            return y_predicted
        # Ensemble predicition
        else:
            # Similar to do ... while ...
            #print X
            #print 'SELF ELM[0] PRECDICT: ', self.elms[0].binarizer.classes_
            predictions = self.classifier[0].predict(X)
            for i in range(1,self.n_estimator,1):
                #print 'SELF ELM[i] PRECDICT: ', self.elms[i].binarizer.classes_
                y_predicted = self.classifier[i].predict(X)
                predictions = np.vstack((predictions, y_predicted)) # Stack predicition

            #print 'Each ELM predicition: ', np.array(predictions[:,0], dtype='int32')
            #print 'ALL PREDICTIONS: ', predictions
            ens_predictions = []
            predictions = np.array(predictions, dtype='int32') # Cast predicition to integer 32
            for i in range(0,np.shape(X)[0],1):
                vote_count = np.bincount(predictions[:,i]) # Count prediction each ensemble vote
                #print 'VOTE_COUNT: ', vote_count
                # TODO: Should modify following code
                max_count = np.max(vote_count) # Get max count on each data
                #print 'MAX_CLASS: ', max_count
                # TODO: Trying to threshold one percent of total class predicted
                threshold = np.ceil(confidence*len(vote_count)) # Generate dynamic threshold
                base_line = max_count - threshold
                # Filter votes
                votes = []
                for ii in range(0,len(vote_count),1):
                    # If this bin count or classes greater or equal than threshold, select as target
                    if vote_count[ii] >= base_line:
                        votes.append(ii)

                ens_predictions.append(votes)

            return ens_predictions


    # PREDICITION Accuracy
    def score(self, X, y):
        from sklearn.metrics import accuracy_score
        return accuracy_score(y, self.predict(X))


    # TODO: PREDICTION Multilabel Evaluation Metrices
    def score_multilabel(self, X, y, confidence=0):
        predicted = self.predict_multilabel(X, confidence=confidence)
        ## EVALUATION METRICES (Tsoumakas 2007)
        # Accuracy, Precision, and Recall (Godbole and Sarawagi)
        n_D = X.shape[0]
        sigma_acc = 0
        sigma_prec = 0
        sigma_recall = 0

        for ii in range(0,n_D,1):
            n_intersect = np.intersect1d(np.array(y[ii], dtype='int32'), predicted[ii])
            n_union = np.union1d(np.array(y[ii], dtype='int32'), predicted[ii])
            sigma_acc += len(n_intersect)/len(n_union)
            sigma_prec += len(n_intersect)/len(predicted[ii])
            sigma_recall += len(n_intersect)/len(y[ii])

        acc = (1./n_D) * sigma_acc
        prec = (1./n_D) * sigma_prec
        recall = (1./n_D) * sigma_recall

        return acc, prec, recall