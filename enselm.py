from elm import ELMClassifier
import numpy as np

def adaptive_ens_elm(train_data, train_target, num_elms):
    elms = []
    w_elm = np.zeros(num_elms)
    for i in range(1,num_elms,1):
        # Create M random ELMs
        elms.append(ELMClassifier()) # Create each elm using default parameter
        # Train each of the ELMs individually on the training data
        elms[i].fit(train_data, train_target)

    w_elm[:] = 1. / num_elms # Initialize ELM weight (each w_i) to 1/M

    # While t < t_end do
    # TODO: Incomplete


class BaggingELMClassifier():
    """
    Ensemble of ELM using Bagging method

    default number of ELMs is 10
    you can change by passing a defined 'n_estimators' parameter
    """
    def __init__(self, n_estimators=10):
        self.elms = []
        self.n_estimators = n_estimators

        for i in range(0,n_estimators,1):
            self.elms.append(ELMClassifier()) # Create each ELM using default parameter


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
        for i in range(0,self.n_estimators,1):
            # Resampling dataset for each elm
            X_bagging, y_bagging = self.bagging(X, y)
            # Train Each ELMs
            self.elms[i].fit(X_bagging, y_bagging)

        return self


    # PREDICTION
    def predict(self, X):
        # If num_estimator = 1, do traditional predicition
        if self.n_estimators == 1:
            y_predicted = self.elms[0].predict(X)
            return y_predicted
        # Ensemble predicition
        else:
            # Similar to do ... while ...
            predictions = self.elms[0].predict(X)
            for i in range(1,self.n_estimators,1):
                y_predicted = self.elms[i].predict(X)
                predictions = np.vstack((predictions, y_predicted)) # Stack predicition

            #print 'Each ELM predicition: ', np.array(predictions[:,0])
            ens_predictions = []
            for i in range(0,np.shape(X)[0],1):
                vote_count = np.bincount(predictions[:,i]) # Count prediction each ensemble vote
                ens_predictions.append(np.argmax(vote_count)) # Return maximum value in voting

            #print 'ELM Ensemble Predicition', ens_predictions

            return np.array(ens_predictions)


    # PREDICT MULTI LABEL
    def predict_multilabel(self, X, confidence=0.1):
        # If num_estimator = 1, do traditional predicition
        if self.n_estimators == 1:
            y_predicted = self.elms[0].predict(X)
            return y_predicted
        # Ensemble predicition
        else:
            # Similar to do ... while ...
            predictions = self.elms[0].predict(X)
            for i in range(1,self.n_estimators,1):
                y_predicted = self.elms[i].predict(X)
                predictions = np.vstack((predictions, y_predicted)) # Stack predicition

            #print 'Each ELM predicition: ', np.array(predictions[:,0])
            ens_predictions = []
            for i in range(0,np.shape(X)[0],1):
                vote_count = np.bincount(predictions[:,i]) # Count prediction each ensemble vote
                # TODO: Should modify following code
                max_count = np.argmax(vote_count) # Get max count on each data
                # TODO: Trying to threshold one percent of total class predicted
                threshold = np.ceil(confidence*len(vote_count)) # Generate dynamic threshold
                base_line = max_count - threshold
                # Filter votes
                votes = []
                jj = 0
                for ii in range(0,len(vote_count),1):
                    # If this bin count or classes greater or equal than threshold, select as target
                    if vote_count[ii] >= base_line:
                        votes.append(ii)
                    ii += 1

                ens_predictions.append(votes)

            return ens_predictions


    # PREDICITION Accuracy
    def score(self, X, y):
        from sklearn.metrics import accuracy_score
        return accuracy_score(y, self.predict(X))