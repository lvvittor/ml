import numpy as np

# Decision stump (ie. decision tree of height 1) used as weak classifier
class DecisionStump:
    def __init__(self):
        self.polarity = 1       # polarity of the stump (+1 or -1)
        self.feature_idx = None # index of feature used for splitting
        self.threshold = None   # threshold value for splitting
        self.alpha = None       # weight of the stump in the final [strong] classifier

    def predict(self, X):
        n_samples = X.shape[0]
        X_column = X[:, self.feature_idx]

        predictions = np.ones(n_samples) # initialize all predictions to 1

        # determine what side of the split will be marked as -1 depending on the polarity of the stump
        if self.polarity == 1:
            predictions[X_column < self.threshold] = -1
        else:
            predictions[X_column > self.threshold] = -1

        return predictions


class Adaboost:
    def __init__(self, n_clf = 5):
        self.n_clf = n_clf # number of [weak] classifiers
        self.clfs = []     # list of classifiers

    def fit(self, X, y):
        n_samples, n_features = X.shape

        # Initialize weights to 1/N
        w = np.full(n_samples, (1 / n_samples))

        # Iterate through classifiers
        for _ in range(self.n_clf):

            # create a new classifier
            clf = DecisionStump()
            min_error = float("inf")

            # greedy search to find best threshold and feature (ie. the one that minimizes error)
            for feature_i in range(n_features):
                X_column = X[:, feature_i]
                thresholds = np.unique(X_column) # get unique values in the column of the current feature

                for threshold in thresholds:
                    # predict with polarity 1
                    p = 1
                    predictions = np.ones(n_samples)
                    predictions[X_column < threshold] = -1 # values to the left of the split are marked as -1

                    # Error = sum of weights of misclassified samples
                    misclassified = w[y != predictions]
                    error = np.sum(misclassified)

                    # since error is a value between 0 and 1, change the polarity of the prediction if the stump is
                    # misclassifying more than half of the weighted samples (ie. make it predict the other way around)
                    if error > 0.5:
                        error = 1 - error
                        p = -1

                    # store the best stump parameters found so far
                    if error < min_error:
                        clf.polarity = p
                        clf.threshold = threshold
                        clf.feature_idx = feature_i
                        min_error = error

            # calculate alpha, and use EPS to avoid division by zero
            EPS = 1e-10
            clf.alpha = 0.5 * np.log((1.0 - min_error + EPS) / (min_error + EPS))

            # calculate predictions and update weights
            predictions = clf.predict(X)

            w *= np.exp(-clf.alpha * y * predictions)
            w /= np.sum(w) # normalize the weights so that they sum up to 1

            # save classifier
            self.clfs.append(clf)

    def predict(self, X):
        clf_preds = [clf.alpha * clf.predict(X) for clf in self.clfs]
        y_pred = np.sum(clf_preds, axis=0)
        y_pred = np.sign(y_pred)

        return y_pred


# Testing
if __name__ == "__main__":

    from sklearn import datasets
    from sklearn.model_selection import train_test_split

    def accuracy(y_true, y_pred):
        accuracy = np.sum(y_true == y_pred) / len(y_true)
        return accuracy

    data = datasets.load_breast_cancer()
    X, y = data.data, data.target

    # Adaboost assumes that the labels are -1 and 1, so we need to change the 0s to -1s
    y[y == 0] = -1

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=5
    )

    # Adaboost classification with 5 weak classifiers
    clf = Adaboost(n_clf=5)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    acc = accuracy(y_test, y_pred)
    print("Accuracy:", acc)