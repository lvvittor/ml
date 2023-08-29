import numpy as np

class NaiveBayes:
    def __init__(self, X, Y):
        self.X = X
        self.Y = Y

        # Calculate probabilities of being from a nationality
        self.p_Y = self.Y.value_counts() / len(self.Y)

        # Create a dictionary to hold conditional probabilities
        self.prob_conditional = {}


    def train(self):
        for x in self.X.columns:
            self.prob_conditional[x] = {}

            for value in self.X[x].unique():
                for y in self.Y.unique():
                    k = len(self.Y.unique())
                    x_mask = (self.X[x] == value)
                    y_mask = (self.Y == y)
                    prob = (np.sum(x_mask & y_mask) + 1) / (np.sum(y_mask) + k)
                    self.prob_conditional[x][(value, y)] = prob
                    

    def predict(self, x_test):
        likelihoods = []

        for y in self.Y.unique():
            likelihood = self.p_Y[y]

            for i, x in enumerate(x_test):
                likelihood *= self.prob_conditional[self.X.columns[i]][(x, y)]

            likelihoods.append(likelihood)

        return np.array(likelihoods) / np.sum(likelihoods)
