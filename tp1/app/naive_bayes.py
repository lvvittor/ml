import numpy as np

class NaiveBayes:
    def __init__(self, df):
        self.attributes = df.drop('nationality', axis=1)
        self.nationalities = df['nationality']

        # Calculate probabilities of being from a nationality
        self.p_nationality = self.nationalities.value_counts() / len(self.nationalities)

        # Calculate all conditional probabilities, prob_conditional is a dictionary where each key is a column from the dataframe
        # and its value is another dictionary where each pair [key,value] is [(possible_value, nationality), P(possible_value | nationality)]
        # with this dataset possible_value is 0 or 1 and nationality I or E
        # e.x prob_conditional["scons"][(1,"I")] returns the probability of "scons" being 1 if the nationality is "I"
        self.prob_conditional = {}


    def train(self):
        for attribute in self.attributes.columns:
            self.prob_conditional[attribute] = {}

            for value in self.attributes[attribute].unique():
                for nationality in self.nationalities.unique():
                    k = len(self.nationalities.unique())
                    # Apply Laplace smoothing
                    prob = (((self.attributes[attribute] == value) & (self.nationalities == nationality)).sum() + 1) / ((self.nationalities == nationality).sum() + k)
                    self.prob_conditional[attribute][(value, nationality)] = prob


    def predict(self, x):
        likelihoods = []

        for nationality in self.nationalities.unique():
            likelihood = self.p_nationality[nationality]

            for i, attribute_value in enumerate(x):
                likelihood *= self.prob_conditional[self.attributes.columns[i]][(attribute_value, nationality)]

            likelihoods.append(likelihood)

        # Turn into propbabilities
        return likelihoods / np.sum(likelihoods)
