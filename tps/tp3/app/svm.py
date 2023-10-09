import numpy as np
import matplotlib.pyplot as plt

from settings import settings

class SVM():
    def __init__(self):
        self.C = 2
        self.w = 0
        self.b = 0


    def hingeloss(self, w, b, x, y):
        # Regularizer term
        reg = 0.5 * (w * w)

        for i in range(x.shape[0]):
            # Optimization term
            opt_term = y[i] * ((np.dot(w, x[i])) + b)

            # calculating loss
            loss = reg + self.C * max(0, 1-opt_term)
        return loss[0][0]
    

    def fit(self, X, Y, batch_size=100, learning_rate=0.001, epochs=1000):
        number_of_features = X.shape[1]
        number_of_samples = X.shape[0]

        c = self.C

        ids = np.arange(number_of_samples)
        np.random.shuffle(ids)

        w = np.zeros((1, number_of_features))
        b = 0
        losses = []

        # Gradient Descent logic
        for i in range(epochs):
            # Calculating the Hinge Loss
            l = self.hingeloss(w, b, X, Y)

            # Appending all losses 
            losses.append(l)
            
            # Starting from 0 to the number of samples with batch_size as interval
            for batch_initial in range(0, number_of_samples, batch_size):
                gradw = 0
                gradb = 0

                for j in range(batch_initial, batch_initial + batch_size):
                    if j < number_of_samples:
                        x = ids[j]
                        ti = Y[x] * (np.dot(w, X[x].T) + b)

                        if ti > 1:
                            gradw += 0
                            gradb += 0
                        else:
                            # w.r.t w 
                            gradw += c * Y[x] * X[x]
                            # w.r.t b
                            gradb += c * Y[x]

                # Updating weights and bias
                w = w - learning_rate * w + learning_rate * gradw
                b = b + learning_rate * gradb
        
        self.w = w
        self.b = b

        return self.w, self.b, losses 
    
    def predict(self, X):
        
        prediction = np.dot(X, self.w[0]) + self.b # w.x + b
        return np.sign(prediction)
    

    def visualize_dataset(self, X, y):
        plt.scatter(X[:, 0], X[:, 1], c=y)


    def visualize_svm(self, X_test, y_test, w, b):

        def get_hyperplane_value(x, w, b, offset):
            return (-w[0][0] * x + b + offset) / w[0][1]

        fig = plt.figure()
        ax = fig.add_subplot(1,1,1)
        plt.scatter(X_test[:, 0], X_test[:, 1], marker="o", c=y_test)

        x0_1 = np.amin(X_test[:, 0])
        x0_2 = np.amax(X_test[:, 0])

        x1_1 = get_hyperplane_value(x0_1, w, b, 0)
        x1_2 = get_hyperplane_value(x0_2, w, b, 0)

        x1_1_m = get_hyperplane_value(x0_1, w, b, -1)
        x1_2_m = get_hyperplane_value(x0_2, w, b, -1)

        x1_1_p = get_hyperplane_value(x0_1, w, b, 1)
        x1_2_p = get_hyperplane_value(x0_2, w, b, 1)

        ax.plot([x0_1, x0_2], [x1_1, x1_2], "y--")
        ax.plot([x0_1, x0_2], [x1_1_m, x1_2_m], "k")
        ax.plot([x0_1, x0_2], [x1_1_p, x1_2_p], "k")

        ax.set_xlim([0, 5])
        ax.set_ylim([0, 5])

        plt.show()