import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

from settings import settings


class StepPerceptron():

    def __init__(
        self, learning_rate: float, inputs: np.array, expected_outputs: np.array
    ):
        """Constructor method

        Args:
            learning_rate (float): learning rate of the perceptron
            inputs (np.array): inputs of the perceptron (x_1, x_2, ..., x_n)
            expected_outputs (np.array): expected outputs of the perceptron (y_1, y_2, ..., y_n)
        """
        self.learning_rate = learning_rate
        # add bias x_0 = 1 to each input => (1, x_1, x_2, ..., x_n)
        self.inputs = np.insert(inputs, 0, 1, axis=1)
        self.expected_outputs = expected_outputs
        # first weight is the bias => (w_0, w_1, w_2, ..., w_n)
        self.weights = np.zeros(self.inputs.shape[1])
        
        # Data for plotting
        self.historical_weights = []
        self.historical_outputs = []

        # Momentum
        self.previous_deltas = np.zeros(self.weights.shape)

    
    def train(self, epochs = 1000):
        """
        Trains the perceptron for a given number of epochs

        Args:
            epochs (Optional[int]): number of epochs to train the perceptron. Defaults to 1000.

        Returns:
            int: number of epochs needed to converge
            bool: whether the perceptron converged or not
        """
        for epoch in range(epochs):
            # save the weights
            self.update_weights()
            self.historical_weights.append(self.weights)
            self.historical_outputs.append(self.get_outputs())

            if self.is_converged():
                break

        return epoch + 1, self.is_converged()

    
    def update_weights(self):
        deltas = self.compute_deltas()

        aux = deltas.copy()
        deltas += 0.9 * self.previous_deltas
        self.previous_deltas = aux

        self.weights = self.weights + np.sum(deltas, axis=0)


    def get_outputs(self):
        """Returns the perceptron's output for each input"""

        # Compute the perceptron's excitation for each input, including the sum of the bias
        excitations = np.dot(self.inputs, self.weights)

        # Apply the activation function to each element of the array
        return np.vectorize(self.activation_func)(excitations)


    def activation_func(self, value):
        return 1 if value >= 0 else -1  # step function


    def get_error(self):
        return np.sum(abs(self.expected_outputs - self.get_outputs()))


    def compute_deltas(self) -> np.array:
        # Get the difference between the expected outputs and the actual outputs
        output_errors = self.expected_outputs - self.get_outputs()
        # Compute the delta weights for each input
        deltas = self.learning_rate * output_errors.reshape(-1, 1) * self.inputs
        
        return deltas


    def is_converged(self):
        return self.get_error() <= 0


    def save_animation_frames(self, file_name = "step_perceptron"):
        # remove bias term
        _inputs = self.inputs[:, 1:]

        for i, (weights, outputs) in enumerate(
            zip(self.historical_weights, self.historical_outputs)
        ):
            # plot the points
            sns.scatterplot(
                x=_inputs[:, 0],
                y=_inputs[:, 1],
                hue=outputs,
                style=outputs,
                palette=["blue", "red"],
                marker="o",
            )

            xmin, xmax = np.min(_inputs[:, 0]), np.max(_inputs[:, 0])
            x = np.linspace(xmin - 100, xmax + 100, 1000)

            # w1*x + w2*y + w0 = 0 => y = -(w1*x + w0) / w2

            # w1*x + w2*y + w0 = 0 => y = -(w1*x + w0) / w2
            if weights[2] == 0:
                y = np.zeros(len(x))
            else:
                y = -(weights[1] * x + weights[0]) / weights[2]

            lineplot = sns.lineplot(x=x, y=y, color="black")

            plt.xlim([0, 5])
            plt.ylim([0, 5])
            plt.legend(markerscale=2)
            plt.title(f"Step Perceptron Epoch {i}")

            # save the plot to a file
            fig = lineplot.get_figure()
            fig.savefig(f"{settings.Config.out_dir}/{file_name}_{i}.png")

            # clear the current figure to prevent overlapping of plots
            plt.clf()


    def save_animation(self, file_name = "step_perceptron"):
        # remove bias term
        _inputs = self.inputs[:, 1:]

        fig, ax = plt.subplots()

        def update(i):
            ax.clear()

            weights, outputs = self.historical_weights[i], self.historical_outputs[i]

            # plot the points
            sns.scatterplot(
                x=_inputs[:, 0],
                y=_inputs[:, 1],
                hue=outputs,
                style=outputs,
                palette=["blue", "red"],
                marker="o",
            )

            xmin, xmax = np.min(_inputs[:, 0]), np.max(_inputs[:, 0])
            x = np.linspace(xmin - 100, xmax + 100, 1000)

            # w1*x + w2*y + w0 = 0 => y = -(w1*x + w0) / w2
            if weights[2] == 0:
                y = np.zeros(len(x))
            else:
                y = -(weights[1] * x + weights[0]) / weights[2]

            # plot the separating hyperplane
            ax.plot(x, y, c="k")

            ax.set_xlim([0, 5])
            ax.set_ylim([0, 5])
            ax.set_title(f"Step Perceptron Epoch {i}")

        anim = FuncAnimation(
            fig, update, frames=len(self.historical_weights), interval=500
        )

        anim.save(
            f"{settings.Config.out_dir}/{file_name}.gif", writer="imagemagick"
        )

        fig.clf()
    

    def __str__(self) -> str:
        output = "Expected - Actual\n"

        for expected, actual in zip(self.expected_outputs, self.get_outputs()):
            output += f"{expected:<10} {actual}\n"

        output += f"\nWeights: {self.weights}"

        return output
