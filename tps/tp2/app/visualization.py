import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from settings import settings

def plot_confusion_matrix(confusion_matrix: list[list[int]], categories: np.array, filename = "confusion_matrix.png"):
        """Given a matrix and a list o labels, plot the confusion_matrix.

        Args:
            confusion_matrix (list[list[int]]): Confusion Matrix.
            categories (np.array): Columns used in the confusion matrix
            showfliers (bool): Whether to show outliers in the boxplot.
            filename (str): Name of the file to save the boxplot.
        """
        row_sums = confusion_matrix.sum(axis=1)
        normalized_confusion_matrix = confusion_matrix / row_sums[:, np.newaxis]

        plt.figure(figsize=(8, 6))

        # FIXME: only showing annotations on the first row, this is probably a bug
        # sns.heatmap(normalized_confusion_matrix, annot=True, fmt=".4f", cmap="Blues",
        #             xticklabels=categories, yticklabels=categories)
        
        # Create a heatmap
        plt.imshow(normalized_confusion_matrix, cmap='Blues')
        # Add annotations for each cell
        for i in range(normalized_confusion_matrix.shape[0]):
            for j in range(normalized_confusion_matrix.shape[1]):
                color = 'white' if normalized_confusion_matrix[i, j] >= 0.5 else 'black'
                plt.text(j, i, f'{normalized_confusion_matrix[i, j]:.4f}', ha='center', va='center', color=color)
        # Add a color bar to show the scale of values
        plt.colorbar()
        # Add x-axis and y-axis tick labels
        plt.xticks(range(normalized_confusion_matrix.shape[1]), labels=categories)
        plt.yticks(range(normalized_confusion_matrix.shape[0]), labels=categories)

        plt.xlabel("Predicted", fontsize=14)
        plt.ylabel("Real", fontsize=14)
        plt.tight_layout()
        plt.savefig(f"{settings.Config.out_dir}/{filename}")
        # plt.show()
        plt.close()


def plot_values_vs_variable(values: dict, variables: list, classes: list, errors: dict,  xlabel: str = "variable", ylabel: str = "value", filename = "values_vs_variable.png"):
    plt.figure(figsize=(8, 6))

    # Get the default color cycle
    prop_cycle = plt.rcParams['axes.prop_cycle']
    colors = prop_cycle.by_key()['color']

    for i, _class in enumerate(classes):
        data = [values[var][_class] for var in variables]
        error = [errors[var][_class] for var in variables]
        plt.errorbar(variables, data, yerr=error, fmt="o", alpha=0.4, capsize=15, color=colors[i])
        plt.plot(variables, data, label=str(_class), marker="o", color=colors[i])

    plt.xlabel(xlabel, fontsize=14)
    plt.ylabel(ylabel, fontsize=14)
    plt.tight_layout()
    plt.legend()
    plt.savefig(f"{settings.Config.out_dir}/{filename}")
    plt.close()


def plot_node_amt_vs_accuracy(node_amt: list, accuracies: list, errors: list = None, filename = "node_amt_vs_accuracy.png"):
    plt.figure(figsize=(8, 6))
    if errors is not None:
        plt.errorbar(node_amt, accuracies, yerr=errors, fmt='o', capsize=4, markersize=6, color="black")
        plt.plot(node_amt, accuracies) # connect markers with straight lines
    else:
        plt.plot(node_amt, accuracies, marker="o")
    plt.xlabel("Node amount", fontsize=14)
    plt.ylabel("Accuracy", fontsize=14)
    plt.tight_layout()
    plt.grid()
    plt.savefig(f"{settings.Config.out_dir}/{filename}")
    plt.clf()
    plt.close()
