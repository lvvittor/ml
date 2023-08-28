import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from settings import settings

class PlotService:
    def __init__(self, df: pd.DataFrame):
        self.df = df

    def confusion_matrix(self, confusion_matrix: list[list[int]], categories: np.array, showfliers = True, filename = "confusion_matrix.png"):
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
        sns.heatmap(normalized_confusion_matrix, annot=True, fmt=".4f", cmap="Blues",
                    xticklabels=categories, yticklabels=categories)
        plt.xlabel("Predicted")
        plt.ylabel("Real")
        plt.tight_layout()
        plt.show()

        plt.savefig(f"{settings.Config.out_dir}/{filename}")
        plt.close()
    