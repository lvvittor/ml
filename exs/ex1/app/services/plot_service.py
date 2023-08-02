import pandas as pd
from settings import settings
import matplotlib.pyplot as plt


class PlotService:
    def __init__(self, df: pd.DataFrame):
        self.df = df

    def boxplot(self, column_names: list[str] = None, showfliers = True):
        """Given a list of columns, plot a boxplot for each column.

        Args:
            column_names (list[str]): List of column names to plot.
            showfliers (bool): Whether to show outliers in the boxplot.
        """

        # Save the boxplot to a file.
        if column_names:
            self.df.boxplot(column=column_names, showfliers=showfliers)
        else:
            self.df.boxplot(showfliers=showfliers)
        plt.savefig(f"{settings.Config.out_dir}/boxplot.png")
        
        
