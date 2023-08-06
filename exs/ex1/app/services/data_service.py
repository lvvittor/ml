import pandas as pd
from settings import settings
from typing import Callable


class DataService:
    def __init__(self, df: pd.DataFrame):
        self.df = df

    def replace_invalid_values(self, column_names: list[str], replace_functions: list[Callable[[pd.DataFrame, pd.Series, str, int], int]]):
        """Given a list of columns and a list of replace functions, call the function to each column in the df.

        Args:
            column_names (list[str]): List of column names to replace.
            replace_functions (list[Callable]): List of replace functions to call.
        """
        for i, row in self.df.iterrows():
            for j, column in enumerate(column_names):
                new_value = replace_functions[j](self.df, row, column, i)
                if new_value is not None:
                    self.df.at[i, column] = new_value


        
        