import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

from settings import settings

class PlotService:
    def __init__(self, df: pd.DataFrame):
        self.df = df

    def boxplot(self, column_names: list[str] = None, showfliers = True, filename = "boxplot.png"):
        """Given a list of columns, plot a boxplot for each column.

        Args:
            column_names (list[str]): List of column names to plot.
            showfliers (bool): Whether to show outliers in the boxplot.
            filename (str): Name of the file to save the boxplot.
        """

        # Save the boxplot to a file.
        if column_names:
            self.df.boxplot(column=column_names, showfliers=showfliers)
        else:
            # Standardize the columns so that they are all in the same scale.
            numerical_columns = self.df.select_dtypes(include=['int64', 'float64']).columns
            non_numerical_columns = self.df.select_dtypes(include=['object']).columns

            scaler = StandardScaler()
            scaled_numerical_data = scaler.fit_transform(self.df[numerical_columns])
            scaled_df = pd.DataFrame(scaled_numerical_data, columns=numerical_columns)
            scaled_df = pd.concat([scaled_df, self.df[non_numerical_columns]], axis=1)
            scaled_df.boxplot(showfliers=showfliers)

        plt.savefig(f"{settings.Config.out_dir}/{filename}")
        plt.close()
    

    def sex_bar_plot(self):
        # Count the occurrences of each category in the "sex" column
        sex_counts = self.df['sex'].value_counts()

        plt.bar(sex_counts.index, sex_counts.values)
        plt.xlabel('Sex')
        plt.ylabel('Count')
        plt.title('Distribution of Sex')
        plt.savefig(f"{settings.Config.out_dir}/sex_barplot.png")
        plt.close()
        
        
    def male_female_boxplot(self, column_names: list[str], showfliers = True):
        """Given a list of columns, plot two boxplot for each column one for male data and one for female data.

        Args:
            column_names (list[str]): List of column names to plot.
            showfliers (bool): Whether to show outliers in the boxplot.
        """
        male_data = self.df[self.df['sex'] == 'M']
        female_data = self.df[self.df['sex'] == 'F']

        for column in column_names:
            plt.figure()
            plt.boxplot([male_data[column], female_data[column]], labels=['Male', 'Female'], showfliers=showfliers)
            plt.title(f'{column} - Male vs Female')
            plt.xticks(rotation=45)
            plt.savefig(f'{settings.Config.out_dir}/{column}_malevsfemale.png')


    def calories_category_boxplot(self, column_names: list[str], showfliers = True):
        """Given a list of columns, plot a boxplot for each column with all calories category
        Categories:
            CATE1: Calories <= 1100
            CATE2: 1100 < Calories <= 1700
            CATE3: 1700 < Calories

        Args:
            column_names (list[str]): List of column names to plot.
            showfliers (bool): Whether to show outliers in the boxplot.
        """
        cate1 = self.df[self.df['calories'] <= 1100]
        cate2 = self.df[(self.df['calories'] > 1100) & (self.df['calories'] <= 1700)]
        cate3 = self.df[self.df['calories'] > 1700]

        for column in column_names:
            plt.figure()
            plt.boxplot([cate1[column], cate2[column], cate3[column]], labels=['CATE1', 'CATE2', "CATE3"], showfliers=showfliers)
            plt.title(f'{column} CATE')
            plt.xticks(rotation=45)
            plt.savefig(f'{settings.Config.out_dir}/{column}cate.png')
