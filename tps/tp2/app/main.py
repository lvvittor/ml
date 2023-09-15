import pandas as pd
import numpy as np
from settings import settings

def main():
	match settings.exercise:
		case 1:
			exercise_1()
		case 2:
			exercise_2()
		case _:
			raise ValueError("Invalid exercise number")


def exercise_1():
	pass


def exercise_2():
    df = pd.read_csv(f"{settings.Config.data_dir}/reviews_sentiment.csv", delimiter=";")
    if settings.verbose: print(df.head())
	
    print(df["titleSentiment"].isna().sum())
    # Discretize
	
    # --------- A -----------
    one_star_wordcount_mean = df[df["Star Rating"] == 1]["wordcount"].mean()
    print(f"Mean wordcount for 1 star reviews: {one_star_wordcount_mean}")
	
    # --------- B -----------
    # Randomize the rows
    df = df.sample(frac=1)

    # Calculate the index to split at
    split_index = int(0.7 * len(df))
    # Split the DataFrame into 70% train and 30% test
    train_data = df.iloc[:split_index]
    test_data = df.iloc[split_index:]

    if settings.verbose:
        print(train_data.head())
        print(test_data.head())



    

if __name__ == "__main__":
    main()
