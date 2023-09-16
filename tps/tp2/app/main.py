import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

from settings import settings
from knn import KNN
from visualization import plot_confusion_matrix, plot_values_vs_variable
from metrics import calc_evaluation_measures

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
	
    print(f"KNN ; weighted = {settings.knn.weighted}\n")


    # --------- A -----------
    one_star_wordcount_mean = df[df["Star Rating"] == 1]["wordcount"].mean()
    print(f"Mean wordcount for 1 star reviews: {one_star_wordcount_mean}\n")

    if settings.verbose: print("Missing title sentiment values", df["titleSentiment"].isna().sum())
    
    # Fill missing titleSentiment values with textSentiment values
    df.loc[df["titleSentiment"].isna(), "titleSentiment"] = df[df["titleSentiment"].isna()]["textSentiment"]
    # Discretize the sentiment values (positive = 1, negative = 0)
    df['titleSentiment'] = df['titleSentiment'].apply(lambda sentiment: 1 if sentiment == "positive" else 0)
	
    # Select the columns to standardize
    columns_to_standardize = ["wordcount", "titleSentiment", "sentimentValue"]
    scaler = StandardScaler()
    # Fit and transform the selected columns
    df[columns_to_standardize] = scaler.fit_transform(df[columns_to_standardize])
	
    # Filter out `Review Title` and `Review Text` columns
    df = df.drop(columns=["Review Title", "Review Text", "textSentiment"])

    if settings.verbose: print(df.head())

    classes = [1,2,3,4,5] # possible star ratings

    # --------- B -----------
    # Randomize the rows
    df = df.sample(frac=1, random_state=42)

    pd.set_option('mode.chained_assignment', None) # FIXME: This is a hack to avoid a warning

    if not settings.knn.run_metrics:
        # Calculate the index to split at
        split_index = int(0.7 * len(df))
        # Split the DataFrame into 70% train and 30% test
        train_data = df.iloc[:split_index]
        test_data = df.iloc[split_index:]

        if settings.verbose:
            print(train_data.head())
            print(test_data.head())


        # --------- C -----------
        knn = KNN(train_data)

        # Predict the star rating for each row in the test data
        test_data["predicted_star_rating"] = test_data.apply(
            lambda row: knn.predict(
                X=row[["wordcount", "titleSentiment", "sentimentValue"]],
                K=settings.knn.k,
                weighted=settings.knn.weighted
            ),
            axis=1
        )

        if settings.verbose: print(test_data.head())


        # --------- D -----------
        # Print the confusion matrix
        confusion_matrix = np.zeros((len(classes), len(classes)), dtype=int)

        for _, row in test_data.iterrows():
            confusion_matrix[int(row["Star Rating"]) - 1][int(row["predicted_star_rating"]) - 1] += 1

        plot_confusion_matrix(confusion_matrix, classes)
        
        ev_measures = calc_evaluation_measures(classes, confusion_matrix)
        
        for rating in classes:
            print(f"Precision for {rating} stars: {ev_measures[rating]['precision']:.3f}")
    else:
        # --------- Data Split Analysis -----------

        split_indexes = [0.5, 0.6, 0.7, 0.8, 0.9]
        precisions = {}

        for percentage in split_indexes:
            precisions[percentage] = {rating: None for rating in classes}

        for percentage in split_indexes:
            split_index = int(percentage * len(df))
            train_data = df.iloc[:split_index]
            test_data = df.iloc[split_index:]

            knn = KNN(train_data)
        
            test_data["predicted_star_rating"] = test_data.apply(
                lambda row: knn.predict(
                    X=row[["wordcount", "titleSentiment", "sentimentValue"]],
                    K=settings.knn.k,
                    weighted=settings.knn.weighted
                ),
                axis=1
            )

            confusion_matrix = np.zeros((len(classes), len(classes)), dtype=int)

            for _, row in test_data.iterrows():
                confusion_matrix[int(row["Star Rating"]) - 1][int(row["predicted_star_rating"]) - 1] += 1
            
            plot_confusion_matrix(confusion_matrix, classes, filename=f"confusion_matrix_{percentage*100}.png")

            ev_measures = calc_evaluation_measures(classes, confusion_matrix)

            for rating in classes:
                precisions[percentage][rating] = ev_measures[rating]['precision']
            
        plot_values_vs_variable(precisions, split_indexes, classes, xlabel="Split percentage", ylabel="Precision", filename="precision_vs_split.png")


        # --------- K parameter Analysis -----------

        ks = [1, 3, 5, 7, 10]

        precisions = {}

        for k in ks:
            precisions[k] = {rating: None for rating in classes}

        for k in ks:
            split_index = int(0.7 * len(df)) # 70% split
            train_data = df.iloc[:split_index]
            test_data = df.iloc[split_index:]

            knn = KNN(train_data)
        
            test_data["predicted_star_rating"] = test_data.apply(
                lambda row: knn.predict(
                    X=row[["wordcount", "titleSentiment", "sentimentValue"]],
                    K=k,
                    weighted=settings.knn.weighted
                ),
                axis=1
            )

            confusion_matrix = np.zeros((len(classes), len(classes)), dtype=int)

            for _, row in test_data.iterrows():
                confusion_matrix[int(row["Star Rating"]) - 1][int(row["predicted_star_rating"]) - 1] += 1
            
            plot_confusion_matrix(confusion_matrix, classes, filename=f"confusion_matrix_k_{k}.png")

            ev_measures = calc_evaluation_measures(classes, confusion_matrix)

            for rating in classes:
                precisions[k][rating] = ev_measures[rating]['precision']
            
        plot_values_vs_variable(precisions, ks, classes, xlabel="K", ylabel="Precision", filename="precision_vs_k.png")


if __name__ == "__main__":
    main()
