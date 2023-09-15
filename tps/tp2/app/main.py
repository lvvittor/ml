import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

from settings import settings
from knn import KNN
from visualization import plot_confusion_matrix

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
	
    print("KNN ; weighted = ", settings.knn_weighted)
	
    # --------- A -----------
    one_star_wordcount_mean = df[df["Star Rating"] == 1]["wordcount"].mean()
    print(f"Mean wordcount for 1 star reviews: {one_star_wordcount_mean}")

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

    # --------- B -----------
    # Randomize the rows
    df = df.sample(frac=1, random_state=42)

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
    pd.set_option('mode.chained_assignment', None) # FIXME: This is a hack to avoid a warning

    K = 4
	# Predict the star rating for each row in the test data
    test_data["predicted_star_rating"] = test_data.apply(
		lambda row: knn.predict(
            X=row[["wordcount", "titleSentiment", "sentimentValue"]],
            K=K,
            weighted=settings.knn_weighted
        ),
		axis=1
	)

    if settings.verbose: print(test_data.head())

    predicted_incorrect = 0
    for _, row in test_data.iterrows():
        if row["predicted_star_rating"] != row["Star Rating"]: predicted_incorrect += 1

    print(f"Predicted incorrect: {predicted_incorrect}")
	

    # --------- D -----------
    # Print the confusion matrix
    classes = [1,2,3,4,5]
    confusion_matrix = np.zeros((len(classes), len(classes)), dtype=int)

    for _, row in test_data.iterrows():
        confusion_matrix[int(row["Star Rating"]) - 1][int(row["predicted_star_rating"]) - 1] += 1

    plot_confusion_matrix(confusion_matrix, classes)
	
    # Calculate TP, TN, FP and FN
    evaluation_measures = {}
    for category in classes:
        evaluation_measures[category] = {
            "TP": 0,
            "TN": 0,
            "FP": 0,
            "FN": 0,
        }
    for i in range(len(classes)):
        for j in range(len(classes)):
            category_actual = classes[i]
            category_predicted = classes[j]
            
            cell_value = confusion_matrix[i][j]
            
            if category_actual == category_predicted:
                evaluation_measures[category_actual]["TP"] += cell_value
                for category_other in classes:
                    if category_other != category_actual:
                        evaluation_measures[category_other]["TN"] += cell_value
            else:
                evaluation_measures[category_actual]["FN"] += cell_value
                evaluation_measures[category_predicted]["FP"] += cell_value
    
    for rating, measures in evaluation_measures.items():
        TP = measures["TP"]
        TN = measures["TN"]
        FP = measures["FP"]
        FN = measures["FN"]
        accuracy = (TP + TN)/(TP+TN+FP+FN)
        precision = (TP)/(TP+FP)
        fp_rate = (FP)/(TN+FP)
        recall = (TP)/(TP+FN)
        f1_score = 2*precision*recall/(precision+recall)
        print(f"Rating: {rating} stars")
        # print(f"TP: {measures['TP']:d}")
        # print(f"TN: {measures['TN']:d}")
        # print(f"FP: {measures['FP']:d}")
        # print(f"FN: {measures['FN']:d}")
        # print(f"Accuracy: {accuracy:.3f}")
        print(f"Precision: {precision:.3f}")
        # print(f"Tasa_FP: {fp_rate:.3f}")
        # print(f"Recall: {recall:.3f}")
        # print(f"F1: {f1_score:.3f}")
        print()

if __name__ == "__main__":
    main()
