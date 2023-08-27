import pandas as pd
import numpy as np
from settings import settings
from naive_bayes import NaiveBayes

def exercise_1():
    df = pd.read_excel(f"{settings.Config.data_dir}/PreferenciasBritanicos.xlsx")
    df.columns = ["scons", "beer", "whiskey", "oatmeal", "football", "nationality"]

    # Train data
    X = df.drop("nationality", axis=1)
    Y = df["nationality"]

    naiveBayesClassifier = NaiveBayes(X, Y)

    # Test data
    x1 = [1, 0, 1, 1, 0]
    x2 = [0, 1, 1, 0, 1]

    naiveBayesClassifier.train()

    x1_likelihoods = naiveBayesClassifier.predict(x1)
    x2_likelihoods = naiveBayesClassifier.predict(x2)

    x1_predicted_nationality = Y.unique()[np.argmax(x1_likelihoods)]
    x2_predicted_nationality = Y.unique()[np.argmax(x2_likelihoods)]

    print(x1_likelihoods)
    print(f'Predicted nationality for x1 {x1}: {x1_predicted_nationality}')
    print(x2_likelihoods)
    print(f'Predicted nationality for x2 {x2}: {x2_predicted_nationality}')


def exercise_2():
    # Note: we filtered out the "Noticias Destacadas" category
    df = pd.read_excel(f"{settings.Config.data_dir}/Noticias_argentinas.xlsx", usecols=[1,3])
    df.columns = ["title", "category"]

    # Filter out rows with category "Noticias destacadas"
    df = df[
        (df["category"] == "Salud") |
        (df["category"] == "Ciencia y Tecnologia") |
        (df["category"] == "Entretenimiento") |
        (df["category"] == "Economia")
    ]

    categories = df["category"].unique()
    print(categories)

    # Randomize the rows
    df = df.sample(frac=1, random_state=42)  # Use a specific random state for reproducibility

    # Calculate the index to split at
    split_index = int(0.8 * len(df))

    # Split the DataFrame into 80% and 20%
    train_data = df.iloc[:split_index]
    test_data = df.iloc[split_index:]
    print(train_data.head())
    print(test_data.head())

    X = train_data.drop('category', axis=1)
    Y = train_data['category']

    word_occurrences = {}
    amount_news = {}
    for category in categories:
        word_occurrences[category] = {}
        amount_news[category] = 0

    # Calculate occurrencies of a word in a title for each category
    for _, row in train_data.iterrows():
        amount_news[row['category']] += 1
        title = list(set(row['title'].split(" ")))
        for word in title:
            word = word.lower()
            if word not in word_occurrences[row["category"]]:
                word_occurrences[row["category"]][word] = 0
            word_occurrences[row["category"]][word] += 1

    print(word_occurrences)

    # Predict value
    actual_values = []
    predicted_values = []

    k = len(categories)
    for _, row in test_data.iterrows():
        actual_values.append(row['category'])
        probabilities = []
        for category in categories:
            p = amount_news[category]/train_data.shape[0]
            title = list(set(row['title'].split(" ")))
            for word in title:
                if word in word_occurrences[row["category"]]:
                    amount_word = word_occurrences[row["category"]][word]
                else:
                    amount_word = 0

                p *= (amount_word + 1)/(amount_news[category] + k)
            
            probabilities.append(p)
        predicted_values.append(categories[np.argmax(probabilities)])


    print(actual_values[:10])
    print(predicted_values[:10])
    count = 0
    for a, b in zip(actual_values, predicted_values):
        if a == b:
            count += 1

    count2 = 0
    for a in predicted_values:
        if a == 'Economia':
            count2 += 1

    print("Precision:", count/len(actual_values))
    print("Count2", count2/len(predicted_values))


    # # Convert X_test and Y_test to NumPy arrays
    # X_test = test_data.drop('category', axis=1).values
    # Y_test = test_data['category'].values

    # # Initialize variables for confusion matrix
    # num_classes = df['category'].unique().size
    # confusion = np.zeros((num_classes, num_classes), dtype=int)

    # print("Building confusion matrix...")
    # # Predict class labels for each test data point and update the confusion matrix
    # for x, true_label in zip(X_test, Y_test):
    #     likelihoods = naiveBayesClassifier.predict(x)
    #     predicted_class = np.argmax(likelihoods)
    #     confusion[true_label, predicted_class] += 1

    # # Print the confusion matrix
    # print("Confusion Matrix:")
    # print(confusion)


def exercise_3():
    pass


if __name__ == "__main__":
	match settings.exercise:
		case 1:
			exercise_1()
		case 2:
			exercise_2()
		case 3:
			exercise_3()

		case _:
			raise ValueError("Invalid exercise number")