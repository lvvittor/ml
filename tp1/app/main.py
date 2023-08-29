import pandas as pd
import numpy as np
from settings import settings
from naive_bayes import NaiveBayes
from services import PlotService

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
        title = row['title'].split(" ")
        for word in title:
            word = word.lower()
            if word not in word_occurrences[row["category"]]:
                word_occurrences[row["category"]][word] = 0
            word_occurrences[row["category"]][word] += 1

    # Predict value
    actual_values = []
    actual_values_probabilities = []
    predicted_values = []

    categories_len = len(categories)
    for i_, row in test_data.iterrows():
        actual_values.append(row['category'])
        probabilities = []
        title = row['title'].split(" ")
        for category in categories:
            p = amount_news[category]/train_data.shape[0]
            k = len(word_occurrences[category].keys())
            for word in title:
                word = word.lower()
                if word in word_occurrences[category]:
                    amount_word = word_occurrences[category][word]
                else:
                    amount_word = 0

                p *= (amount_word + 1)/(np.array(list(word_occurrences[category].values())).sum() + k)
            
            probabilities.append(p)
        predicted_values.append(categories[np.argmax(probabilities)])
        # Turn the probabilities into a probability distribution
        probabilities = np.array(probabilities) / np.sum(probabilities)
        # Save the probability of the correct category
        actual_values_probabilities.append(probabilities[np.where(categories == row['category'])[0][0]])

    confusion_matrix = np.zeros((categories_len, categories_len), dtype=int)

    for actual, predicted in zip(actual_values, predicted_values):
        actual_idx = np.where(categories == actual)[0][0]
        predicted_idx = np.where(categories == predicted)[0][0]
        confusion_matrix[actual_idx][predicted_idx] += 1

    print(confusion_matrix)

    plot_service = PlotService(df)

    plot_service.confusion_matrix(confusion_matrix, categories)

    evaluation_measures = {}
    for category in categories:
        evaluation_measures[category] = {
            "TP": 0,
            "TN": 0,
            "FP": 0,
            "FN": 0,
            "Accuracy": 0,
            "Precision": 0,
            "Tasa_FP": 0,
            "Recall": 0,
            "F1": 0
        }
    
    # Calculate TP, TN, FP and FN
    for i in range(categories_len):
        for j in range(categories_len):
            category_actual = categories[i]
            category_predicted = categories[j]
            
            cell_value = confusion_matrix[i][j]
            
            if category_actual == category_predicted:
                evaluation_measures[category_actual]["TP"] += cell_value
                for category_other in categories:
                    if category_other != category_actual:
                        evaluation_measures[category_other]["TN"] += cell_value
            else:
                evaluation_measures[category_actual]["FN"] += cell_value
                evaluation_measures[category_predicted]["FP"] += cell_value


    for category, measures in evaluation_measures.items():
        TP = measures["TP"]
        TN = measures["TN"]
        FP = measures["FP"]
        FN = measures["FN"]
        measures["Accuracy"] = (TP + TN)/(TP+TN+FP+FN)
        measures["Precision"] = (TP)/(TP+FP)
        measures["Tasa_FP"] = (FP)/(TN+FP)
        measures["Recall"] = (TP)/(TP+FN)
        measures["F1"] = 2*measures["Precision"]*measures["Recall"]/(measures["Precision"]+measures["Recall"])
        print(f"Categoría: {category}")
        print(f"TP: {measures['TP']:d}")
        print(f"TN: {measures['TN']:d}")
        print(f"FP: {measures['FP']:d}")
        print(f"FN: {measures['FN']:d}")
        print(f"Accuracy: {measures['Accuracy']:.3f}")
        print(f"Precision: {measures['Precision']:.3f}")
        print(f"Tasa_FP: {measures['Tasa_FP']:.3f}")
        print(f"Recall: {measures['Recall']:.3f}")
        print(f"F1: {measures['F1']:.3f}")
        print()
    
    # Plot ROC curve
    TVPs = {}
    TFPs = {}

    for category in categories:
        TVPs[category] = []
        TFPs[category] = []

    for threshold in range(1, 11):
        threshold /= 10

        predictions = np.empty(len(actual_values), dtype=object)
        
        for i in range(len(actual_values)):
            # Si la prediccion sobre la categoria correcta tiene una probabilidad mayor o 
            # igual al threshold, entonces la prediccion es la categoria correcta.
            if actual_values_probabilities[i] >= threshold:
                predictions[i] = actual_values[i]
            # Si no, la prediccion es la categoria con mayor probabilidad.
            else:
                predictions[i] = predicted_values[i]

        for category in categories:
            confusion_matrix = np.zeros((2, 2), dtype=int)
            
            # category: "salud"
            # actual: "salud"
            # predicted: "salud"
            # actual_idx: 0, predicted_idx: 0

            # category: "salud"
            # actual: "salud"
            # predicted: "deportes"
            # actual_idx: 0, predicted_idx: 1

            # category: "salud"
            # actual: "deportes"
            # predicted: "deportes"
            # actual_idx: 1, predicted_idx: 0

            # category: "salud"
            # actual: "deportes"
            # predicted: "comida"
            # actual_idx: 1, predicted_idx: 1

            # Generamos la matriz de confusion para la categoria y threshol actuales
            for actual, predicted in zip(actual_values, predictions):
                if actual == category:
                    actual_idx = 0
                else:
                    actual_idx = 1
                
                if predicted == category:
                    predicted_idx = 0
                else:
                    predicted_idx = 1
                
                confusion_matrix[actual_idx][predicted_idx] += 1

            print("CATEGORY:", category)
            print("THRESHOLD:", threshold)
            print(confusion_matrix)
            print()
            
            TP = confusion_matrix[0][0]
            TN = confusion_matrix[1][1]
            FP = confusion_matrix[1][0]
            FN = confusion_matrix[0][1]

            TVP = TP/(TP+FN)
            TFP = FP/(FP+TN)

            TVPs[category].append(TVP)
            TFPs[category].append(TFP)
        
    plot_service.roc_curve(categories, TVPs, TFPs)


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