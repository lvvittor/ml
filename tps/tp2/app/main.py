import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from metrics import MetricsCalculator
from sklearn.model_selection import train_test_split

from settings import settings
from knn import KNN
from visualization import plot_confusion_matrix, plot_values_vs_variable, plot_node_amt_vs_accuracy
from metrics import calc_evaluation_measures, MetricsCalculator
from decision_tree import DecisionTree
from random_forest import RandomForest

def main():
	match settings.exercise:
		case 1:
			exercise_1()
		case 2:
			exercise_2()
		case _:
			raise ValueError("Invalid exercise number")


def exercise_1():
    # Read the CSV file into a DataFrame and clean the data (remove the rows with missing values)
    df = pd.read_csv(f"{settings.Config.data_dir}/german_credit.csv", delimiter=",")
    # split the data into features and target variable where the target column is Creditability
    X = df.drop("Creditability", axis=1).to_numpy()
    y = df["Creditability"].to_numpy()
    classes = df["Creditability"].unique()

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=42)

    metrics = MetricsCalculator()

    if not settings.decision_tree.run_metrics:
        # Decision Tree
        decision_tree = DecisionTree()
        decision_tree.fit(X_train, y_train)
        predictions = decision_tree.predict(X_test)
        cm = metrics.confusion_matrix(y_test, predictions, 1)
        plot_confusion_matrix(cm, classes, filename="decision_tree_cm_v2.png")
        print(f"Decision tree node amount: {decision_tree.total_nodes}")
        print(f"Decision tree accuracy: {metrics.accuracy(y_test, predictions)}\n")

        # Random Forest
        random_forest = RandomForest()
        random_forest.fit(X_train, y_train)
        predictions = random_forest.predict(X_test)
        cm = metrics.confusion_matrix(y_test, predictions, 1)
        plot_confusion_matrix(cm, classes, filename="random_forest_cm_v2.png")
        print(f"Forest tree's node amount:", [tree.total_nodes for tree in random_forest.trees])
        print(f"Forest tree's AVG node amount:", np.mean([tree.total_nodes for tree in random_forest.trees]))
        print(f"Random forest accuracy: {metrics.accuracy(y_test, predictions)}")
    else:
        # Run node amount analysis
        min_samples_splits = [1, 2, 5, 10, 15, 20, 25, 35]
        max_depths = [20, 12, 10, 8, 6, 4, 3, 2]
        runs = 10

        print("---------- Running metrics for Decision Tree ---------\n")

        run_node_amounts = {}
        run_accuracies = {}

        for run in range(runs):
            print(f"Run {run+1} of {runs}")

            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4)

            for i in range(len(min_samples_splits)):
                decision_tree = DecisionTree(
                    min_samples_split=min_samples_splits[i],
                    max_depth=max_depths[i]
                )
                decision_tree.fit(X_train, y_train)
                predictions = decision_tree.predict(X_test)

                run_node_amounts[run] = run_node_amounts.get(run, []) + [decision_tree.total_nodes]
                run_accuracies[run] = run_accuracies.get(run, []) + [metrics.accuracy(y_test, predictions)]
        
        node_amounts = []
        accuracies = []
        errors = []
            
        for i in range (len(min_samples_splits)):
            node_amounts.append(np.mean([run_node_amounts[run][i] for run in range(runs)]))
            accuracies.append(np.mean([run_accuracies[run][i] for run in range(runs)]))
            errors.append(np.std([run_accuracies[run][i] for run in range(runs)]))

            print(f"Metrics for mins_samples={min_samples_splits[i]} ; max_depth={max_depths[i]}")

            print(f"Node amount: {node_amounts[i]}")
            print(f"Accuracy: {accuracies[i]}")
            print(f"Error: {errors[i]}")
            print()

        plot_node_amt_vs_accuracy(node_amounts, accuracies, errors=errors, filename="DT_node_amt_vs_accuracy.png")


        print("--------- Running metrics for Random Forest -----------\n")

        runs = 10
        run_node_amounts = {}
        run_accuracies = {}

        for run in range(runs):
            print(f"Run {run+1} of {runs}")

            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4)

            for i in range(len(min_samples_splits)):
                random_forest = RandomForest(
                    min_samples_split=min_samples_splits[i],
                    max_depth=max_depths[i]
                )
                random_forest.fit(X_train, y_train)
                predictions = random_forest.predict(X_test)

                avg_node_amount = np.mean([tree.total_nodes for tree in random_forest.trees])
                run_node_amounts[run] = run_node_amounts.get(run, []) + [avg_node_amount]
                run_accuracies[run] = run_accuracies.get(run, []) + [metrics.accuracy(y_test, predictions)]
        
        node_amounts = []
        accuracies = []
        errors = []
            
        for i in range (len(min_samples_splits)):
            node_amounts.append(np.mean([run_node_amounts[run][i] for run in range(runs)]))
            accuracies.append(np.mean([run_accuracies[run][i] for run in range(runs)]))
            errors.append(np.std([run_accuracies[run][i] for run in range(runs)]))

            print(f"Metrics for mins_samples={min_samples_splits[i]} ; max_depth={max_depths[i]}")

            print(f"Node amount: {node_amounts[i]}")
            print(f"Accuracy: {accuracies[i]}")
            print(f"Error: {errors[i]}")
            print()

        plot_node_amt_vs_accuracy(node_amounts, accuracies, errors=errors, filename="RF_node_amt_vs_accuracy.png")


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
            print(f"Accuracy for {rating} stars: {ev_measures[rating]['accuracy']:.3f}")
    else:
        # --------- Data Split Analysis -----------

        runs = 5
        run_accuracies = {}

        split_indexes = [0.5, 0.6, 0.7, 0.8, 0.9]
        precisions = {}
        errors = {}

        for percentage in split_indexes:
            precisions[percentage] = {rating: None for rating in classes}
            errors[percentage] = {rating: None for rating in classes}

        for run in range(runs):
            print(f"Split run {run+1} of {runs}")

            run_accuracies[run] = {}
            df = df.sample(frac=1) # randomize rows for each run

            for percentage in split_indexes:
                run_accuracies[run][percentage] = {}

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
                
                # plot_confusion_matrix(confusion_matrix, classes, filename=f"confusion_matrix_{percentage*100}.png")

                ev_measures = calc_evaluation_measures(classes, confusion_matrix)

                for rating in classes:
                    run_accuracies[run][percentage][rating] = ev_measures[rating]['accuracy']

        for percentage in split_indexes:
            for rating in classes:
                precisions[percentage][rating] = np.mean([run_accuracies[run][percentage][rating] for run in range(runs)])
                errors[percentage][rating] = np.std([run_accuracies[run][percentage][rating] for run in range(runs)])
            
        plot_values_vs_variable(precisions, split_indexes, classes,
            errors=errors,
            xlabel="Split percentage",
            ylabel="Accuracy",
            filename="precision_vs_split.png"
        )


        # --------- K parameter Analysis -----------

        run_accuracies = {}

        ks = [1, 3, 7, 10, 15]
        precisions = {}
        errors = {}

        for k in ks:
            precisions[k] = {rating: None for rating in classes}
            errors[k] = {rating: None for rating in classes}

        for run in range(runs):
            print(f"K run {run+1} of {runs}")

            run_accuracies[run] = {}
            df = df.sample(frac=1) # randomize rows for each run

            for k in ks:
                run_accuracies[run][k] = {}

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
                
                # plot_confusion_matrix(confusion_matrix, classes, filename=f"confusion_matrix_k_{k}.png")

                ev_measures = calc_evaluation_measures(classes, confusion_matrix)

                for rating in classes:
                    run_accuracies[run][k][rating] = ev_measures[rating]['accuracy']
        
        for k in ks:
            for rating in classes:
                precisions[k][rating] = np.mean([run_accuracies[run][k][rating] for run in range(runs)])
                errors[k][rating] = np.std([run_accuracies[run][k][rating] for run in range(runs)])
            
        plot_values_vs_variable(precisions, ks, classes,
            errors=errors,
            xlabel="K",
            ylabel="Accuracy",
            filename="precision_vs_k.png"
        )


if __name__ == "__main__":
    main()
