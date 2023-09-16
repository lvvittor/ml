import pandas as pd
import numpy as np

class KNN:
    def __init__(self, train_data: pd.DataFrame):
        self.data = train_data


    def predict(self, X: pd.Series, K: int = 5, weighted: bool = False):
        """
        Args:
            X (pd.Series): The ROW of data to predict
            K (int, optional): The number of neighbors to consider. Defaults to 5.
        """
        data_distances = self.data.copy()

        # Define a new column and initialize it with some default value
        data_distances["distance"] = None

        for index, row in data_distances.iterrows():
            data_distances.at[index, "distance"] = self._calculate_distance(row, X)

        data_distances = data_distances.sort_values(by="distance", ascending=True)

        nearest_neighbors = data_distances.head(K)

        # Neighbors = [{"1": 0.5, "2": 0.7", "1": 0.3, "2": 0.4, "1": 1.2}]
        # Regular KNN: "1" => 3 ; "2" => 2 => predicted = "1"
        # Weighted KNN:
        #   "1" => (1/0.5^2) + (1/0.3^2) + (1/1.2^2) = 4.44 ;
        #   "2" => (1/0.7^2) + (1/0.4^2) = 3.06 
        #   => predicted = "1"
        prediction = None

        if weighted:
            weighted_dict = {}

            for index, row in nearest_neighbors.iterrows():
                if row["distance"] == 0:
                    prediction = row["Star Rating"]
                    break

                if row["Star Rating"] not in weighted_dict:
                    weighted_dict[row["Star Rating"]] = 1/np.square(row["distance"])
                else:
                    weighted_dict[row["Star Rating"]] += 1/np.square(row["distance"])
            
            if prediction is None:
                prediction = list(weighted_dict.keys())[np.argmax(list(weighted_dict.values()))]
        else:
            prediction = nearest_neighbors["Star Rating"].value_counts().idxmax()

        return prediction
    

    def _calculate_distance(self, row, X: pd.Series):
        return np.linalg.norm(row[["wordcount", "titleSentiment", "sentimentValue"]] - X)
