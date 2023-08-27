import pandas as pd
import numpy as np
from settings import settings
from naive_bayes import NaiveBayes

def exercise_1():
    df = pd.read_excel(f"{settings.Config.data_dir}/PreferenciasBritanicos.xlsx")
    df.columns = ["scons", "beer", "whiskey", "oatmeal", "football", "nationality"]

    naiveBayesClassifier = NaiveBayes(df)

    # Test data
    x1 = [1, 0, 1, 1, 0]
    x2 = [0, 1, 1, 0, 1]

    naiveBayesClassifier.train()

    x1_likelihoods = naiveBayesClassifier.predict(x1)
    x2_likelihoods = naiveBayesClassifier.predict(x2)

    x1_predicted_nationality = df["nationality"].unique()[np.argmax(x1_likelihoods)]
    x2_predicted_nationality = df["nationality"].unique()[np.argmax(x2_likelihoods)]

    print(x1_likelihoods)
    print(f'Predicted nationality for x1 {x1}: {x1_predicted_nationality}')
    print(x2_likelihoods)
    print(f'Predicted nationality for x2 {x2}: {x2_predicted_nationality}')


def exercise_2():
    pass


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