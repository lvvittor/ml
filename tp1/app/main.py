import pandas as pd
import random
import numpy as np
from settings import settings

def exercise_1():
    df = pd.read_excel(f"{settings.Config.data_dir}/PreferenciasBritanicos.xlsx")
    df.columns = ["scons", "beer", "whiskey", "oatmeal", "football", "nationality"]
    input_values = [1, 0, 1, 1, 0]

    x = df.drop('nationality', axis=1)
    y = df['nationality']

    # Calculate probabilities of being from a nationality
    p_nationality = y.value_counts() / len(y)

    # Calculate all conditional probabilities, prob_conditional is a dictionary where each key is a column from the dataframe
    # and its value is another dictionary where each pair [key,value] is [(possible_value, nationality), P(possible_value | nationality)]
    # with this dataset possible_value is 0 or 1 and nationality I or E
    # e.x prob_conditional["scons"][(1,"I")] returns the probability of "scons" being 1 if the nationality is "I"
    prob_conditional = {}

    for attribute in x.columns:
        prob_conditional[attribute] = {}

        for value in x[attribute].unique():
            for nationality in y.unique():
                prob = ((x[attribute] == value) & (y == nationality)).sum() / (y == nationality).sum()
                prob_conditional[attribute][(value, nationality)] = prob


    # Calculate the probability of input_values being from each nationality
    
    likelihoods = []

    for nationality in y.unique():
        likelihood = p_nationality[nationality] # TODO: this is P(vj), the probability of being from a certain nationality, check if it should be calculated using the dataset or 1/nationalities or no

        for i, attribute_value in enumerate(input_values):
            likelihood *= prob_conditional[x.columns[i]][(attribute_value, nationality)]

        likelihoods.append(likelihood)

    predicted_nationality = y.unique()[np.argmax(likelihoods)]
    print(likelihoods)
    print(f'Predicted nationality for input_values: {predicted_nationality}')




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