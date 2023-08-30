import pandas as pd
import numpy as np
from settings import settings
from naive_bayes import NaiveBayes
from services import BayesianNetwork



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
    df = pd.read_csv(f"{settings.Config.data_dir}/binary.csv")
    df['gre_d'] = pd.cut(df['gre'], bins=[-float('inf'), 500, float('inf')], labels=['gre < 500', 'gre ≥ 500'])
    df['gpa_d'] = pd.cut(df['gpa'], bins=[-float('inf'), 3, float('inf')], labels=['gpa < 3', 'gpa ≥ 3'])
    print(df)
    dag = {
	    'admit': ['rank', 'gre_d', 'gpa_d'],
	    'gre_d': ['rank'],
	    'gpa_d': ['rank'],
	    'rank': []
    }

    bn = BayesianNetwork(df, dag)

    # P(admit = 0 | rank = 1)
    admit_table = bn.network_tables['admit']
    admit_rank_1 = admit_table[admit_table['rank'] == 1]
    admit_no_rank_1_proba = admit_rank_1[0].sum()
    admit_yes_rank_1_proba = admit_rank_1[1].sum()
    proba_admit_0_rank_1 =  admit_no_rank_1_proba / (admit_no_rank_1_proba + admit_yes_rank_1_proba)
    print(proba_admit_0_rank_1)

    # for node, table in bn.network_tables.items():
    #     print(f"Condition Probability Table for {node}: ")
    #     print(table)


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