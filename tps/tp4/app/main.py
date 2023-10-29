import pandas as pd
import numpy as np

from kohonen import Kohonen
from visualization import boxplot, biplot, component_barplot, country_heatmap, u_matrix, variable_value_scatter
from settings import settings

def main():
	match settings.exercise:
		case 1:
			kohonen()
		case _:
			raise ValueError("Invalid exercise number")
		

def kohonen():
	df = pd.read_csv(settings.Config.data_dir+"/movie_data.csv", delimiter=";")
	genres = df["genres"]
	variables_data = df.drop(columns=["genres"])
	variables = variables_data.to_numpy()
	standardized_vars = (variables - np.mean(variables, axis=0)) / np.std(variables, axis=0)

	print(df.head)

	k = 4
	epochs = 10_000

	kohonen = Kohonen(k, standardized_vars)
	kohonen.train(epochs)

	umatrix = kohonen.get_umatrix()			  	 # get u matrix

	# Plot results
	u_matrix(umatrix)
	pass


if __name__ == "__main__":
    main()
