import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

from settings import settings
import visualization as vis
from kohonen import Kohonen

def main():
	df = pd.read_csv(settings.Config.data_dir+"/movie_data.csv", delimiter=";")

	# Analyze data and clean it
	df, std_df = data_preprocessing(df)

	# match settings.exercise:
	# 	case 1:
	# 		kohonen()
	# 	case _:
	# 		raise ValueError("Invalid exercise number")


def data_preprocessing(df):
	print(df.info())

	# Keep only `float64` columns and drop rows with missing values
	df = df.select_dtypes(include='float64').dropna()

	print(df.info())

	# Histograms + Kernel Density Estimation (KDE) of numerical variables

	# Note: a KDE plot is produced by drawing a small continuous curve (also called kernel) for every individual
	# data point along an axis, all of these curves are then added together to obtain a single smooth density
	# estimation. Unlike a histogram, KDE produces a smooth estimate.

	vis.plot_hist_kde(df, "budget", "Presupuesto en dólares")
	vis.plot_hist_kde(df, "popularity", "Popularidad")
	vis.plot_hist_kde(df, "production_companies", "Compañías productoras")
	vis.plot_hist_kde(df, "production_countries", "Países productores")
	vis.plot_hist_kde(df, "revenue", "Ingresos en dólares")
	vis.plot_hist_kde(df, "runtime", "Duración en minutos")
	vis.plot_hist_kde(df, "spoken_languages", "Idiomas")
	vis.plot_hist_kde(df, "vote_average", "Voto Promedio")
	vis.plot_hist_kde(df, "vote_count", "Cantidad de votos")

	# Standardize variables
	scaler = StandardScaler()
	std_df = scaler.fit_transform(df)
	std_df = pd.DataFrame(data=std_df, columns=df.columns.values)

	# Boxplot
	vis.boxplot(std_df, standardized=True)

	return (df, std_df)


def kohonen(df):
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
	vis.u_matrix(umatrix)


if __name__ == "__main__":
    main()
